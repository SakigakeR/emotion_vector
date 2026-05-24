"""
情绪干预模块
通过钩子机制在模型前向传播中修改激活值
"""

import torch
from typing import Optional, Literal
from dataclasses import dataclass
from enum import Enum


class OperatorType(Enum):
    """干预算子类型"""
    ADD = "add"  # 简单向量加法
    PROJECTION_ENHANCE = "projection_enhance"  # 投影增强
    PROJECTION_ELIMINATE = "projection_eliminate"  # 投影消除


@dataclass
class InterventionConfig:
    """干预配置"""
    layer_idx: int  # 干预层索引
    strength: float  # 干预强度（正数增强，负数抑制）
    operator_type: OperatorType = OperatorType.ADD  # 干预算子类型
    projection_strength: float = 1.0  # 投影算子的强度系数（仅用于投影类算子）
    apply_to_all_tokens: bool = True  # 是否应用到所有 token


class EmotionIntervention:
    """
    情绪向量干预器
    在模型前向传播中修改激活值，实现情绪增强或抑制
    """
    
    def __init__(
        self,
        layer_idx: int,
        emotion_vector: torch.Tensor,
        strength: float = 1.0,
        operator_type: OperatorType = OperatorType.ADD,
        projection_strength: float = 1.0,
        apply_to_all_tokens: bool = True
    ):
        """
        初始化干预器
        
        Args:
            layer_idx: 干预层索引
            emotion_vector: 情绪向量 (hidden_size,)
            strength: 干预强度（正数增强，负数抑制）
            operator_type: 干预算子类型
            projection_strength: 投影算子的强度系数
            apply_to_all_tokens: 是否应用到所有 token
        """
        self.layer_idx = layer_idx
        self.emotion_vector = emotion_vector.detach().clone()
        self.strength = strength
        self.operator_type = operator_type
        self.projection_strength = projection_strength
        self.apply_to_all_tokens = apply_to_all_tokens
        self.hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self._device = None
    
    def to(self, device: torch.device) -> 'EmotionIntervention':
        """将情绪向量移到指定设备"""
        self.emotion_vector = self.emotion_vector.to(device)
        self._device = device
        return self
    
    def _apply_add_operator(self, hidden_states: torch.Tensor, emotion_vector: torch.Tensor) -> torch.Tensor:
        """
        简单向量加法算子
        
        Args:
            hidden_states: 隐藏状态 (batch_size, seq_len, hidden_size)
            emotion_vector: 情绪向量 (hidden_size,)
            
        Returns:
            修改后的隐藏状态
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if self.apply_to_all_tokens:
            intervention = emotion_vector * self.strength
            intervention = intervention.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        else:
            intervention = emotion_vector * self.strength
            intervention = intervention.unsqueeze(0).expand(batch_size, -1)
            full_intervention = torch.zeros_like(hidden_states)
            full_intervention[:, -1, :] = intervention
            intervention = full_intervention
        
        return hidden_states + intervention
    
    def _apply_projection_enhance(self, hidden_states: torch.Tensor, emotion_vector: torch.Tensor) -> torch.Tensor:
        """
        投影增强算子
        
        将隐藏状态投影到情绪向量方向，然后增强该方向的分量。
        公式：h' = h + strength * projection_strength * proj_e(h)
        其中 proj_e(h) = (h · e) / (e · e) * e 是 h 在 e 方向上的投影
        
        Args:
            hidden_states: 隐藏状态 (batch_size, seq_len, hidden_size)
            emotion_vector: 情绪向量 (hidden_size,)
            
        Returns:
            修改后的隐藏状态
        """
        # 计算情绪向量的范数平方
        e_dot_e = torch.sum(emotion_vector * emotion_vector) + 1e-8
        
        # 计算隐藏状态在情绪向量方向上的投影系数 (batch_size, seq_len)
        # h · e
        h_dot_e = torch.sum(hidden_states * emotion_vector, dim=-1)
        
        # 投影系数除以 e·e 得到缩放因子
        projection_coeff = h_dot_e / e_dot_e
        
        # 重建投影向量 (batch_size, seq_len, hidden_size)
        projection = projection_coeff.unsqueeze(-1) * emotion_vector.unsqueeze(0).unsqueeze(0)
        
        # 增强投影方向的分量
        enhancement = projection * self.strength * self.projection_strength
        
        return hidden_states + enhancement
    
    def _apply_projection_eliminate(self, hidden_states: torch.Tensor, emotion_vector: torch.Tensor) -> torch.Tensor:
        """
        投影消除算子
        
        将隐藏状态投影到情绪向量方向，然后消除该方向的分量。
        公式：h' = h - strength * projection_strength * proj_e(h)
        其中 proj_e(h) = (h · e) / (e · e) * e 是 h 在 e 方向上的投影
        
        Args:
            hidden_states: 隐藏状态 (batch_size, seq_len, hidden_size)
            emotion_vector: 情绪向量 (hidden_size,)
            
        Returns:
            修改后的隐藏状态
        """
        # 计算情绪向量的范数平方
        e_dot_e = torch.sum(emotion_vector * emotion_vector) + 1e-8
        
        # 计算隐藏状态在情绪向量方向上的投影系数 (batch_size, seq_len)
        h_dot_e = torch.sum(hidden_states * emotion_vector, dim=-1)
        
        # 投影系数除以 e·e 得到缩放因子
        projection_coeff = h_dot_e / e_dot_e
        
        # 重建投影向量 (batch_size, seq_len, hidden_size)
        projection = projection_coeff.unsqueeze(-1) * emotion_vector.unsqueeze(0).unsqueeze(0)
        
        # 消除投影方向的分量（使用负号）
        elimination = projection * self.strength * self.projection_strength
        
        return hidden_states - elimination
    
    def hook_fn(self, module, input: tuple, output: tuple) -> tuple:
        """
        修改激活值的钩子函数
        
        Args:
            module: 模型层模块
            input: 输入元组
            output: 输出元组
            
        Returns:
            修改后的输出元组
        """
        # 获取隐藏状态
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # 确保情绪向量在正确的设备和 dtype 上
        if self.emotion_vector.device != hidden_states.device:
            self.emotion_vector = self.emotion_vector.to(hidden_states.device)
        
        # 关键：将情绪向量转换为与隐藏状态相同的 dtype（处理 bfloat16 等情况）
        emotion_vector = self.emotion_vector.to(hidden_states.dtype)
        
        # 根据算子类型应用不同的干预
        if self.operator_type == OperatorType.ADD:
            modified_hidden = self._apply_add_operator(hidden_states, emotion_vector)
        elif self.operator_type == OperatorType.PROJECTION_ENHANCE:
            modified_hidden = self._apply_projection_enhance(hidden_states, emotion_vector)
        elif self.operator_type == OperatorType.PROJECTION_ELIMINATE:
            modified_hidden = self._apply_projection_eliminate(hidden_states, emotion_vector)
        else:
            raise ValueError(f"未知的算子类型：{self.operator_type}")
        
        # 返回修改后的输出
        if isinstance(output, tuple):
            return (modified_hidden,) + output[1:]
        return modified_hidden
    
    def register(self, model) -> None:
        """
        注册干预钩子
        
        Args:
            model: 语言模型
        """
        if self.hook is not None:
            print("⚠️ 警告：干预钩子已注册，先移除旧钩子")
            self.remove()
        if hasattr(model.model,"layers"):
            target_layer = model.model.layers[self.layer_idx]
        elif hasattr(model.model,"language_model"):
            target_layer = model.model.language_model.layers[self.layer_idx]
        self.hook = target_layer.register_forward_hook(self.hook_fn)
        print(f"✅ 已注册情绪干预钩子到第 {self.layer_idx} 层 (算子：{self.operator_type.value})")
    
    def remove(self) -> None:
        """移除干预钩子"""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
            print("✅ 已移除情绪干预钩子")
    
    def __enter__(self) -> 'EmotionIntervention':
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.remove()
    
    def __repr__(self) -> str:
        return (f"EmotionIntervention(layer={self.layer_idx}, "
                f"strength={self.strength}, "
                f"operator={self.operator_type.value}, "
                f"projection_strength={self.projection_strength}, "
                f"vector_shape={tuple(self.emotion_vector.shape)})")


class MultiEmotionIntervention:
    """
    多情绪干预器
    支持同时应用多个情绪向量，每个情绪有独立的强度和算子类型
    """
    
    def __init__(
        self,
        layer_idx: int,
        operator_type: OperatorType = OperatorType.ADD,
        projection_strength: float = 1.0,
        apply_to_all_tokens: bool = True
    ):
        """
        初始化多情绪干预器
        
        Args:
            layer_idx: 干预层索引
            operator_type: 干预算子类型
            projection_strength: 投影算子的强度系数
            apply_to_all_tokens: 是否应用到所有 token
        """
        self.layer_idx = layer_idx
        self.operator_type = operator_type
        self.projection_strength = projection_strength
        self.apply_to_all_tokens = apply_to_all_tokens
        self.interventions: dict = {}  # {emotion: (vector, strength)}
        self.hook = None
    
    def add_emotion(
        self,
        emotion: str,
        vector: torch.Tensor,
        strength: float = 1.0
    ) -> None:
        """
        添加情绪向量
        
        Args:
            emotion: 情绪名称
            vector: 情绪向量
            strength: 强度
        """
        self.interventions[emotion] = (vector.detach().clone(), strength)
    
    def remove_emotion(self, emotion: str) -> None:
        """移除指定情绪"""
        if emotion in self.interventions:
            del self.interventions[emotion]
    
    def clear(self) -> None:
        """清空所有情绪"""
        self.interventions.clear()
    
    def _apply_add_operator(
        self,
        hidden_states: torch.Tensor,
        combined_intervention: torch.Tensor
    ) -> torch.Tensor:
        """简单向量加法算子"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if self.apply_to_all_tokens:
            intervention = combined_intervention.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        else:
            intervention = combined_intervention.unsqueeze(0).expand(batch_size, -1)
            full_intervention = torch.zeros_like(hidden_states)
            full_intervention[:, -1, :] = intervention
            intervention = full_intervention
        
        return hidden_states + intervention
    
    def _apply_projection_enhance(
        self,
        hidden_states: torch.Tensor,
        combined_intervention: torch.Tensor
    ) -> torch.Tensor:
        """
        投影增强算子
        
        将隐藏状态投影到合并情绪向量方向，然后增强该方向的分量
        """
        if combined_intervention.norm() < 1e-8:
            return hidden_states
        
        # 计算合并情绪向量的范数平方
        e_dot_e = torch.sum(combined_intervention * combined_intervention) + 1e-8
        
        # 计算隐藏状态在合并情绪向量方向上的投影系数 (batch_size, seq_len)
        h_dot_e = torch.sum(hidden_states * combined_intervention, dim=-1)
        
        # 投影系数除以 e·e 得到缩放因子
        projection_coeff = h_dot_e / e_dot_e
        
        # 重建投影向量 (batch_size, seq_len, hidden_size)
        projection = projection_coeff.unsqueeze(-1) * combined_intervention.unsqueeze(0).unsqueeze(0)
        
        # 增强投影方向的分量
        enhancement = projection * self.projection_strength
        
        return hidden_states + enhancement
    
    def _apply_projection_eliminate(
        self,
        hidden_states: torch.Tensor,
        combined_intervention: torch.Tensor
    ) -> torch.Tensor:
        """
        投影消除算子
        
        将隐藏状态投影到合并情绪向量方向，然后消除该方向的分量
        """
        if combined_intervention.norm() < 1e-8:
            return hidden_states
        
        # 计算合并情绪向量的范数平方
        e_dot_e = torch.sum(combined_intervention * combined_intervention) + 1e-8
        
        # 计算隐藏状态在合并情绪向量方向上的投影系数 (batch_size, seq_len)
        h_dot_e = torch.sum(hidden_states * combined_intervention, dim=-1)
        
        # 投影系数除以 e·e 得到缩放因子
        projection_coeff = h_dot_e / e_dot_e
        
        # 重建投影向量 (batch_size, seq_len, hidden_size)
        projection = projection_coeff.unsqueeze(-1) * combined_intervention.unsqueeze(0).unsqueeze(0)
        
        # 消除投影方向的分量
        elimination = projection * self.projection_strength
        
        return hidden_states - elimination
    
    def hook_fn(self, module, input: tuple, output: tuple) -> tuple:
        """合并多个情绪向量的钩子函数"""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # 合并所有情绪向量
        combined_intervention = torch.zeros(
            hidden_states.shape[-1],
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        
        for emotion, (vector, strength) in self.interventions.items():
            if vector.device != hidden_states.device:
                vector = vector.to(hidden_states.device)
            vector = vector.to(hidden_states.dtype)
            combined_intervention += vector * strength
        
        # 根据算子类型应用不同的干预
        if self.operator_type == OperatorType.ADD:
            modified_hidden = self._apply_add_operator(hidden_states, combined_intervention)
        elif self.operator_type == OperatorType.PROJECTION_ENHANCE:
            modified_hidden = self._apply_projection_enhance(hidden_states, combined_intervention)
        elif self.operator_type == OperatorType.PROJECTION_ELIMINATE:
            modified_hidden = self._apply_projection_eliminate(hidden_states, combined_intervention)
        else:
            raise ValueError(f"未知的算子类型：{self.operator_type}")
        
        if isinstance(output, tuple):
            return (modified_hidden,) + output[1:]
        return modified_hidden
    
    def register(self, model) -> None:
        """注册干预钩子"""
        if not self.interventions:
            raise ValueError("❌ 没有添加任何情绪向量")
        if hasattr(model.model,"layers"):
            target_layer = model.model.layers[self.layer_idx]
        elif hasattr(model.model,"language_model"):
            target_layer = model.model.language.layers[self.layer_idx]
        self.hook = target_layer.register_forward_hook(self.hook_fn)
        print(f"✅ 已注册多情绪干预钩子，共 {len(self.interventions)} 个情绪 (算子：{self.operator_type.value})")
    
    def remove(self) -> None:
        """移除干预钩子"""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
            print("✅ 已移除多情绪干预钩子")
