"""
情绪干预模块
通过钩子机制在模型前向传播中修改激活值
"""

import torch
from typing import Optional
from dataclasses import dataclass


@dataclass
class InterventionConfig:
    """干预配置"""
    layer_idx: int  # 干预层索引
    strength: float  # 干预强度（正数增强，负数抑制）
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
        apply_to_all_tokens: bool = True
    ):
        """
        初始化干预器
        
        Args:
            layer_idx: 干预层索引
            emotion_vector: 情绪向量 (hidden_size,)
            strength: 干预强度（正数增强，负数抑制）
            apply_to_all_tokens: 是否应用到所有 token
        """
        self.layer_idx = layer_idx
        self.emotion_vector = emotion_vector.detach().clone()
        self.strength = strength
        self.apply_to_all_tokens = apply_to_all_tokens
        self.hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self._device = None
    
    def to(self, device: torch.device) -> 'EmotionIntervention':
        """将情绪向量移到指定设备"""
        self.emotion_vector = self.emotion_vector.to(device)
        self._device = device
        return self
    
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
        
        # 获取形状信息
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 创建干预向量
        if self.apply_to_all_tokens:
            # 扩展到 batch 和 seq 维度，应用到所有 token
            intervention = emotion_vector * self.strength
            intervention = intervention.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        else:
            # 只应用到最后一个 token
            intervention = emotion_vector * self.strength
            intervention = intervention.unsqueeze(0).expand(batch_size, -1)
            # 只在最后一个位置应用
            full_intervention = torch.zeros_like(hidden_states)
            full_intervention[:, -1, :] = intervention
            intervention = full_intervention
        
        # 应用干预
        modified_hidden = hidden_states + intervention
        
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
        print(f"✅ 已注册情绪干预钩子到第 {self.layer_idx} 层")
    
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
                f"vector_shape={tuple(self.emotion_vector.shape)})")


class MultiEmotionIntervention:
    """
    多情绪干预器
    支持同时应用多个情绪向量，每个情绪有独立的强度
    """
    
    def __init__(self, layer_idx: int):
        """
        初始化多情绪干预器
        
        Args:
            layer_idx: 干预层索引
        """
        self.layer_idx = layer_idx
        self.interventions: dict = {}  # {emotion: (vector, strength)}
        self.hook = None
    
    def add_emotion(self, emotion: str, vector: torch.Tensor, strength: float = 1.0) -> None:
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
    
    def hook_fn(self, module, input: tuple, output: tuple) -> tuple:
        """合并多个情绪向量的钩子函数"""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 合并所有情绪向量
        combined_intervention = torch.zeros(hidden_size, device=hidden_states.device)
        
        for emotion, (vector, strength) in self.interventions.items():
            if vector.device != hidden_states.device:
                vector = vector.to(hidden_states.device)
            combined_intervention += vector * strength
        
        # 扩展到 batch 和 seq 维度
        if combined_intervention.norm() > 0:
            intervention = combined_intervention.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
            modified_hidden = hidden_states + intervention
        else:
            modified_hidden = hidden_states
        
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
        print(f"✅ 已注册多情绪干预钩子，共 {len(self.interventions)} 个情绪")
    
    def remove(self) -> None:
        """移除干预钩子"""
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
            print("✅ 已移除多情绪干预钩子")
