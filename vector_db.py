"""
情绪向量数据库工具类
使用 FAISS 实现本地轻量向量存储/加载

提供多样的数据管理接口，支持：
- 上下文管理器模式
- 批量操作
- 版本管理
- 数据验证
"""

import torch
import faiss
import json
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from datetime import datetime


class VectorDBError(Exception):
    """向量数据库异常基类"""
    pass


class VectorNotFoundError(VectorDBError):
    """向量未找到异常"""
    pass


class VectorDimensionError(VectorDBError):
    """向量维度不匹配异常"""
    pass


class MetadataError(VectorDBError):
    """元数据错误异常"""
    pass


@dataclass
class VectorMetadata:
    """向量元数据"""
    emotion: str
    model_id: str
    layer_idx: int
    sample_count: int
    created_at: str = ""
    vector_hash: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class EmotionVectorDB:
    """
    情绪向量数据库
    
    支持存储、加载、检索预计算的情绪向量。
    使用上下文管理器模式，支持多样的资源管理。
    
    Example:
        # 方式 1: 上下文管理器
        with EmotionVectorDB(vector_dim=4096, db_path=path) as db:
            db.save_vectors(vectors, model_id="xxx")
        
        # 方式 2: 显式加载/保存
        db = EmotionVectorDB(vector_dim=4096, db_path=path)
        db.load()
        vectors = db.get_vectors(model_id="xxx")
        db.save()
    """
    
    def __init__(self, vector_dim: int, db_path: Optional[Path] = None,
                 meta_path: Optional[Path] = None, backup_path: Optional[Path] = None):
        """
        初始化向量数据库
        
        Args:
            vector_dim: 向量维度（模型隐藏层维度）
            db_path: 向量数据库文件路径
            meta_path: 元数据文件路径
            backup_path: JSON 备份文件路径
        """
        if vector_dim <= 0:
            raise ValueError(f"向量维度必须大于 0，当前：{vector_dim}")
        
        self.vector_dim = vector_dim
        self.db_path = db_path or Path("emotion_vectors.faiss")
        self.meta_path = meta_path or Path("emotion_meta.json")
        self.backup_path = backup_path or Path("emotion_vectors.json")
        
        # 内部状态
        self._index: Optional[faiss.Index] = None
        self._meta_list: List[Dict] = []
        self._vectors_cache: Dict[Tuple[str, str], np.ndarray] = {}  # (model_id, emotion) -> vector
        self._is_loaded = False
    
    # ====================== 上下文管理器 ======================
    def __enter__(self) -> "EmotionVectorDB":
        """进入上下文时加载数据库"""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文时保存数据库"""
        if exc_type is None:
            self.save()
        return False
    
    # ====================== 1. 加载/保存 ======================
    def load(self) -> "EmotionVectorDB":
        """
        从磁盘加载数据库
        
        Returns:
            self
        """
        if self._is_loaded:
            return self
        
        if not self.db_path.exists() or not self.meta_path.exists():
            # 初始化空数据库
            self._index = faiss.IndexFlatL2(self.vector_dim)
            self._meta_list = []
            self._vectors_cache = {}
            self._is_loaded = True
            return self
        
        # 加载索引
        self._index = faiss.read_index(str(self.db_path))
        
        # 加载元数据
        try:
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self._meta_list = json.load(f)
        except json.JSONDecodeError as e:
            raise MetadataError(f"元数据文件损坏：{e}")
        
        # 缓存所有向量
        self._vectors_cache = {}
        for idx, meta in enumerate(self._meta_list):
            key = (meta["model_id"], meta["emotion"])
            vec_np = self._index.reconstruct(idx)
            self._vectors_cache[key] = vec_np
        
        self._is_loaded = True
        print(f"✅ 数据库加载完成：{len(self._meta_list)} 个向量")
        return self
    
    def save(self) -> "EmotionVectorDB":
        """
        保存数据库到磁盘
        
        Returns:
            self
        """
        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 重建索引（确保顺序与元数据一致）
        self._index = faiss.IndexFlatL2(self.vector_dim)
        
        vectors_to_add = []
        for meta in self._meta_list:
            key = (meta["model_id"], meta["emotion"])
            if key in self._vectors_cache:
                vectors_to_add.append(self._vectors_cache[key])
        
        if vectors_to_add:
            all_vectors = np.vstack(vectors_to_add)
            self._index.add(all_vectors)
        
        # 保存索引
        faiss.write_index(self._index, str(self.db_path))
        
        # 保存元数据
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta_list, f, ensure_ascii=False, indent=2)
        
        # 保存 JSON 备份
        self._save_json_backup()
        
        self._is_loaded = True
        print(f"✅ 数据库保存完成：{len(self._meta_list)} 个向量")
        return self
    
    # ====================== 2. 存储向量 ======================
    def add_vectors(self, vectors_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
                    model_id: str, layer_idx: int = 15,
                    sample_count: int = 20, overwrite: bool = False) -> int:
        """
        添加情绪向量到数据库
        
        Args:
            vectors_dict: 情绪向量字典 {emotion: tensor/array}
            model_id: 模型唯一标识
            layer_idx: 干预层索引
            sample_count: 样本数量
            overwrite: 是否覆盖已存在的同模型向量
            
        Returns:
            添加的向量数量
        """
        if not self._is_loaded:
            self.load()
        
        # 如果存在同模型向量且不允许覆盖，先删除
        if not overwrite:
            existing = self._find_existing(model_id)
            if existing:
                raise VectorDBError(
                    f"模型 '{model_id}' 的向量已存在。"
                    f"设置 overwrite=True 以覆盖，或先调用 delete_by_model() 删除。"
                )
        else:
            self.delete_by_model(model_id)
        
        added_count = 0
        for emotion, vector in vectors_dict.items():
            # 转换并验证向量
            vec_np = self._convert_vector(vector, emotion)
            
            # 计算向量哈希（用于完整性校验）
            vector_hash = hashlib.md5(vec_np.tobytes()).hexdigest()[:12]
            
            # 创建元数据
            meta = asdict(VectorMetadata(
                emotion=emotion,
                model_id=model_id,
                layer_idx=layer_idx,
                sample_count=sample_count,
                vector_hash=vector_hash
            ))
            
            # 缓存向量
            key = (model_id, emotion)
            self._vectors_cache[key] = vec_np
            self._meta_list.append(meta)
            added_count += 1
        
        print(f"✅ 已添加 {added_count} 个情绪向量")
        return added_count
    
    def update_vectors(self, vectors_dict: Dict[str, Union[torch.Tensor, np.ndarray]],
                       model_id: str) -> int:
        """
        更新指定模型的向量（仅更新存在的情绪，添加新情绪）
        
        Args:
            vectors_dict: 情绪向量字典
            model_id: 模型唯一标识
            
        Returns:
            更新的向量数量
        """
        if not self._is_loaded:
            self.load()
        
        # 获取现有元数据
        existing_meta = {m["emotion"]: m for m in self._meta_list
                        if m["model_id"] == model_id}
        
        updated_count = 0
        for emotion, vector in vectors_dict.items():
            vec_np = self._convert_vector(vector, emotion)
            key = (model_id, emotion)
            
            # 更新哈希
            vector_hash = hashlib.md5(vec_np.tobytes()).hexdigest()[:12]
            
            # 更新缓存
            self._vectors_cache[key] = vec_np
            
            # 更新元数据
            if emotion in existing_meta:
                existing_meta[emotion]["vector_hash"] = vector_hash
                existing_meta[emotion]["updated_at"] = datetime.now().isoformat()
            else:
                new_meta = asdict(VectorMetadata(
                    emotion=emotion,
                    model_id=model_id,
                    layer_idx=15,  # 从现有元数据获取
                    sample_count=20,
                    vector_hash=vector_hash
                ))
                self._meta_list.append(new_meta)
            
            updated_count += 1
        
        print(f"✅ 已更新 {updated_count} 个情绪向量")
        return updated_count
    
    # ====================== 3. 获取向量 ======================
    def get_vectors(self, model_id: str) -> Dict[str, torch.Tensor]:
        """
        获取指定模型的所有情绪向量
        
        Args:
            model_id: 模型唯一标识
            
        Returns:
            情绪向量字典 {emotion: tensor}
        """
        if not self._is_loaded:
            self.load()
        
        vectors_dict = {}
        for meta in self._meta_list:
            if meta["model_id"] == model_id:
                key = (model_id, meta["emotion"])
                if key in self._vectors_cache:
                    vectors_dict[meta["emotion"]] = torch.tensor(
                        self._vectors_cache[key], dtype=torch.float32
                    )
        
        if not vectors_dict:
            raise VectorNotFoundError(f"未找到模型 '{model_id}' 的向量")
        
        return vectors_dict
    
    def get_vector(self, model_id: str, emotion: str) -> torch.Tensor:
        """
        获取指定模型和情绪的单个向量
        
        Args:
            model_id: 模型唯一标识
            emotion: 情绪名称
            
        Returns:
            情绪向量 tensor
        """
        vectors = self.get_vectors(model_id)
        if emotion not in vectors:
            raise VectorNotFoundError(
                f"未找到模型 '{model_id}' 的情绪 '{emotion}'"
            )
        return vectors[emotion]
    
    # ====================== 4. 搜索 ======================
    def search(self, query_vector: Union[torch.Tensor, np.ndarray],
               k: int = 3, model_id: Optional[str] = None) -> List[Dict]:
        """
        搜索最相似的向量
        
        Args:
            query_vector: 查询向量
            k: 返回结果数量
            model_id: 可选，限制搜索特定模型
            
        Returns:
            搜索结果列表 [{emotion, model_id, distance, ...}, ...]
        """
        if not self._is_loaded:
            self.load()
        
        query_np = self._convert_vector(query_vector, "query").reshape(1, -1)
        
        # 获取搜索范围
        search_meta = self._meta_list
        if model_id:
            search_meta = [m for m in self._meta_list if m["model_id"] == model_id]
        
        if not search_meta:
            return []
        
        # 执行搜索
        distances, indices = self._index.search(query_np, min(k, len(search_meta)))
        
        results = []
        meta_map = {i: m for i, m in enumerate(self._meta_list)}
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self._meta_list):
                meta = meta_map[idx]
                if model_id is None or meta["model_id"] == model_id:
                    results.append({
                        "emotion": meta["emotion"],
                        "model_id": meta["model_id"],
                        "distance": float(dist),
                        "layer_idx": meta["layer_idx"],
                        "sample_count": meta["sample_count"]
                    })
        
        return results
    
    # ====================== 5. 删除 ======================
    def delete_by_model(self, model_id: str) -> int:
        """
        删除指定模型的所有向量
        
        Args:
            model_id: 模型唯一标识
            
        Returns:
            删除的向量数量
        """
        if not self._is_loaded:
            self.load()
        
        # 统计要删除的数量
        delete_count = sum(1 for m in self._meta_list if m["model_id"] == model_id)
        
        if delete_count == 0:
            print(f"⚠️ 未找到模型 '{model_id}' 的向量")
            return 0
        
        # 删除元数据和缓存
        self._meta_list = [m for m in self._meta_list if m["model_id"] != model_id]
        self._vectors_cache = {
            k: v for k, v in self._vectors_cache.items()
            if k[0] != model_id
        }
        
        print(f"✅ 已删除模型 '{model_id}' 的 {delete_count} 个向量")
        return delete_count
    
    def delete_by_emotion(self, emotion: str, model_id: Optional[str] = None) -> int:
        """
        删除指定情绪的向量
        
        Args:
            emotion: 情绪名称
            model_id: 可选，限制删除特定模型
            
        Returns:
            删除的向量数量
        """
        if not self._is_loaded:
            self.load()
        
        def should_delete(meta):
            if meta["emotion"] != emotion:
                return False
            if model_id and meta["model_id"] != model_id:
                return False
            return True
        
        delete_count = sum(1 for m in self._meta_list if should_delete(m))
        
        if delete_count == 0:
            print(f"⚠️ 未找到匹配的向量")
            return 0
        
        self._meta_list = [m for m in self._meta_list if not should_delete(m)]
        self._vectors_cache = {
            k: v for k, v in self._vectors_cache.items()
            if k[1] != emotion or (model_id and k[0] != model_id)
        }
        
        print(f"✅ 已删除 {delete_count} 个 '{emotion}' 向量")
        return delete_count
    
    # ====================== 6. 查询 ======================
    def list_models(self) -> List[str]:
        """获取所有模型 ID 列表"""
        if not self._is_loaded:
            self.load()
        return list(set(m["model_id"] for m in self._meta_list))
    
    def list_emotions(self, model_id: Optional[str] = None) -> List[str]:
        """
        获取所有情绪列表
        
        Args:
            model_id: 可选，限制特定模型
        """
        if not self._is_loaded:
            self.load()
        
        if model_id:
            return list(set(
                m["emotion"] for m in self._meta_list
                if m["model_id"] == model_id
            ))
        return list(set(m["emotion"] for m in self._meta_list))
    
    def exists(self, model_id: str) -> bool:
        """检查模型是否存在"""
        if not self._is_loaded:
            self.load()
        return any(m["model_id"] == model_id for m in self._meta_list)
    
    def get_stats(self) -> Dict:
        """获取数据库统计信息"""
        if not self._is_loaded:
            self.load()
        
        return {
            "vector_dim": self.vector_dim,
            "total_vectors": len(self._meta_list),
            "models": self.list_models(),
            "emotions": self.list_emotions(),
            "vectors_per_model": {
                model_id: sum(1 for m in self._meta_list if m["model_id"] == model_id)
                for model_id in self.list_models()
            }
        }
    
    # ====================== 内部方法 ======================
    def _convert_vector(self, vector: Union[torch.Tensor, np.ndarray],
                        name: str) -> np.ndarray:
        """
        转换向量为 numpy 数组并验证维度
        
        Args:
            vector: 输入向量（tensor 或 array）
            name: 向量名称（用于错误信息）
            
        Returns:
            float32 numpy 数组
        """
        if isinstance(vector, torch.Tensor):
            vec_np = vector.float().cpu().numpy()
        else:
            vec_np = vector.astype(np.float32)
        
        # 展平为一维
        vec_np = vec_np.flatten()
        
        # 验证维度
        if vec_np.shape[0] != self.vector_dim:
            raise VectorDimensionError(
                f"向量 '{name}' 维度不匹配：期望 {self.vector_dim}, 实际 {vec_np.shape[0]}"
            )
        
        return vec_np
    
    def _find_existing(self, model_id: str) -> List[Dict]:
        """查找已存在的模型向量"""
        return [m for m in self._meta_list if m["model_id"] == model_id]
    
    def _save_json_backup(self):
        """保存 JSON 格式备份"""
        backup_data = {}
        for meta in self._meta_list:
            emotion = meta["emotion"]
            model_id = meta["model_id"]
            
            # 按模型分组
            if model_id not in backup_data:
                backup_data[model_id] = {
                    "metadata": {
                        "layer_idx": meta["layer_idx"],
                        "sample_count": meta["sample_count"],
                        "created_at": meta.get("created_at", "")
                    },
                    "vectors": {}
                }
            
            # 添加向量
            key = (model_id, emotion)
            if key in self._vectors_cache:
                backup_data[model_id]["vectors"][emotion] = self._vectors_cache[key].tolist()
        
        # 写入文件
        self.backup_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.backup_path, "w", encoding="utf-8") as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)
    
    def __len__(self) -> int:
        """返回向量数量"""
        return len(self._meta_list)
    
    def __contains__(self, model_id: str) -> bool:
        """检查是否包含指定模型"""
        return self.exists(model_id)
    
    def __repr__(self) -> str:
        models = self.list_models() if self._is_loaded else []
        return f"EmotionVectorDB(dim={self.vector_dim}, vectors={len(self)}, models={models})"
