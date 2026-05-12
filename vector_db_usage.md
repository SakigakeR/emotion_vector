# 情绪向量数据库使用指南

本文档介绍 `EmotionVectorDB` 的各种使用方式和最佳实践。

## 快速开始

### 1. 基本初始化

```python
from pathlib import Path
from vector_db import EmotionVectorDB

# 创建数据库实例
db = EmotionVectorDB(
    vector_dim=4096,           # 模型隐藏层维度
    db_path=Path("data/vectors.faiss"),
    meta_path=Path("data/meta.json"),
    backup_path=Path("data/backup.json")
)
```

### 2. 使用上下文管理器（推荐）

```python
import torch
from vector_db import EmotionVectorDB

# 创建示例向量
emotion_vectors = {
    "happy": torch.randn(4096),
    "sad": torch.randn(4096),
    "angry": torch.randn(4096),
}

# 使用上下文管理器，自动处理加载和保存
with EmotionVectorDB(vector_dim=4096) as db:
    # 添加向量（overwrite=True 会覆盖已存在的同模型向量）
    db.add_vectors(
        vectors_dict=emotion_vectors,
        model_id="my_model_v1",
        layer_idx=15,
        sample_count=20,
        overwrite=True
    )
    
    # 获取统计信息
    stats = db.get_stats()
    print(f"总向量数：{stats['total_vectors']}")
    print(f"模型列表：{stats['models']}")
```

### 3. 显式加载/保存

```python
from vector_db import EmotionVectorDB

db = EmotionVectorDB(vector_dim=4096)

# 显式加载
db.load()

# 执行操作
vectors = db.get_vectors("my_model_v1")

# 显式保存
db.save()
```

## 核心 API

### 存储向量

#### 添加新向量

```python
with EmotionVectorDB(vector_dim=4096) as db:
    db.add_vectors(
        vectors_dict={"happy": vec},
        model_id="model_v1",
        layer_idx=15,
        sample_count=20,
        overwrite=False  # False 时如果已存在会抛出异常
    )
```

#### 更新现有向量

```python
with EmotionVectorDB(vector_dim=4096) as db:
    # 仅更新存在的情绪，添加新情绪
    db.update_vectors(
        vectors_dict={"happy": new_vec, "excited": new_vec2},
        model_id="model_v1"
    )
```

### 获取向量

#### 获取模型的所有向量

```python
with EmotionVectorDB(vector_dim=4096) as db:
    vectors = db.get_vectors("model_v1")
    # vectors: {"happy": tensor, "sad": tensor, ...}
```

#### 获取单个向量

```python
with EmotionVectorDB(vector_dim=4096) as db:
    happy_vec = db.get_vector("model_v1", "happy")
```

### 搜索相似向量

```python
with EmotionVectorDB(vector_dim=4096) as db:
    # 搜索最相似的 3 个向量
    results = db.search(query_vector=some_vec, k=3)
    
    for result in results:
        print(f"情绪：{result['emotion']}, 距离：{result['distance']:.4f}")
    
    # 限制搜索特定模型
    results = db.search(query_vector=some_vec, k=3, model_id="model_v1")
```

### 删除向量

#### 按模型删除

```python
with EmotionVectorDB(vector_dim=4096) as db:
    count = db.delete_by_model("model_v1")
    print(f"删除了 {count} 个向量")
```

#### 按情绪删除

```python
with EmotionVectorDB(vector_dim=4096) as db:
    # 删除所有模型的 happy 向量
    count = db.delete_by_emotion("happy")
    
    # 仅删除特定模型的 happy 向量
    count = db.delete_by_emotion("happy", model_id="model_v1")
```

### 查询信息

```python
with EmotionVectorDB(vector_dim=4096) as db:
    # 获取所有模型 ID
    models = db.list_models()
    
    # 获取所有情绪类型
    emotions = db.list_emotions()
    
    # 获取特定模型的情绪类型
    emotions = db.list_emotions(model_id="model_v1")
    
    # 检查模型是否存在
    if "model_v1" in db:
        print("模型存在")
    
    # 获取统计信息
    stats = db.get_stats()
    # {
    #     "vector_dim": 4096,
    #     "total_vectors": 10,
    #     "models": ["model_v1", "model_v2"],
    #     "emotions": ["happy", "sad", "angry"],
    #     "vectors_per_model": {"model_v1": 5, "model_v2": 5}
    # }
```

## 异常处理

```python
from vector_db import (
    EmotionVectorDB,
    VectorNotFoundError,
    VectorDimensionError,
    VectorDBError
)

try:
    with EmotionVectorDB(vector_dim=4096) as db:
        vec = db.get_vector("nonexistent_model", "happy")
except VectorNotFoundError as e:
    print(f"向量未找到：{e}")

try:
    with EmotionVectorDB(vector_dim=4096) as db:
        db.add_vectors({"happy": torch.randn(2048)}, "model_v1")
except VectorDimensionError as e:
    print(f"维度不匹配：{e}")

try:
    with EmotionVectorDB(vector_dim=4096) as db:
        db.add_vectors({"happy": vec}, "model_v1", overwrite=False)
        db.add_vectors({"happy": vec}, "model_v1", overwrite=False)  # 会抛出异常
except VectorDBError as e:
    print(f"数据库错误：{e}")
```

## 高级用法

### 批量操作

```python
import torch
from vector_db import EmotionVectorDB

# 准备多个模型的向量
models_data = {
    "model_v1": {"happy": torch.randn(4096), "sad": torch.randn(4096)},
    "model_v2": {"happy": torch.randn(4096), "sad": torch.randn(4096)},
}

with EmotionVectorDB(vector_dim=4096) as db:
    for model_id, vectors in models_data.items():
        db.add_vectors(vectors, model_id=model_id, overwrite=True)
```

### 数据验证

```python
with EmotionVectorDB(vector_dim=4096) as db:
    # 检查数据库状态
    stats = db.get_stats()
    
    # 验证特定模型的数据完整性
    if "model_v1" in db:
        expected_emotions = {"happy", "sad", "angry"}
        actual_emotions = set(db.list_emotions(model_id="model_v1"))
        
        if expected_emotions == actual_emotions:
            print("数据完整性验证通过")
        else:
            missing = expected_emotions - actual_emotions
            print(f"缺少情绪：{missing}")
```

### 备份与恢复

```python
# 数据库会自动创建 JSON 备份
with EmotionVectorDB(
    vector_dim=4096,
    db_path=Path("data/vectors.faiss"),
    meta_path=Path("data/meta.json"),
    backup_path=Path("data/backup.json")  # 自动备份
) as db:
    db.add_vectors(vectors, "model_v1")
    # save() 时会自动创建 JSON 备份
```

## 最佳实践

1. **始终使用上下文管理器**：确保数据正确保存
2. **设置 overwrite 参数**：避免意外覆盖数据
3. **定期检查统计信息**：确保数据完整性
4. **使用异常处理**：优雅处理错误情况
5. **保留 JSON 备份**：便于调试和数据恢复

## 完整示例

```python
import torch
from pathlib import Path
from vector_db import EmotionVectorDB

def main():
    # 配置
    VECTOR_DIM = 4096
    MODEL_ID = "qwen-7b-v1"
    LAYER_IDX = 15
    SAMPLE_COUNT = 20
    
    # 创建示例向量（实际使用时替换为真实向量）
    emotion_vectors = {
        "happy": torch.randn(VECTOR_DIM),
        "sad": torch.randn(VECTOR_DIM),
        "angry": torch.randn(VECTOR_DIM),
        "fear": torch.randn(VECTOR_DIM),
    }
    
    # 保存到数据库
    with EmotionVectorDB(
        vector_dim=VECTOR_DIM,
        db_path=Path("data/emotion_vectors.faiss"),
        meta_path=Path("data/emotion_meta.json"),
        backup_path=Path("data/emotion_backup.json")
    ) as db:
        # 添加向量
        db.add_vectors(
            vectors_dict=emotion_vectors,
            model_id=MODEL_ID,
            layer_idx=LAYER_IDX,
            sample_count=SAMPLE_COUNT,
            overwrite=True
        )
        
        # 验证
        stats = db.get_stats()
        print(f"保存完成：{stats}")
    
    # 从数据库加载
    with EmotionVectorDB(vector_dim=VECTOR_DIM) as db:
        vectors = db.get_vectors(MODEL_ID)
        print(f"加载完成：{list(vectors.keys())}")
        
        # 搜索相似向量
        query = torch.randn(VECTOR_DIM)
        results = db.search(query, k=3)
        print(f"最相似的向量：{results}")

if __name__ == "__main__":
    main()
```
