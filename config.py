"""
情绪向量工程配置文件
包含模型配置、情绪列表、干预参数等
"""

import torch
from pathlib import Path

# ====================== 模型配置 ======================
MODEL_NAME = "./weights"  # 可替换为其他开源模型
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ====================== 情绪配置 ======================
EMOTIONS = ["happy", "sad", "angry", "fearful", "calm", "desperate"]  # 6 种核心情绪
NUM_SAMPLES_PER_EMOTION = 20  # 每种情绪样本数
NUM_NEUTRAL_SAMPLES = 50  # 中性样本数

# ====================== 干预配置 ======================
INTERVENTION_LAYER = 15  # 干预层（Mistral-7B 共 32 层，选择中间层）
INTERVENTION_STRENGTH = 3.0  # 干预强度（正数增强，负数抑制）

# ====================== 生成参数 ======================
MAX_NEW_TOKENS = 512  # 故事生成最大 token 数
GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_P = 0.95
MAX_INPUT_LENGTH = 512  # 输入文本最大长度

# ====================== 量化配置 ======================
USE_4BIT_QUANTIZATION = True  # 是否使用 4 位量化

# ====================== 文件路径配置 ======================
BASE_DIR = Path(__file__).parent
VECTOR_DB_PATH = BASE_DIR / "emotion_vectors.faiss"  # 向量数据库文件
META_DATA_PATH = BASE_DIR / "emotion_meta.json"  # 元数据文件
VICTORS_JSON_PATH = BASE_DIR / "emotion_vectors.json"  # JSON 格式备份
RESULTS_DIR = BASE_DIR / "results"  # 实验结果保存目录

# ====================== 模型唯一标识 ======================
# 绑定：模型名称 + 干预层（避免向量混用）
MODEL_ID = f"{MODEL_NAME.replace('/', '-')}-intervention-layer-{INTERVENTION_LAYER}"

# ====================== 评估配置 ======================
SENTIMENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"  # 情感分析模型
