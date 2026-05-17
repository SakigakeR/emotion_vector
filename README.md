# 情绪向量工程

Anthropic 情绪向量实验的本地复现与 Transformers 实践

## 项目概述

本项目实现了 Anthropic 论文《大型语言模型中的情绪概念及其功能》(Emotion Concepts and their Function in a Large Language Model) 的核心实验，可以在本地开源模型中复现情绪向量的提取和因果干预。

### 核心功能

1. **情绪向量提取**: 从模型中间层残差流中提取情绪概念向量
2. **因果干预**: 通过修改激活值直接影响模型输出的情绪表达
3. **向量持久化**: 使用 FAISS 本地向量数据库存储预计算的情绪向量
4. **效果评估**: 使用情感分析模型评估干预效果

## 项目结构

```
emotion_vector/
├── config.py           # 配置文件：模型、情绪、干预参数
├── vector_db.py        # 向量数据库工具类：FAISS 存储/加载
├── extractor.py        # 情绪向量提取模块：生成故事、提取激活
├── intervention.py     # 情绪干预模块：钩子机制修改激活值
├── precompute.py       # 预计算脚本：一次性计算并存储向量
├── inference.py        # 推理脚本：加载向量进行情绪干预
├── evaluator.py        # 评估模块：情感分析评估干预效果
├── main.py             # 主入口脚本：统一命令行接口
├── gui.py              # Gradio 图形界面
├── requirements.txt    # 项目依赖
└── PROJECT_README.md   # 本说明文档

# 运行时生成的文件
├── emotion_vectors.faiss  # FAISS 向量数据库
├── emotion_meta.json      # 向量元数据
└── emotion_vectors.json   # JSON 格式备份
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置参数

编辑 [`config.py`](config.py) 文件，修改以下参数：

```python
# 模型配置
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"  # 可替换为其他模型

# 情绪配置
EMOTIONS = ["happy", "sad", "angry", "fearful", "calm", "desperate"]

# 干预配置
INTERVENTION_LAYER = 15  # 干预层（Mistral-7B 共 32 层）
INTERVENTION_STRENGTH = 2.0  # 干预强度
```

### 3. 预计算情绪向量（首次运行）

```bash
python main.py precompute
```

此过程将：
- 加载语言模型
- 生成情绪和中性故事
- 提取模型激活值
- 计算情绪向量
- 保存到 FAISS 向量数据库

**注意**: 此过程可能需要较长时间（取决于模型大小和样本数量），请确保有足够的显存。

### 4. 运行推理实验

```bash
python main.py inference
```

此过程将：
- 加载语言模型
- 从 FAISS 数据库加载预计算的情绪向量
- 对测试提示进行情绪干预生成
- 输出对比结果

### 5. 运行评估

```bash
python main.py evaluate
```

## 使用方式

### 命令行接口

```bash
# 查看帮助
python main.py --help

# 预计算情绪向量
python main.py precompute

# 运行推理实验
python main.py inference

# 运行评估
python main.py evaluate

# 启动图形界面
python main.py gui

# 检查环境
python main.py check
```

### 图形界面 (GUI)

```bash
# 启动 Gradio 图形界面
python main.py gui
# 或
python gui.py
```

启动后，界面将在 http://localhost:7860 打开，提供以下功能：

- **系统状态**: 查看向量数据库状态和配置信息
- **预计算**: 一键运行预计算流程
- **推理实验**: 输入提示、选择情绪、调整参数进行情绪干预生成
- **结果对比**: 同时查看原始输出和干预后输出

**⚠️ 重要说明**: GUI 模式**未适配 chat_template**，仅支持**文本补全**模式。如果使用支持对话的模型（如 Instruct/Chat 模型），需要**手动构造 chat_template**作为输入提示。

例如，对于 Mistral-Instruct 模型，需要手动构造如下格式的提示：

```text
<s>[INST] 请续写以下故事：Once upon a time, there was a little girl... [/INST]
```

对于 Llama-3-Instruct 模型：

```text
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

请续写以下故事：Once upon a time...<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```

建议参考对应模型的官方文档获取正确的 chat_template 格式。

### 直接使用模块

```python
from vector_db import EmotionVectorDB
from intervention import EmotionIntervention
from config import MODEL_ID, INTERVENTION_LAYER

# 加载预计算的情绪向量
vec_db = EmotionVectorDB(vector_dim=4096)
emotion_vectors = vec_db.load_vectors(model_id=MODEL_ID)

# 创建干预器
intervention = EmotionIntervention(
    layer_idx=INTERVENTION_LAYER,
    emotion_vector=emotion_vectors["happy"],
    strength=2.0
)

# 注册到模型
intervention.register(model)

# 生成文本...

# 移除干预
intervention.remove()
```

## 核心原理

### 1. 情绪向量提取

```
情绪向量 = normalize(情绪样本激活均值 - 中性样本激活均值)
```

- 为每种情绪生成多个无明确情绪词的故事
- 收集模型中间层的残差流激活
- 计算情绪样本与中性样本的激活差异
- 归一化到单位长度

### 2. 情绪向量干预

```
修改后激活 = 原始激活 + 情绪向量 × 强度系数
```

- 在模型前向传播过程中，通过钩子机制捕获层输出
- 将情绪向量广播到 batch 和序列维度
- 加到原始激活上，实现情绪增强或抑制

## 配置说明

### 模型选择

支持的开源模型：
- `mistralai/Mistral-7B-Instruct-v0.2` 
- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- 其他支持 `AutoModelForCausalLM` 的模型

### 干预层选择

- Mistral-7B (32 层): 推荐第 15-20 层
- Llama-2-7B (32 层): 推荐第 15-20 层
- Llama-3-8B (32 层): 推荐第 15-20 层

一般选择总层数的 1/2 到 2/3 处。

### 干预强度

- 正数：增强该情绪表达
- 负数：抑制该情绪表达
- 推荐范围：取决于模型架构和参数量，详见 [`docs/INTERVENTION_GUIDE.md`](docs/INTERVENTION_GUIDE.md)

**重要说明**: 干预强度的选择与以下因素相关：
- **神经网络层数**: 不同层数的模型需要不同的干预层和强度
- **任务类型**: 故事创作、对话、情感分析等不同场景需要不同强度

请参考 [`docs/INTERVENTION_GUIDE.md`](docs/INTERVENTION_GUIDE.md) 获取详细的参数建议表。

## 优化建议

| 问题 | 解决方案 |
|------|----------|
| 情绪向量效果弱 | 增加样本数量、调整干预层、提高干预强度 |
| 模型输出不稳定 | 固定随机种子、降低 temperature |
| 显存不足 | 使用 4 位量化、换用更小模型 |
| 情绪混淆 | 增加样本多样性、使用 Gram-Schmidt 正交化 |

## 性能优化

### 4 位量化

默认启用 4 位量化，大幅减少显存占用：

```python
# config.py
USE_4BIT_QUANTIZATION = True
```

### 向量复用

情绪向量与模型 + 干预层绑定，一次计算永久复用：

```
预计算 (耗时) → 存储到 FAISS → 推理时秒级加载
```

## 故障排除

### 1. 显存不足

```python
# 使用更小的模型
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 减少样本数量
NUM_SAMPLES_PER_EMOTION = 10
NUM_NEUTRAL_SAMPLES = 20
```

### 2. 向量数据库不存在

```bash
# 先运行预计算
python main.py precompute
```

### 3. 模型加载失败

```bash
# 检查 transformers 版本
pip install --upgrade transformers

# 使用 trust_remote_code
# 已在代码中设置
```

## 扩展功能

### 1. 添加新情绪

编辑 [`config.py`](config.py):

```python
EMOTIONS = ["happy", "sad", "angry", "fearful", "calm", "desperate", "excited", "anxious"]
```

### 2. 多模型管理

每个模型有独立的 `model_id`，向量数据库自动区分：

```python
MODEL_ID = f"{MODEL_NAME.replace('/', '-')}-intervention-layer-{INTERVENTION_LAYER}"
```

### 3. 自定义评估指标

扩展 [`evaluator.py`](evaluator.py) 添加自定义评估逻辑。

## 参考文献

1. Anthropic. "Emotion Concepts and their Function in a Large Language Model"
2. Transformers Library: https://github.com/huggingface/transformers
3. FAISS: https://github.com/facebookresearch/faiss

## 许可证

本项目仅供学习和研究使用。
