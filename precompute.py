"""
情绪向量预计算脚本
一次性计算所有情绪向量并存储到 FAISS 向量数据库
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from config import (
    MODEL_NAME, DEVICE, EMOTIONS,
    NUM_SAMPLES_PER_EMOTION, NUM_NEUTRAL_SAMPLES,
    INTERVENTION_LAYER, INTERVENTION_STRENGTH,
    VECTOR_DB_PATH, META_DATA_PATH, VICTORS_JSON_PATH,
    MODEL_ID, USE_4BIT_QUANTIZATION,MAX_NEW_TOKENS,
    GENERATION_TEMPERATURE, GENERATION_TOP_P
)
from extractor import (
    create_quantization_config,
    generate_emotion_stories,
    get_model_activations,
    compute_emotion_vector
)
from vector_db import EmotionVectorDB


def load_model_and_tokenizer(model_name: str) -> tuple:
    """
    加载模型和分词器
    
    Args:
        model_name: 模型名称
        
    Returns:
        (tokenizer, model) 元组
    """
    print("="*60)
    print("加载模型和分词器")
    print("="*60)
    print(f"模型：{model_name}")
    print(f"设备：{DEVICE}")
    print(f"4 位量化：{USE_4BIT_QUANTIZATION}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    # 设置填充 token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 创建量化配置
    bnb_config = create_quantization_config()
    
    # 仅加载文本解码模块，不加载视觉模块以节省显存
    load_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        # 仅加载文本部分，跳过视觉编码器
        "attn_implementation": "flash_attention_2" if torch.cuda.is_available() else "eager",
    }
    
    if bnb_config:
        print("使用 4 位量化加载模型（仅文本解码模块）...")
        load_kwargs["quantization_config"] = bnb_config
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
    else:
        print("使用全精度加载模型（仅文本解码模块）...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
    
    # 如果模型包含视觉模块，显式卸载以释放显存
    if hasattr(model, "vision_tower"):
        print("检测到视觉模块，已跳过加载...")
    if hasattr(model, "vision_model"):
        print("检测到视觉模型，已跳过加载...")
    if hasattr(model, "vision_encoder"):
        print("检测到视觉编码器，已跳过加载...")
    
    model.eval()
    # 通用适配方案
    if hasattr(model.config, "hidden_size"):
        hidden_size = model.config.hidden_size
    elif hasattr(model.config, "decoder") and hasattr(model.config.decoder, "hidden_size"):
        hidden_size = model.config.decoder.hidden_size
    elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        hidden_size = model.config.text_config.hidden_size
    elif hasattr(model.config, "llm_config") and hasattr(model.config.llm_config, "hidden_size"):
        hidden_size = model.config.llm_config.hidden_size
    else:
       raise AttributeError("无法找到 hidden_size 参数，请检查模型配置")

    print(f"✅ 模型加载完成，隐藏层维度：{hidden_size}")
    
    return tokenizer, model


def generate_neutral_stories(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    num_samples: int
) -> list:
    """
    生成中性故事（使用 chat_template）
    
    Args:
        tokenizer: 分词器
        model: 语言模型
        num_samples: 样本数量
        
    Returns:
        中性故事列表
    """
    print(f"\n生成 {num_samples} 个中性故事...")
    
    # 使用 chat 格式的消息
    messages = [
        {"role": "user", "content": "Write a story (5-10 sentences) about a character doing a normal daily activity. No strong emotions."}
    ]
    
    # 使用 chat_template 编码
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=DEVICE,
        pad_token_id=tokenizer.eos_token_id
    )
    
    stories = []
    for i in tqdm(range(num_samples), desc="Generating neutral stories"):
        output = generator(
            input_text,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            do_sample=True
        )
        generated_text = output[0]["generated_text"]
        # 提取助手回复部分（去掉系统提示和用户消息）
        story = generated_text[len(input_text):].strip()
        # 清理可能的特殊 token
        story = story.replace("</s>", "").strip()
        print(story)
        stories.append(story)
    
    print(f"✅ 生成 {len(stories)} 个中性故事完成")
    return stories


def precompute_emotion_vectors(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    emotions: list,
    neutral_texts: list,
    layer_idx: int,
    samples_per_emotion: int
) -> dict:
    """
    预计算所有情绪向量
    
    Args:
        tokenizer: 分词器
        model: 语言模型
        emotions: 情绪列表
        neutral_texts: 中性文本列表
        layer_idx: 干预层索引
        samples_per_emotion: 每种情绪的样本数
        
    Returns:
        情绪向量字典
    """
    print("\n" + "="*60)
    print("开始预计算情绪向量")
    print("="*60)
    print(f"干预层：{layer_idx}")
    print(f"情绪数量：{len(emotions)}")
    print(f"每种情绪样本数：{samples_per_emotion}")
    print(f"中性样本数：{len(neutral_texts)}")
    
    # 1. 提取中性样本激活
    print(f"\n[1/3] 提取中性样本激活值...")
    neutral_activations = get_model_activations(model, tokenizer, neutral_texts, layer_idx)
    print(f"   中性激活形状：{neutral_activations.shape}")
    
    # 2. 提取每种情绪的激活并计算向量
    print(f"\n[2/3] 提取情绪激活并计算向量...")
    emotion_vectors = {}
    
    for emotion in emotions:
        print(f"\n   处理：{emotion}")
        
        # 生成情绪故事
        print(f"   - 生成 {samples_per_emotion} 个故事...")
        emotion_stories = generate_emotion_stories(tokenizer, model, emotion, samples_per_emotion)
        
        # 提取激活
        print(f"   - 提取激活值...")
        emotion_acts = get_model_activations(model, tokenizer, emotion_stories, layer_idx)
        
        # 计算情绪向量
        print(f"   - 计算情绪向量...")
        emotion_vectors[emotion] = compute_emotion_vector(emotion_acts, neutral_activations)
        
        norm = torch.norm(emotion_vectors[emotion])
        print(f"   ✅ {emotion} 完成，范数：{norm:.4f}")
    
    print(f"\n[3/3] 所有情绪向量计算完成！")
    return emotion_vectors


def save_emotion_vectors(
    emotion_vectors: dict,
    model: AutoModelForCausalLM,
    model_id: str,
    layer_idx: int,
    samples_per_emotion: int
) -> None:
    """
    保存情绪向量到向量数据库
    
    Args:
        emotion_vectors: 情绪向量字典
        model: 语言模型
        model_id: 模型唯一标识
        layer_idx: 干预层索引
        samples_per_emotion: 样本数量
    """
    print("\n" + "="*60)
    print("保存情绪向量")
    print("="*60)
    
        
    if hasattr(model.config, "hidden_size"):
        hidden_dim = model.config.hidden_size
    elif hasattr(model.config, "decoder") and hasattr(model.config.decoder, "hidden_size"):
        hidden_dim = model.config.decoder.hidden_size
    elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        hidden_dim = model.config.text_config.hidden_size
    elif hasattr(model.config, "llm_config") and hasattr(model.config.llm_config, "hidden_size"):
        hidden_dim = model.config.llm_config.hidden_size
    else:
        raise AttributeError("无法找到 hidden_size 参数，请检查模型配置")

    
    # 使用上下文管理器，自动处理加载和保存
    with EmotionVectorDB(
        vector_dim=hidden_dim,
        db_path=VECTOR_DB_PATH,
        meta_path=META_DATA_PATH,
        backup_path=VICTORS_JSON_PATH
    ) as vec_db:
        # 添加向量（overwrite=True 会自动覆盖已存在的同模型向量）
        vec_db.add_vectors(
            vectors_dict=emotion_vectors,
            model_id=model_id,
            layer_idx=layer_idx,
            sample_count=samples_per_emotion,
            overwrite=True
        )
        
        # 打印统计信息
        stats = vec_db.get_stats()
        print("\n" + "="*60)
        print("保存完成！")
        print("="*60)
        print(f"向量数据库：{VECTOR_DB_PATH}")
        print(f"元数据文件：{META_DATA_PATH}")
        print(f"JSON 备份：{VICTORS_JSON_PATH}")
        print(f"模型标识：{model_id}")
        print(f"向量维度：{hidden_dim}")
        print(f"情绪数量：{stats['total_vectors']}")


def main():
    """主函数：执行完整的预计算流程"""
    print("\n" + "="*60)
    print("情绪向量预计算脚本")
    print("="*60)
    print(f"模型：{MODEL_NAME}")
    print(f"设备：{DEVICE}")
    print(f"情绪列表：{EMOTIONS}")
    print(f"干预层：{INTERVENTION_LAYER}")
    print(f"模型标识：{MODEL_ID}")
    
    # 检查是否已存在向量数据库
    if VECTOR_DB_PATH.exists() and META_DATA_PATH.exists():
        print("\n⚠️ 警告：已存在向量数据库文件")
        print("   本次运行将覆盖现有数据")
        input("   按回车键继续...")
    
    # 1. 加载模型
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME)
    
    # 2. 生成中性故事
    print("\n" + "="*60)
    print("生成中性故事")
    print("="*60)
    neutral_stories = generate_neutral_stories(
        tokenizer, model, NUM_NEUTRAL_SAMPLES
    )
    
    # 3. 预计算情绪向量
    print("\n" + "="*60)
    print("预计算情绪向量")
    print("="*60)
    emotion_vectors = precompute_emotion_vectors(
        tokenizer, model, EMOTIONS,
        neutral_stories, INTERVENTION_LAYER,
        NUM_SAMPLES_PER_EMOTION
    )
    
    # 4. 保存情绪向量
    save_emotion_vectors(
        emotion_vectors, model, MODEL_ID,
        INTERVENTION_LAYER, NUM_SAMPLES_PER_EMOTION
    )
    
    print("\n" + "="*60)
    print("✅ 预计算完成！")
    print("="*60)
    print("\n后续推理时，直接运行 inference.py 即可加载预计算的向量")


if __name__ == "__main__":
    main()
