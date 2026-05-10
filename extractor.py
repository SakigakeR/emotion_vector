"""
情绪向量提取模块
负责生成情绪故事、提取模型激活、计算情绪向量
"""

import torch
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig

from config import (
    DEVICE, USE_4BIT_QUANTIZATION,
    MAX_NEW_TOKENS, GENERATION_TEMPERATURE, GENERATION_TOP_P,
    MAX_INPUT_LENGTH
)


def create_quantization_config() -> BitsAndBytesConfig:
    """创建量化配置"""
    if not USE_4BIT_QUANTIZATION:
        return None
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
    )


def generate_emotion_stories(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    emotion: str,
    num_samples: int = 10
) -> List[str]:
    """
    生成无明确情绪词的情绪故事（使用 chat_template）
    
    Args:
        tokenizer: 分词器
        model: 语言模型
        emotion: 情绪类型
        num_samples: 生成样本数量
        
    Returns:
        故事列表
    """
    # 使用 chat 格式的消息
    messages = [
        {"role": "user", "content": f"""Write a short story (3-5 sentences) about a character experiencing {emotion}.
IMPORTANT: Do NOT use the word "{emotion}" or any synonyms. Show the emotion through actions, body language, and environment only."""}
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
    for i in tqdm(range(num_samples), desc=f"Generating {emotion} stories"):
        output = generator(
            input_text,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=GENERATION_TEMPERATURE,
            top_p=GENERATION_TOP_P,
            do_sample=True
        )
        # 提取助手回复部分（去掉系统提示和用户消息）
        generated_text = output[0]["generated_text"]
        story = generated_text[len(input_text):].strip()
        # 清理可能的特殊 token
        story = story.replace("</s>", "").strip()
        stories.append(story)
    
    return stories


def get_model_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    layer_idx: int
) -> torch.Tensor:
    """
    获取模型特定层的激活值
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        texts: 文本列表
        layer_idx: 目标层索引
        
    Returns:
        激活值张量 (num_texts, hidden_size)
    """
    activations = []
    hooks = []
    
    def hook_fn(module, input, output):
        """钩子函数：捕获层输出"""
        # 不同模型的输出格式可能不同
        if isinstance(output, tuple):
            # Mistral/LLaMA 等模型的输出是元组
            hidden_states = output[0]
        else:
            hidden_states = output
        
        #  detach 并移到 CPU 以节省显存
        activations.append(hidden_states.detach().cpu())
    
    # 注册钩子（针对 transformer 层）
    target_layer = model.model.layers[layer_idx]
    hook = target_layer.register_forward_hook(hook_fn)
    hooks.append(hook)
    
    # 处理文本
    for text in tqdm(texts, desc="Extracting activations"):
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=MAX_INPUT_LENGTH
        ).to(DEVICE)
        
        with torch.no_grad():
            model(**inputs)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    # 平均所有 token 的激活值，得到每个文本的向量表示
    avg_activations = [act.mean(dim=1).squeeze(0) for act in activations]
    return torch.stack(avg_activations)


def compute_emotion_vector(
    emotion_activations: torch.Tensor,
    neutral_activations: torch.Tensor
) -> torch.Tensor:
    """
    计算情绪向量：情绪激活均值 - 中性激活均值，并归一化
    
    Args:
        emotion_activations: 情绪样本激活值 (num_samples, hidden_size)
        neutral_activations: 中性样本激活值 (num_samples, hidden_size)
        
    Returns:
        归一化的情绪向量 (hidden_size,)
    """
    emotion_mean = emotion_activations.mean(dim=0)
    neutral_mean = neutral_activations.mean(dim=0)
    emotion_vector = emotion_mean - neutral_mean
    
    # 归一化到单位长度
    norm = torch.norm(emotion_vector)
    if norm > 0:
        emotion_vector = emotion_vector / norm
    
    return emotion_vector


def extract_all_emotion_vectors(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    emotions: List[str],
    neutral_texts: List[str],
    layer_idx: int,
    samples_per_emotion: int
) -> Dict[str, torch.Tensor]:
    """
    提取所有情绪向量
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        emotions: 情绪列表
        neutral_texts: 中性文本列表
        layer_idx: 干预层索引
        samples_per_emotion: 每种情绪的样本数
        
    Returns:
        情绪向量字典 {emotion: vector}
    """
    print("\n" + "="*60)
    print("开始提取情绪向量")
    print("="*60)
    
    # 1. 提取中性样本激活
    print(f"\n[1/3] 提取 {len(neutral_texts)} 个中性样本的激活值...")
    neutral_activations = get_model_activations(model, tokenizer, neutral_texts, layer_idx)
    print(f"   中性激活形状：{neutral_activations.shape}")
    
    # 2. 提取每种情绪的激活并计算向量
    print(f"\n[2/3] 提取 {len(emotions)} 种情绪的激活值并计算向量...")
    emotion_vectors = {}
    
    for emotion in emotions:
        print(f"\n   处理情绪：{emotion}")
        
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
        print(f"   ✅ {emotion} 向量计算完成，范数：{norm:.4f}")
    
    print(f"\n[3/3] 所有情绪向量提取完成！")
    print(f"   共提取 {len(emotion_vectors)} 个情绪向量")
    print(f"   向量维度：{list(emotion_vectors.values())[0].shape}")
    
    return emotion_vectors
