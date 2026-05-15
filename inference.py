"""
情绪干预推理脚本
加载预计算的情绪向量，进行情绪干预实验
"""

import torch
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    MODEL_NAME, DEVICE, EMOTIONS,
    INTERVENTION_LAYER, INTERVENTION_STRENGTH,
    VECTOR_DB_PATH, META_DATA_PATH, MODEL_ID,
    USE_4BIT_QUANTIZATION, RESULTS_DIR,MAX_NEW_TOKENS
)
from vector_db import EmotionVectorDB
from intervention import EmotionIntervention
from extractor import create_quantization_config


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
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = create_quantization_config()
    
    if bnb_config:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )
    
    model.eval()
    print(f"✅ 模型加载完成")
    
    return tokenizer, model


def load_emotion_vectors(model: AutoModelForCausalLM, model_id: str) -> Dict[str, torch.Tensor]:
    """
    从向量数据库加载预计算的情绪向量
    
    Args:
        model: 语言模型
        model_id: 模型唯一标识
        
    Returns:
        情绪向量字典
    """
    print("\n" + "="*60)
    print("加载预计算的情绪向量")
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

    # 使用上下文管理器加载数据库
    with EmotionVectorDB(
        vector_dim=hidden_dim,
        db_path=VECTOR_DB_PATH,
        meta_path=META_DATA_PATH
    ) as vec_db:
        # 获取指定模型的向量
        emotion_vectors = vec_db.get_vectors(model_id)
    
    # 确保向量在正确的设备上
    for emotion in emotion_vectors:
        emotion_vectors[emotion] = emotion_vectors[emotion].to(DEVICE)
    
    return emotion_vectors


def generate_with_intervention(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    emotion: Optional[str],
    emotion_vectors: Dict[str, torch.Tensor],
    layer_idx: int,
    strength: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = 0.7,
    top_p: float = 0.95
) -> str:
    """
    使用情绪干预生成文本（使用 chat_template）
    
    Args:
        model: 语言模型
        tokenizer: 分词器
        prompt: 输入提示
        emotion: 情绪类型（None 表示无干预）
        emotion_vectors: 情绪向量字典
        layer_idx: 干预层索引
        strength: 干预强度
        max_new_tokens: 最大生成 token 数
        temperature: 生成温度
        top_p: 核采样参数
        
    Returns:
        生成的文本
    """
    # 使用 chat 格式的消息
    messages = [{"role": "user", "content": prompt}]
    
    # 使用 chat_template 编码
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # 编码输入
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    
    # 如果有情绪，注册干预
    intervention = None
    if emotion and emotion in emotion_vectors:
        intervention = EmotionIntervention(
            layer_idx=layer_idx,
            emotion_vector=emotion_vectors[emotion],
            strength=strength
        )
        intervention.register(model)
    
    # 生成文本
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # 解码输出 - 先解码输入部分（带特殊 token），再解码完整输出（不带特殊 token）
    # 这样可以正确计算输入部分的长度
    input_length = len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False))
    full_text_with_special = tokenizer.decode(outputs[0], skip_special_tokens=False)
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取助手回复部分
    # 方法：使用输入部分的 token 数量来切片
    input_token_count = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_token_count:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    # 移除干预
    if intervention:
        intervention.remove()
    
    return response


def run_intervention_experiment(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    emotion_vectors: Dict[str, torch.Tensor],
    prompts: List[str],
    emotions: List[str],
    layer_idx: int,
    strength: float
) -> Dict[str, Dict[str, str]]:
    """
    运行情绪干预实验
    
    Args:
        tokenizer: 分词器
        model: 语言模型
        emotion_vectors: 情绪向量字典
        prompts: 测试提示列表
        emotions: 情绪列表
        layer_idx: 干预层索引
        strength: 干预强度
        
    Returns:
        实验结果 {prompt: {emotion: response}}
    """
    print("\n" + "="*60)
    print("运行情绪干预实验")
    print("="*60)
    print(f"提示数量：{len(prompts)}")
    print(f"情绪数量：{len(emotions)}")
    print(f"干预层：{layer_idx}")
    print(f"干预强度：{strength}")
    
    results = {}
    
    for prompt_idx, prompt in enumerate(prompts, 1):
        print(f"\n[提示 {prompt_idx}/{len(prompts)}] {prompt}")
        results[prompt] = {}
        
        # 无干预 baseline
        print("   - 无干预 (baseline)...")
        baseline = generate_with_intervention(
            model, tokenizer, prompt, None,
            emotion_vectors, layer_idx, strength
        )
        results[prompt]["none"] = baseline
        print(f"     {baseline[:100]}...")
        
        # 各种情绪干预
        for emotion in emotions:
            if emotion not in emotion_vectors:
                print(f"   ⚠️ 跳过 {emotion}: 向量不存在")
                continue
            
            print(f"   - {emotion} 干预...")
            response = generate_with_intervention(
                model, tokenizer, prompt, emotion,
                emotion_vectors, layer_idx, strength
            )
            results[prompt][emotion] = response
            print(f"     {response[:100]}...")
    
    return results


def print_results(results: Dict[str, Dict[str, str]]) -> None:
    """打印实验结果"""
    print("\n" + "="*60)
    print("实验结果")
    print("="*60)
    
    for prompt, emotion_results in results.items():
        print(f"\n{'─'*60}")
        print(f"提示：{prompt}")
        print(f"{'─'*60}")
        
        for emotion, response in emotion_results.items():
            emotion_label = emotion.upper() if emotion else "BASELINE"
            print(f"\n[{emotion_label}]:")
            print(response)


def save_results_to_markdown(
    results: Dict[str, Dict[str, str]],
    model_name: str,
    layer_idx: int,
    strength: float
) -> str:
    """
    将实验结果保存为 markdown 文件
    
    Args:
        results: 实验结果 {prompt: {emotion: response}}
        model_name: 模型名称
        layer_idx: 干预层索引
        strength: 干预强度
        
    Returns:
        保存的文件路径
    """
    # 创建目录结构：results/models_name/date/emotion/
    date_str = datetime.now().strftime("%Y-%m-%d")
    model_dir = RESULTS_DIR / model_name.replace("/", "-") / date_str
    
    # 获取所有情绪类型（从第一个 prompt 的 emotion_results 中提取）
    saved_files = []
    emotions_list = list(next(iter(results.values())).keys()) if results else ["none"]
    
    for emotion in emotions_list:
        emotion_dir = model_dir / emotion
        emotion_dir.mkdir(parents=True, exist_ok=True)
        
        # 构建 markdown 内容
        md_content = f"""# 情绪干预实验结果

## 实验配置

- **模型**: {model_name}
- **干预层**: {layer_idx}
- **干预强度**: {strength}
- **实验日期**: {date_str}
- **情绪类型**: {emotion.upper() if emotion else "BASELINE"}

---

## 实验结果

"""
        # 添加每个 prompt 的结果
        for prompt, emotion_results in results.items():
            if emotion in emotion_results:
                response = emotion_results[emotion]
                md_content += f"""### 提示：{prompt}

{response}

---

"""
        
        # 保存文件
        output_file = emotion_dir / f"emotion_{emotion}_results.md"
        output_file.write_text(md_content, encoding="utf-8")
        saved_files.append(str(output_file))
    
    # 同时保存一个汇总文件
    summary_dir = model_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    summary_content = f"""# 情绪干预实验汇总

## 实验配置

- **模型**: {model_name}
- **干预层**: {layer_idx}
- **干预强度**: {strength}
- **实验日期**: {date_str}

---

## 实验结果对比

"""
    # 为每个 prompt 创建对比表格
    for prompt, emotion_results in results.items():
        summary_content += f"""### 提示：{prompt}

| 情绪类型 | 回复 |
|---------|------|
"""
        for emotion, response in emotion_results.items():
            emotion_label = emotion.upper() if emotion else "BASELINE"
            # 截断过长的回复
            display_response = response[:200] + "..." if len(response) > 200 else response
            # 转义表格中的特殊字符
            display_response = display_response.replace("|", "\\|").replace("\n", " ")
            summary_content += f"| {emotion_label} | {display_response} |\n"
        
        summary_content += "\n---\n"
    
    summary_file = summary_dir / "summary.md"
    summary_file.write_text(summary_content, encoding="utf-8")
    saved_files.append(str(summary_file))
    
    print(f"\n✅ 结果已保存到：{saved_files}")
    return saved_files


def main():
    """主函数：执行推理实验"""
    print("\n" + "="*60)
    print("情绪干预推理脚本")
    print("="*60)
    print(f"模型：{MODEL_NAME}")
    print(f"设备：{DEVICE}")
    print(f"模型标识：{MODEL_ID}")
    
    # 1. 加载模型
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME)
    
    # 2. 加载预计算的情绪向量
    emotion_vectors = load_emotion_vectors(model, MODEL_ID)
    
    # 3. 定义测试提示
    test_prompts = [
        "Describe your day so far.",
        "How are you feeling today?",
        "Tell me about your weekend plans.",
        "What's your favorite season and why?",
        #### Chinese ####
        "构想一个人拿起了一杯水，他往水面看去，他看见了什么？",
        #### Japanese ####
        "初音ミクを筆頭に有名な合成音声(Voiceroidなど)を紹介してください。",
    ]
    
    # 4. 运行干预实验
    results = run_intervention_experiment(
        tokenizer, model, emotion_vectors,
        test_prompts, EMOTIONS,
        INTERVENTION_LAYER, INTERVENTION_STRENGTH
    )
    
    # 5. 打印结果
    print_results(results)
    
    # 6. 保存结果到 markdown 文件
    save_results_to_markdown(
        results, MODEL_NAME,
        INTERVENTION_LAYER, INTERVENTION_STRENGTH
    )
    
    print("\n" + "="*60)
    print("✅ 推理实验完成！")
    print("="*60)


if __name__ == "__main__":
    main()
