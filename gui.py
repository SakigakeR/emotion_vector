"""
情绪向量工程 Gradio GUI
提供图形界面进行情绪干预推理和预计算
"""

import gradio as gr
import torch
from pathlib import Path
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import json

from config import (
    MODEL_NAME, DEVICE, EMOTIONS,
    INTERVENTION_LAYER, INTERVENTION_STRENGTH,
    VECTOR_DB_PATH, META_DATA_PATH, MODEL_ID,
    USE_4BIT_QUANTIZATION, RESULTS_DIR, MAX_NEW_TOKENS,
    GENERATION_TEMPERATURE, GENERATION_TOP_P
)
from vector_db import EmotionVectorDB
from intervention import EmotionIntervention
from extractor import create_quantization_config


def check_database_status():
    """检查向量数据库状态"""
    if not VECTOR_DB_PATH.exists():
        return "❌ 向量数据库不存在\n请先运行预计算模式"
    
    if not META_DATA_PATH.exists():
        return "❌ 元数据文件不存在\n请先运行预计算模式"
    
    try:
        from transformers import AutoConfig
        
        config = AutoConfig.from_pretrained(MODEL_NAME)
        
        # 获取 hidden_size
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        elif hasattr(config, "decoder") and hasattr(config.decoder, "hidden_size"):
            hidden_size = config.decoder.hidden_size
        elif hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            hidden_size = config.text_config.hidden_size
        elif hasattr(config, "llm_config") and hasattr(config.llm_config, "hidden_size"):
            hidden_size = config.llm_config.hidden_size
        else:
            hidden_size = 4096  # 默认值
        
        with EmotionVectorDB(
            vector_dim=hidden_size,
            db_path=VECTOR_DB_PATH,
            meta_path=META_DATA_PATH
        ) as vec_db:
            stats = vec_db.get_stats()
            emotions = stats.get('emotions', [])
            total = stats.get('total_vectors', 0)
            
            return f"✅ 向量数据库已就绪\n情绪类型：{', '.join(emotions)}\n向量总数：{total}"
    except Exception as e:
        return f"⚠️ 检查数据库时出错：{str(e)}"


def get_hidden_size(model):
    """获取模型的 hidden_size"""
    if hasattr(model.config, "hidden_size"):
        return model.config.hidden_size
    elif hasattr(model.config, "decoder") and hasattr(model.config.decoder, "hidden_size"):
        return model.config.decoder.hidden_size
    elif hasattr(model.config, "text_config") and hasattr(model.config.text_config, "hidden_size"):
        return model.config.text_config.hidden_size
    elif hasattr(model.config, "llm_config") and hasattr(model.config.llm_config, "hidden_size"):
        return model.config.llm_config.hidden_size
    else:
        raise AttributeError("无法找到 hidden_size 参数")


_model = None
_tokenizer = None

def load_model_once():
    """单例加载模型和分词器"""
    global _model, _tokenizer
    
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    try:
        print("正在加载模型...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        bnb_config = create_quantization_config()
        
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        if bnb_config:
            load_kwargs["quantization_config"] = bnb_config
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            **load_kwargs
        )
        
        model.eval()
        print("✅ 模型加载完成")
        
        _model = model
        _tokenizer = tokenizer
        return _model, _tokenizer
    except Exception as e:
        raise e

def run_inference(prompt: str, emotion: str, strength: float, temperature: float, max_tokens: int):
    """运行情绪干预推理"""
    
    # 检查数据库
    if not VECTOR_DB_PATH.exists() or not META_DATA_PATH.exists():
        return "❌ 向量数据库不存在，请先运行预计算模式", "", ""
    
    # 加载模型
    try:
        model, tokenizer = load_model_once()
    except Exception as e:
        return f"❌ 模型加载失败：{str(e)}", "", ""
    
    # 获取情绪向量
    try:
        hidden_size = get_hidden_size(model)
        
        with EmotionVectorDB(
            vector_dim=hidden_size,
            db_path=VECTOR_DB_PATH,
            meta_path=META_DATA_PATH
        ) as vec_db:
            emotion_vectors = vec_db.get_vectors(MODEL_ID)
    except Exception as e:
        return f"❌ 加载情绪向量失败：{str(e)}", "", ""
    
    if emotion not in emotion_vectors:
        return f"❌ 未找到情绪 '{emotion}' 的向量", "", ""
    
    emotion_vector = emotion_vectors[emotion]
    
    # 编码输入
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # 生成原始输出（无干预）
    with torch.no_grad():
        original_output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else None,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True if temperature > 0 else False
        )
    
    original_text = tokenizer.decode(original_output[0], skip_special_tokens=True)
    
    # 生成干预后输出
    try:
        # 创建干预器
        intervention = EmotionIntervention(
            layer_idx=INTERVENTION_LAYER,
            emotion_vector=emotion_vector,
            strength=strength,
            apply_to_all_tokens=True
        )
        intervention.to(DEVICE)
        
        # 注册钩子
        # 获取干预层
        if hasattr(model, "layers"):
            target_layer = model.layers[INTERVENTION_LAYER]
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            target_layer = model.model.layers[INTERVENTION_LAYER]
        elif hasattr(model.model,"language_model") and hasattr(model.model.language_model, "layers"):
            target_layer = model.model.language_model.layers[INTERVENTION_LAYER]
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            target_layer = model.transformer.h[INTERVENTION_LAYER]
        else:
            return f"❌ 无法找到干预层 {INTERVENTION_LAYER}", original_text, ""
        
        # 注册钩子到层的输出
        hook = target_layer.register_forward_hook(intervention.hook_fn)
        
        # 生成干预后输出
        with torch.no_grad():
            intervened_output = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True if temperature > 0 else False
            )
        
        # 移除钩子
        hook.remove()
        
        intervened_text = tokenizer.decode(intervened_output[0], skip_special_tokens=True)
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ 干预生成失败：{str(e)}\n\n{error_detail}", original_text, ""
    
    # 准备结果
    result = f"""
=== 情绪干预结果 ===

原始提示：{prompt}

情绪类型：{emotion}
干预强度：{strength}
温度参数：{temperature}
最大 token 数：{max_tokens}

--- 原始输出（无干预）---
{original_text}

--- 干预后输出（{emotion}）---
{intervened_text}
"""
    
    return result, original_text, intervened_text


def run_precompute():
    """运行预计算"""
    from precompute import main as precompute_main
    
    try:
        precompute_main()
        return "✅ 预计算完成！请刷新页面查看数据库状态。"
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ 预计算失败：{str(e)}\n\n{error_detail}"


def get_system_info():
    """获取系统信息"""
    info = []
    info.append(f"模型：{MODEL_NAME}")
    info.append(f"设备：{DEVICE}")
    info.append(f"4 位量化：{USE_4BIT_QUANTIZATION}")
    info.append(f"干预层：{INTERVENTION_LAYER}")
    info.append(f"默认干预强度：{INTERVENTION_STRENGTH}")
    info.append(f"情绪列表：{', '.join(EMOTIONS)}")
    info.append(f"向量数据库：{'✅ 存在' if VECTOR_DB_PATH.exists() else '❌ 不存在'}")
    info.append(f"元数据文件：{'✅ 存在' if META_DATA_PATH.exists() else '❌ 不存在'}")
    
    return "\n".join(info)


def create_interface():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="情绪向量工程", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎭 情绪向量工程
        
        Anthropic 情绪向量实验本地复现 - Gradio 图形界面
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📊 系统状态")
                
                db_status = gr.Textbox(
                    label="向量数据库状态",
                    value="点击'刷新状态'按钮检查",
                    interactive=False
                )
                
                refresh_btn = gr.Button("🔄 刷新状态", variant="secondary")
                
                gr.Markdown("### ℹ️ 系统信息")
                sys_info = gr.Textbox(
                    label="配置信息",
                    value=get_system_info(),
                    interactive=False,
                    lines=10
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 预计算")
                
                precompute_btn = gr.Button("🚀 运行预计算", variant="primary")
                precompute_output = gr.Textbox(
                    label="预计算结果",
                    interactive=False,
                    lines=5
                )
                
                gr.Markdown("""
                **注意**: 预计算需要较长时间和较大显存，请耐心等待。
                """)
        
        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ✍️ 输入提示")
                
                prompt_input = gr.Textbox(
                    label="输入提示",
                    placeholder="请输入要生成文本的提示，例如：'Once upon a time, there was a little girl who lived in a small village...'",
                    lines=4
                )
                
                with gr.Row():
                    emotion_input = gr.Dropdown(
                        choices=EMOTIONS,
                        value=EMOTIONS[0],
                        label="选择情绪"
                    )
                    strength_input = gr.Slider(
                        minimum=0.1,
                        maximum=256.0,
                        value=INTERVENTION_STRENGTH,
                        step=0.1,
                        label="干预强度"
                    )
                
                with gr.Row():
                    temp_input = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=GENERATION_TEMPERATURE,
                        step=0.1,
                        label="温度"
                    )
                    max_tokens_input = gr.Slider(
                        minimum=64,
                        maximum=1024,
                        value=MAX_NEW_TOKENS,
                        step=64,
                        label="最大 Token 数"
                    )
                
                infer_btn = gr.Button("🎯 运行推理", variant="primary")
            
            with gr.Column(scale=2):
                gr.Markdown("### 📝 推理结果")
                
                result_output = gr.Textbox(
                    label="完整结果",
                    interactive=False,
                    lines=20
                )
        
        with gr.Row():
            gr.Markdown("### 📄 原始输出（无干预）")
            original_output = gr.Textbox(
                label="原始文本",
                interactive=False,
                lines=10
            )
        
        with gr.Row():
            gr.Markdown("### 🎭 干预后输出")
            intervened_output = gr.Textbox(
                label=f"{EMOTIONS[0]} 情绪文本",
                interactive=False,
                lines=10
            )
        
        # 绑定事件
        refresh_btn.click(
            fn=check_database_status,
            outputs=[db_status]
        )
        
        precompute_btn.click(
            fn=run_precompute,
            outputs=[precompute_output]
        )
        
        infer_btn.click(
            fn=run_inference,
            inputs=[prompt_input, emotion_input, strength_input, temp_input, max_tokens_input],
            outputs=[result_output, original_output, intervened_output]
        )
    
    return demo


def main():
    """主函数"""
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
