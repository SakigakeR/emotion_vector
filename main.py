"""
情绪向量工程主入口脚本
提供统一的命令行接口，支持预计算、推理和评估三种模式

使用方法:
    python main.py precompute    # 预计算情绪向量
    python main.py inference     # 运行推理实验
    python main.py evaluate      # 运行评估
    python main.py --help        # 显示帮助信息
"""

import argparse
import sys
from pathlib import Path


def print_banner():
    """打印程序横幅"""
    print("\n" + "="*70)
    print(" " * 20 + "情绪向量工程")
    print(" " * 10 + "Anthropic 情绪向量实验本地复现")
    print("="*70)


def run_precompute():
    """运行预计算流程"""
    from precompute import main as precompute_main
    print_banner()
    print("\n模式：预计算情绪向量")
    print("-"*70)
    print("\n此模式将:")
    print("  1. 加载语言模型")
    print("  2. 生成情绪和中性故事")
    print("  3. 提取模型激活值")
    print("  4. 计算情绪向量")
    print("  5. 保存到 FAISS 向量数据库")
    print("\n注意：此过程可能需要较长时间，请确保有足够的显存！")
    print("-"*70)
    
    input("\n按回车键开始预计算...")
    
    precompute_main()


def run_inference():
    """运行推理实验"""
    from inference import main as inference_main
    print_banner()
    print("\n模式：情绪干预推理")
    print("-"*70)
    print("\n此模式将:")
    print("  1. 加载语言模型")
    print("  2. 从 FAISS 数据库加载预计算的情绪向量")
    print("  3. 对测试提示进行情绪干预生成")
    print("  4. 输出对比结果")
    print("\n注意：请确保已先运行预计算模式生成向量数据库！")
    print("-"*70)
    
    # 检查向量数据库是否存在
    from config import VECTOR_DB_PATH, META_DATA_PATH
    if not VECTOR_DB_PATH.exists() or not META_DATA_PATH.exists():
        print("\n❌ 错误：未找到向量数据库文件！")
        print(f"   向量文件：{VECTOR_DB_PATH}")
        print(f"   元数据文件：{META_DATA_PATH}")
        print("\n请先运行：python main.py precompute")
        sys.exit(1)
    
    input("\n按回车键开始推理...")
    
    inference_main()


def run_evaluate():
    """运行评估"""
    from evaluator import main as evaluator_main
    print_banner()
    print("\n模式：情绪评估")
    print("-"*70)
    print("\n此模式将:")
    print("  1. 加载情感分析模型")
    print("  2. 评估示例文本的情感倾向")
    print("  3. 演示评估功能")
    print("-"*70)
    
    input("\n按回车键开始评估...")
    
    evaluator_main()


def check_dependencies():
    """检查依赖是否安装"""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import faiss
    except ImportError:
        missing.append("faiss-cpu")
    
    try:
        from tqdm import tqdm
    except ImportError:
        missing.append("tqdm")
    
    if missing:
        print("\n❌ 缺少以下依赖:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n请运行：pip install -r requirements.txt")
        sys.exit(1)
    
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="情绪向量工程 - Anthropic 情绪向量实验本地复现",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py precompute    预计算情绪向量并保存到 FAISS 数据库
  python main.py inference     加载预计算向量进行情绪干预推理
  python main.py evaluate      运行情绪评估演示
  python main.py check         检查依赖和环境

注意事项:
  1. 预计算模式需要较长时间和较大显存，建议使用 GPU
  2. 推理模式需要先运行预计算模式生成向量数据库
  3. 可通过 config.py 修改模型、情绪列表等配置
        """
    )
    
    parser.add_argument(
        "mode",
        choices=["precompute", "inference", "evaluate", "check"],
        help="运行模式"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式，跳过确认提示"
    )
    
    args = parser.parse_args()
    
    # 检查依赖
    print("\n检查依赖...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ 依赖检查通过")
    
    # 根据模式执行
    if args.mode == "precompute":
        run_precompute()
    elif args.mode == "inference":
        run_inference()
    elif args.mode == "evaluate":
        run_evaluate()
    elif args.mode == "check":
        print("\n✅ 环境检查完成")
        print("\n已安装的依赖:")
        print("  - torch")
        print("  - transformers")
        print("  - faiss-cpu")
        print("  - tqdm")
        
        from config import VECTOR_DB_PATH, META_DATA_PATH, MODEL_NAME, MODEL_ID
        print(f"\n配置信息:")
        print(f"  - 模型：{MODEL_NAME}")
        print(f"  - 模型标识：{MODEL_ID}")
        print(f"  - 向量数据库：{'存在' if VECTOR_DB_PATH.exists() else '不存在'}")
        
        if VECTOR_DB_PATH.exists():
            from vector_db import EmotionVectorDB
            import torch
            from transformers import AutoConfig
            
            config = AutoConfig.from_pretrained(MODEL_NAME)
            vec_db = EmotionVectorDB(vector_dim=config.hidden_size)
            stats = vec_db.get_stats()
            print(f"  - 向量数量：{stats['total_vectors']}")
            print(f"  - 情绪类型：{', '.join(stats['emotions'])}")


if __name__ == "__main__":
    main()
