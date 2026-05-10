"""
情绪干预评估模块
使用情感分析模型评估干预效果
"""

import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass
from transformers import pipeline
from tqdm import tqdm
import numpy as np

from config import DEVICE, SENTIMENT_MODEL


@dataclass
class EmotionEvaluationResult:
    """情绪评估结果"""
    emotion: str
    text: str
    sentiment_label: str
    sentiment_score: float
    is_positive: bool


class EmotionEvaluator:
    """
    情绪评估器
    使用预训练的情感分析模型评估文本的情绪倾向
    """
    
    def __init__(self, model_name: str = SENTIMENT_MODEL, device: str = None):
        """
        初始化评估器
        
        Args:
            model_name: 情感分析模型名称
            device: 运行设备
        """
        self.device = device or DEVICE
        print(f"加载情感分析模型：{model_name}")
        
        self.analyzer = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=self.device
        )
        
        print("✅ 情感分析模型加载完成")
    
    def analyze(self, text: str) -> Tuple[str, float]:
        """
        分析文本情感
        
        Args:
            text: 输入文本
            
        Returns:
            (label, score) 元组
        """
        result = self.analyzer(text)[0]
        return result['label'], result['score']
    
    def evaluate_text(
        self,
        text: str,
        target_emotion: str
    ) -> EmotionEvaluationResult:
        """
        评估单个文本的情绪表达
        
        Args:
            text: 生成的文本
            target_emotion: 目标情绪
            
        Returns:
            评估结果
        """
        label, score = self.analyze(text)
        
        return EmotionEvaluationResult(
            emotion=target_emotion,
            text=text,
            sentiment_label=label,
            sentiment_score=score,
            is_positive=(label == "POSITIVE")
        )
    
    def evaluate_intervention(
        self,
        baseline_text: str,
        intervened_text: str,
        target_emotion: str
    ) -> Dict:
        """
        评估干预效果：对比干预前后的变化
        
        Args:
            baseline_text: 无干预的基线文本
            intervened_text: 干预后的文本
            target_emotion: 目标情绪
            
        Returns:
            评估结果字典
        """
        baseline_result = self.evaluate_text(baseline_text, "baseline")
        intervened_result = self.evaluate_text(intervened_text, target_emotion)
        
        # 计算情感变化
        sentiment_shift = intervened_result.sentiment_score - baseline_result.sentiment_score
        label_changed = baseline_result.sentiment_label != intervened_result.sentiment_label
        
        return {
            "baseline": {
                "label": baseline_result.sentiment_label,
                "score": baseline_result.sentiment_score
            },
            "intervened": {
                "label": intervened_result.sentiment_label,
                "score": intervened_result.sentiment_score
            },
            "shift": sentiment_shift,
            "label_changed": label_changed,
            "target_emotion": target_emotion
        }


class EmotionVectorEvaluator:
    """
    情绪向量评估器
    评估情绪向量的有效性和一致性
    """
    
    def __init__(self):
        self.results = []
    
    def compute_cosine_similarity(
        self,
        vector1: torch.Tensor,
        vector2: torch.Tensor
    ) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vector1: 第一个向量
            vector2: 第二个向量
            
        Returns:
            余弦相似度值
        """
        return torch.nn.functional.cosine_similarity(
            vector1.unsqueeze(0),
            vector2.unsqueeze(0)
        ).item()
    
    def evaluate_vector_consistency(
        self,
        emotion_vectors: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        评估情绪向量之间的一致性（正交性）
        
        Args:
            emotion_vectors: 情绪向量字典
            
        Returns:
            相似度矩阵 {emotion1_emotion2: similarity}
        """
        emotions = list(emotion_vectors.keys())
        similarities = {}
        
        for i, em1 in enumerate(emotions):
            for j, em2 in enumerate(emotions):
                if i < j:  # 只计算上三角
                    key = f"{em1}_{em2}"
                    sim = self.compute_cosine_similarity(
                        emotion_vectors[em1],
                        emotion_vectors[em2]
                    )
                    similarities[key] = sim
        
        return similarities
    
    def evaluate_emotion_distinctiveness(
        self,
        similarities: Dict[str, float]
    ) -> Dict:
        """
        评估情绪向量的区分度
        
        Args:
            similarities: 相似度字典
            
        Returns:
            评估结果
        """
        if not similarities:
            return {"error": "No similarities computed"}
        
        sim_values = list(similarities.values())
        
        return {
            "mean_similarity": np.mean(sim_values),
            "std_similarity": np.std(sim_values),
            "min_similarity": np.min(sim_values),
            "max_similarity": np.max(sim_values),
            "highly_similar_pairs": [
                pair for pair, sim in similarities.items()
                if sim > 0.8
            ]
        }


def run_comprehensive_evaluation(
    evaluator: EmotionEvaluator,
    results: Dict[str, Dict[str, str]],
    emotions: List[str]
) -> Dict:
    """
    运行综合评估
    
    Args:
        evaluator: 情绪评估器
        results: 干预实验结果
        emotions: 情绪列表
        
    Returns:
        综合评估报告
    """
    print("\n" + "="*60)
    print("运行综合评估")
    print("="*60)
    
    all_evaluations = []
    
    for prompt, emotion_results in tqdm(results.items(), desc="评估每个提示"):
        baseline = emotion_results.get("none", "")
        
        for emotion in emotions:
            if emotion not in emotion_results:
                continue
            
            intervened = emotion_results[emotion]
            evaluation = evaluator.evaluate_intervention(
                baseline, intervened, emotion
            )
            evaluation["prompt"] = prompt
            all_evaluations.append(evaluation)
    
    # 生成统计报告
    report = generate_evaluation_report(all_evaluations, emotions)
    return report


def generate_evaluation_report(
    evaluations: List[Dict],
    emotions: List[str]
) -> Dict:
    """
    生成评估报告
    
    Args:
        evaluations: 所有评估结果
        emotions: 情绪列表
        
    Returns:
        评估报告
    """
    report = {
        "total_evaluations": len(evaluations),
        "emotion_stats": {},
        "overall_stats": {}
    }
    
    # 按情绪统计
    for emotion in emotions:
        emotion_evals = [e for e in evaluations if e["target_emotion"] == emotion]
        
        if not emotion_evals:
            continue
        
        shifts = [e["shift"] for e in emotion_evals]
        label_changes = sum(1 for e in emotion_evals if e["label_changed"])
        
        report["emotion_stats"][emotion] = {
            "count": len(emotion_evals),
            "mean_shift": np.mean(shifts),
            "std_shift": np.std(shifts),
            "label_change_rate": label_changes / len(emotion_evals),
            "positive_shifts": sum(1 for s in shifts if s > 0),
            "negative_shifts": sum(1 for s in shifts if s < 0)
        }
    
    # 整体统计
    all_shifts = [e["shift"] for e in evaluations]
    all_label_changes = sum(1 for e in evaluations if e["label_changed"])
    
    report["overall_stats"] = {
        "mean_shift": np.mean(all_shifts),
        "std_shift": np.std(all_shifts),
        "overall_label_change_rate": all_label_changes / len(evaluations) if evaluations else 0,
        "effective_interventions": all_label_changes
    }
    
    return report


def print_evaluation_report(report: Dict) -> None:
    """打印评估报告"""
    print("\n" + "="*60)
    print("评估报告")
    print("="*60)
    
    print(f"\n总评估次数：{report['total_evaluations']}")
    
    print("\n各情绪统计:")
    print("-"*60)
    for emotion, stats in report["emotion_stats"].items():
        print(f"\n{emotion.upper()}:")
        print(f"  样本数：{stats['count']}")
        print(f"  平均情感变化：{stats['mean_shift']:.4f} (+/- {stats['std_shift']:.4f})")
        print(f"  标签变化率：{stats['label_change_rate']:.2%}")
        print(f"  正向变化：{stats['positive_shifts']}, 负向变化：{stats['negative_shifts']}")
    
    print("\n整体统计:")
    print("-"*60)
    overall = report["overall_stats"]
    print(f"  平均情感变化：{overall['mean_shift']:.4f} (+/- {overall['std_shift']:.4f})")
    print(f"  整体标签变化率：{overall['overall_label_change_rate']:.2%}")
    print(f"  有效干预次数：{overall['effective_interventions']}")


def main():
    """主函数：演示评估功能"""
    print("\n" + "="*60)
    print("情绪干预评估脚本")
    print("="*60)
    
    # 初始化评估器
    evaluator = EmotionEvaluator()
    
    # 示例评估
    test_texts = {
        "happy": "Today was amazing! Everything went perfectly and I felt so joyful.",
        "sad": "I had a terrible day. Nothing went right and I feel so down.",
        "angry": "I'm furious! This is completely unacceptable and I can't believe it.",
        "baseline": "Today was a normal day. I did my usual activities."
    }
    
    print("\n示例文本评估:")
    print("-"*60)
    
    for emotion, text in test_texts.items():
        label, score = evaluator.analyze(text)
        print(f"\n{emotion.upper()}:")
        print(f"  文本：{text[:50]}...")
        print(f"  情感：{label} ({score:.4f})")
    
    print("\n" + "="*60)
    print("✅ 评估演示完成！")
    print("="*60)


if __name__ == "__main__":
    main()
