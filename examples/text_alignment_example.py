"""
Kaldi文本对齐工具使用示例
演示如何使用TextAligner和TextComparator进行文本比较
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.text_alignment import TextAligner, TextDiffVisualizer
from evaluation.text_comparison import TextComparator, ASRTextEvaluator
from evaluation.text_normalizer import TextNormalizer


def example_basic_alignment():
    """基本文本对齐示例"""
    print("=== 基本文本对齐示例 ===")

    # 创建文本对齐器
    aligner = TextAligner()

    # 示例文本
    reference = "今天天气很好我们去公园散步"
    hypothesis = "今天天气很好我们去公司散步"

    # 进行文本对齐
    result = aligner.generate_diff_report(reference, hypothesis)

    print("参考文本:", result['reference'])
    print("识别结果:", result['hypothesis'])
    print()
    print("对齐结果:")
    print(result['formatted_output'])
    print()
    print("错误统计:")
    for key, value in result['statistics'].items():
        print(f"  {key}: {value}")


def example_multilingual_comparison():
    """多语言文本比较示例"""
    print("\n=== 多语言文本比较示例 ===")

    # 中文示例
    print("\n【中文示例】")
    comparator_zh = TextComparator(language="zh")
    result_zh = comparator_zh.compare_texts(
        "我今天要去超市买牛奶和面包",
        "我今天要去超市买牛奶"
    )
    print(comparator_zh.format_comparison_report(result_zh))

    # 英文示例
    print("\n【英文示例】")
    comparator_en = TextComparator(language="en")
    result_en = comparator_en.compare_texts(
        "I want to go to the supermarket to buy milk and bread",
        "I want to go to supermarket buy milk"
    )
    print(comparator_en.format_comparison_report(result_en))


def example_visualization():
    """基于align-text的可视化示例"""
    print("\n=== 基于align-text的文本差异可视化示例 ===")

    aligner = TextAligner()
    visualizer = TextDiffVisualizer()

    reference = "The quick brown fox jumps over the lazy dog"
    hypothesis = "The quick brown cat jumps over lazy dogs"

    # 获取对齐结果
    result = aligner.generate_diff_report(reference, hypothesis)
    alignment = result['alignment']

    # 使用基于align-text的可视化
    ref_colored, hyp_colored = visualizer.color_diff_from_alignment(alignment)
    side_by_side = visualizer.side_by_side_diff_from_alignment(alignment, width=30)
    word_level = visualizer.word_level_diff(alignment)

    print("【彩色差异显示】:")
    print("参考文本:", ref_colored)
    print("识别结果:", hyp_colored)

    print("\n【并排差异显示】:")
    print(side_by_side)

    print("\n【词级差异分析】:")
    print(word_level)


def example_asr_evaluation():
    """ASR评估示例"""
    print("\n=== ASR评估示例 ===")

    evaluator = ASRTextEvaluator(language="zh")

    # 模拟ASR结果
    asr_results = [
        {
            "reference_text": "你好世界",
            "asr_text": "你好世界",
            "audio_path": "audio1.wav"
        },
        {
            "reference_text": "今天天气真好",
            "asr_text": "今天天气真差",
            "audio_path": "audio2.wav"
        },
        {
            "reference_text": "我喜欢吃苹果",
            "asr_text": "我喜欢吃香蕉",
            "audio_path": "audio3.wav"
        }
    ]

    # 批量评估
    batch_result = evaluator.evaluate_asr_batch(asr_results)

    print("批量评估结果:")
    print(f"总样本数: {batch_result['total_samples']}")
    print(f"总词数: {batch_result['total_words']}")
    print(f"总体WER: {batch_result['overall_wer']:.2%}")
    print(f"总体准确率: {batch_result['overall_accuracy']:.2%}")

    print("\n错误分析:")
    for error_type, count in batch_result['error_breakdown'].items():
        print(f"  {error_type}: {count}")


def example_with_normalization():
    """使用文本规范化的示例"""
    print("\n=== 文本规范化示例 ===")

    normalizer = TextNormalizer(language="zh")

    # 带数字和标点的文本
    original_text = "我今天花了￥123.45买了2瓶牛奶和3个面包"
    normalized_text = normalizer.normalize(original_text)

    print("原始文本:", original_text)
    print("规范化后:", normalized_text)

    # 使用规范化后的文本进行对齐
    comparator = TextComparator(language="zh")
    result = comparator.compare_texts(
        "我今天花了123.45元买了两瓶牛奶和三个面包",
        "我今天花了123元买了两瓶牛奶",
        normalize=True
    )

    print(comparator.format_comparison_report(result))


if __name__ == "__main__":
    print("Kaldi文本对齐工具使用示例")
    print("=" * 50)

    # 运行所有示例
    example_basic_alignment()
    example_multilingual_comparison()
    example_visualization()
    example_asr_evaluation()
    example_with_normalization()

    print("\n示例运行完成！")
    print("\n使用说明:")
    print("1. 安装Kaldi以获得最佳文本对齐效果")
    print("2. 根据需要选择语言：zh/en/ja/jp")
    print("3. 可选依赖：WeTextProcessing（中文）、NVIDIA NeMo（英文/日文）")
    print("4. 运行：python examples/text_alignment_example.py")