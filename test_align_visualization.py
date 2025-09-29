#!/usr/bin/env python3
"""
基于align-text结果的文本差异可视化测试
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from evaluation.text_alignment import TextAligner, TextDiffVisualizer

def test_align_text_visualization():
    """测试基于align-text的可视化"""
    print("=== 基于Kaldi align-text结果的文本差异可视化 ===\n")

    aligner = TextAligner()
    visualizer = TextDiffVisualizer()

    # 测试用例
    test_cases = [
        ("今天天气很好我们去公园散步", "今天天气很好我们去公司散步"),
        ("我喜欢吃苹果和香蕉", "我喜欢吃苹果"),
        ("你好世界", "你好美丽世界"),
        ("机器学习很有趣", "机器学习非常有意思")
    ]

    for i, (ref, hyp) in enumerate(test_cases, 1):
        print(f"测试用例 {i}:")
        print(f"参考文本: {ref}")
        print(f"识别结果: {hyp}")
        print()

        # 获取对齐结果
        result = aligner.generate_diff_report(ref, hyp)
        alignment = result['alignment']

        # 使用基于align-text的可视化
        ref_colored, hyp_colored = visualizer.color_diff_from_alignment(alignment)
        side_by_side = visualizer.side_by_side_diff_from_alignment(alignment, width=20)
        word_level = visualizer.word_level_diff(alignment)

        print("【彩色差异显示】:")
        print(f"参考: {ref_colored}")
        print(f"识别: {hyp_colored}")
        print()

        print("【并排差异显示】:")
        print(side_by_side)
        print()

        print(word_level)
        print("-" * 60)
        print()

def test_word_level_analysis():
    """测试词级分析"""
    print("=== 词级详细分析 ===\n")

    aligner = TextAligner()

    ref = "我今天要去超市买牛奶和面包"
    hyp = "我今天要去商店买牛奶"

    result = aligner.generate_diff_report(ref, hyp)

    print("原始文本:")
    print(f"参考: {ref}")
    print(f"识别: {hyp}")
    print()

    print("对齐结果:")
    for i, (ref_word, hyp_word) in enumerate(result['alignment'], 1):
        if ref_word == hyp_word and ref_word != "":
            print(f"{i:2d}. ✓ {ref_word}")
        elif ref_word and hyp_word and ref_word != hyp_word:
            print(f"{i:2d}. ✗ {ref_word} → {hyp_word} (替换)")
        elif ref_word and not hyp_word:
            print(f"{i:2d}. ✗ {ref_word} → [删除]")
        elif not ref_word and hyp_word:
            print(f"{i:2d}. ✗ [插入] → {hyp_word}")

    print()
    print("统计信息:")
    stats = result['statistics']
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_align_text_visualization()
    test_word_level_analysis()