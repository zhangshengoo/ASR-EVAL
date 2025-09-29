#!/usr/bin/env python3
"""
基于词级对齐的文本差异可视化测试
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from evaluation.text_alignment import TextAligner, TextDiffVisualizer

def test_word_level_alignment():
    """测试词级对齐和可视化"""
    print("=== 基于词级对齐的文本差异可视化 ===\n")

    aligner = TextAligner()
    visualizer = TextDiffVisualizer()

    # 测试用例 - 按词分割
    test_cases = [
        ("今天 天气 很好 我们 去 公园 散步", "今天 天气 很好 我们 去 公司 散步"),
        ("我 喜欢 吃 苹果 和 香蕉", "我 喜欢 吃 苹果"),
        ("你好 世界", "你好 美丽 世界"),
        ("机器 学习 很 有趣", "机器 学习 非常 有意思")
    ]

    for i, (ref, hyp) in enumerate(test_cases, 1):
        print(f"测试用例 {i}:")
        print(f"参考文本: {ref}")
        print(f"识别结果: {hyp}")
        print()

        # 获取对齐结果
        result = aligner.generate_diff_report(ref, hyp)
        alignment = result['alignment']

        print("【词级对齐结果】:")
        for j, (ref_word, hyp_word) in enumerate(alignment, 1):
            if ref_word == hyp_word and ref_word:
                print(f"{j:2d}. ✓ {ref_word}")
            elif ref_word and hyp_word and ref_word != hyp_word:
                print(f"{j:2d}. ✗ {ref_word} → {hyp_word} (替换)")
            elif ref_word and not hyp_word:
                print(f"{j:2d}. ✗ {ref_word} → [删除]")
            elif not ref_word and hyp_word:
                print(f"{j:2d}. ✗ [插入] → {hyp_word}")

        # 使用基于align-text的可视化
        ref_colored, hyp_colored = visualizer.color_diff_from_alignment(alignment)
        side_by_side = visualizer.side_by_side_diff_from_alignment(alignment, width=15)

        print("\n【彩色差异显示】:")
        print(f"参考: {ref_colored}")
        print(f"识别: {hyp_colored}")

        print("\n【并排差异显示】:")
        print(side_by_side)
        print("-" * 60)
        print()

def test_chinese_example():
    """测试中文示例"""
    print("=== 中文文本对齐示例 ===\n")

    aligner = TextAligner()
    visualizer = TextDiffVisualizer()

    # 中文示例
    ref = "今天 天气 很好 我们 去 公园 散步"
    hyp = "今天 天气 很好 我们 去 公司 散步"

    result = aligner.generate_diff_report(ref, hyp)
    alignment = result['alignment']

    print("原始文本:")
    print(f"参考: {ref}")
    print(f"识别: {hyp}")
    print()

    print("对齐结果:")
    for i, (ref_word, hyp_word) in enumerate(alignment, 1):
        if ref_word == hyp_word and ref_word:
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

    # 使用新的可视化
    ref_colored, hyp_colored = visualizer.color_diff_from_alignment(alignment)
    print(f"\n彩色差异显示:")
    print(f"参考: {ref_colored}")
    print(f"识别: {hyp_colored}")

if __name__ == "__main__":
    test_word_level_alignment()
    test_chinese_example()