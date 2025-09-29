"""
测试Kaldi文本对齐功能
"""

import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from evaluation.text_alignment import TextAligner, TextDiffVisualizer
from evaluation.text_comparison import TextComparator, ASRTextEvaluator
from evaluation.text_normalizer import TextNormalizer


class TestTextAligner:
    """测试TextAligner类"""

    def test_init(self):
        """测试初始化"""
        aligner = TextAligner()
        assert aligner.kaldi_path == ""
        assert aligner.align_text_cmd == "align-text"

    def test_fallback_alignment(self):
        """测试备用对齐方法"""
        aligner = TextAligner()

        # 测试相同文本
        result = aligner._fallback_alignment("hello world", "hello world")
        assert len(result) == 2
        assert result[0] == ("hello", "hello")
        assert result[1] == ("world", "world")

        # 测试不同长度文本
        result = aligner._fallback_alignment("hello", "hello world")
        assert len(result) == 3
        assert result[0] == ("hello", "hello")
        assert result[1] == ("", "world")

    def test_parse_alignment_output(self):
        """测试对齐输出解析"""
        aligner = TextAligner()

        # 模拟Kaldi输出格式
        output = "utt1 hello hello\nutt1 world world\nutt1 test *\nutt1 * extra"
        result = aligner._parse_alignment_output(output)

        assert len(result) == 4
        assert result[0] == ("hello", "hello")
        assert result[1] == ("world", "world")
        assert result[2] == ("test", "")
        assert result[3] == ("", "extra")

    def test_format_alignment(self):
        """测试对齐结果格式化"""
        aligner = TextAligner()
        alignment = [("hello", "hello"), ("world", "word"), ("test", "")]

        formatted = aligner.format_alignment(alignment)
        assert "hello | hello | ✓" in formatted
        assert "world | word  | ✗" in formatted
        assert "test  |       | ✗" in formatted

    def test_calculate_error_statistics(self):
        """测试错误统计计算"""
        aligner = TextAligner()
        alignment = [("hello", "hello"), ("world", "word"), ("test", ""), ("", "extra")]

        stats = aligner.calculate_error_statistics(alignment)

        assert stats["total"] == 4
        assert stats["correct"] == 1
        assert stats["substitutions"] == 1
        assert stats["deletions"] == 1
        assert stats["insertions"] == 1


class TestTextDiffVisualizer:
    """测试TextDiffVisualizer类"""

    def test_color_diff(self):
        """测试彩色差异显示"""
        visualizer = TextDiffVisualizer()
        ref, hyp = visualizer.color_diff("hello", "hello")
        assert "hello" in ref
        assert "hello" in hyp

    def test_side_by_side_diff(self):
        """测试并排差异显示"""
        visualizer = TextDiffVisualizer()
        result = visualizer.side_by_side_diff("hello world", "hello earth", width=20)
        assert "hello" in result
        assert "world" in result
        assert "earth" in result


class TestTextComparator:
    """测试TextComparator类"""

    def test_compare_texts(self):
        """测试文本比较"""
        comparator = TextComparator(language="en")
        result = comparator.compare_texts("hello world", "hello earth")

        assert "original" in result
        assert "normalized" in result
        assert "alignment" in result
        assert "visualization" in result

    def test_format_comparison_report(self):
        """测试比较报告格式化"""
        comparator = TextComparator(language="en")
        comparison = comparator.compare_texts("hello", "hello")
        report = comparator.format_comparison_report(comparison)

        assert "ASR文本比较分析报告" in report
        assert "hello" in report

    def test_batch_compare(self):
        """测试批量比较"""
        comparator = TextComparator(language="en")
        text_pairs = [("hello", "hello"), ("world", "word")]
        results = comparator.batch_compare(text_pairs)

        assert len(results) == 2
        assert all("alignment" in result for result in results)


class TestASRTextEvaluator:
    """测试ASR文本评估器"""

    def test_evaluate_asr_result(self):
        """测试单个ASR结果评估"""
        evaluator = ASRTextEvaluator(language="en")
        result = evaluator.evaluate_asr_result("hello world", "hello word")

        assert "simple_report" in result
        assert "text_comparison" in result
        assert "wer" in result["simple_report"]

    def test_evaluate_asr_batch(self):
        """测试批量ASR结果评估"""
        evaluator = ASRTextEvaluator(language="en")
        asr_results = [
            {"reference_text": "hello", "asr_text": "hello"},
            {"reference_text": "world", "asr_text": "word"}
        ]

        batch_result = evaluator.evaluate_asr_batch(asr_results)

        assert batch_result["total_samples"] == 2
        assert "overall_wer" in batch_result
        assert "individual_results" in batch_result


class TestIntegration:
    """测试集成"""

    def test_normalizer_with_aligner(self):
        """测试规范化器与对齐器的集成"""
        normalizer = TextNormalizer(language="en")
        comparator = TextComparator(language="en")

        original = "Hello, World!"
        normalized = normalizer.normalize(original)

        assert normalized == "hello world"  # 小写并移除标点

        result = comparator.compare_texts(original, original)
        assert result["original"]["reference"] == original
        assert result["normalized"]["reference"] == "hello world"

    def test_multilingual_support(self):
        """测试多语言支持"""
        languages = ["zh", "en", "ja"]

        for lang in languages:
            comparator = TextComparator(language=lang)
            result = comparator.compare_texts("test", "test")
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])