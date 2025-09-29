"""
文本比较分析模块
结合TextNormalizer和TextAligner提供完整的ASR文本比较功能
"""

from typing import Dict, List, Optional, Tuple, Any
from evaluation.text_normalizer import TextNormalizer
from evaluation.text_alignment import TextAligner, TextDiffVisualizer


class TextComparator:
    """文本比较器 - 结合规范化和文本对齐"""

    def __init__(self, language: str = "zh", kaldi_path: Optional[str] = None,
                 use_wetextprocessing: bool = True, use_nemo: bool = True):
        """
        初始化文本比较器

        Args:
            language: 语言类型 (zh, en, ja, jp)
            kaldi_path: Kaldi工具路径
            use_wetextprocessing: 是否使用WeTextProcessing
            use_nemo: 是否使用NVIDIA NeMo
        """
        self.normalizer = TextNormalizer(language, use_wetextprocessing, use_nemo)
        self.aligner = TextAligner(kaldi_path)
        self.visualizer = TextDiffVisualizer()

    def compare_texts(self, reference: str, hypothesis: str,
                     normalize: bool = True) -> Dict[str, Any]:
        """
        比较两个文本，包含规范化和详细分析

        Args:
            reference: 参考文本
            hypothesis: 假设文本（ASR识别结果）
            normalize: 是否进行文本规范化

        Returns:
            包含比较结果的字典
        """
        # 文本规范化
        if normalize:
            norm_reference = self.normalizer.normalize(reference)
            norm_hypothesis = self.normalizer.normalize(hypothesis)
        else:
            norm_reference = reference
            norm_hypothesis = hypothesis

        # 使用Kaldi对齐工具进行文本对齐
        alignment_result = self.aligner.generate_diff_report(norm_reference, norm_hypothesis)

        # 使用align-text结果的可视化
        visualization = alignment_result.get('visualization', {})
        ref_colored = visualization.get('colored_reference', '')
        hyp_colored = visualization.get('colored_hypothesis', '')
        side_by_side = visualization.get('side_by_side', '')
        word_level = visualization.get('word_level_diff', '')

        result = {
            "original": {
                "reference": reference,
                "hypothesis": hypothesis
            },
            "normalized": {
                "reference": norm_reference,
                "hypothesis": norm_hypothesis
            },
            "alignment": alignment_result,
            "visualization": {
                "colored_reference": ref_colored,
                "colored_hypothesis": hyp_colored,
                "side_by_side": side_by_side,
                "word_level_diff": word_level
            },
            "kaldi_available": self.aligner._check_kaldi_available()
        }

        return result

    def format_comparison_report(self, comparison_result: Dict[str, Any]) -> str:
        """格式化比较报告"""
        lines = []

        # 标题
        lines.append("=" * 80)
        lines.append("ASR文本比较分析报告")
        lines.append("=" * 80)

        # 原始文本
        lines.append("\n【原始文本】")
        lines.append(f"参考文本: {comparison_result['original']['reference']}")
        lines.append(f"识别结果: {comparison_result['original']['hypothesis']}")

        # 规范化文本
        lines.append("\n【规范化文本】")
        lines.append(f"参考文本: {comparison_result['normalized']['reference']}")
        lines.append(f"识别结果: {comparison_result['normalized']['hypothesis']}")

        # 对齐结果
        lines.append("\n【文本对齐结果】")
        lines.append(comparison_result['alignment']['formatted_output'])

        # 错误统计
        stats = comparison_result['alignment']['statistics']
        lines.append("\n【错误统计】")
        lines.append(f"总词数: {stats['total']}")
        lines.append(f"正确词数: {stats['correct']}")
        lines.append(f"替换错误: {stats['substitutions']}")
        lines.append(f"删除错误: {stats['deletions']}")
        lines.append(f"插入错误: {stats['insertions']}")

        if stats['total'] > 0:
            accuracy = (stats['correct'] / stats['total']) * 100
            lines.append(f"词级准确率: {accuracy:.2f}%")

        # 工具状态
        lines.append(f"\n【工具状态】")
        lines.append(f"Kaldi对齐工具: {'可用' if comparison_result['kaldi_available'] else '不可用（使用备用对齐）'}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def batch_compare(self, text_pairs: List[Tuple[str, str]],
                     normalize: bool = True) -> List[Dict[str, Any]]:
        """
        批量比较文本对

        Args:
            text_pairs: (reference, hypothesis) 文本对列表
            normalize: 是否进行文本规范化

        Returns:
            比较结果列表
        """
        results = []
        for ref, hyp in text_pairs:
            result = self.compare_texts(ref, hyp, normalize)
            results.append(result)
        return results


class ASRTextEvaluator:
    """ASR文本评估器 - 专门用于ASR结果评估"""

    def __init__(self, language: str = "zh", kaldi_path: Optional[str] = None):
        """
        初始化ASR文本评估器

        Args:
            language: 语言类型
            kaldi_path: Kaldi工具路径
        """
        self.comparator = TextComparator(language, kaldi_path)

    def evaluate_asr_result(self, reference_text: str, asr_text: str,
                           audio_path: Optional[str] = None) -> Dict[str, Any]:
        """
        评估单个ASR结果

        Args:
            reference_text: 参考文本
            asr_text: ASR识别结果
            audio_path: 音频文件路径（可选）

        Returns:
            评估结果
        """
        comparison = self.comparator.compare_texts(reference_text, asr_text)

        # 添加音频信息
        if audio_path:
            comparison['audio_path'] = audio_path

        # 生成简化报告
        stats = comparison['alignment']['statistics']
        simple_report = {
            'wer': (stats['substitutions'] + stats['deletions'] + stats['insertions']) / max(stats['total'], 1),
            'accuracy': stats['correct'] / max(stats['total'], 1),
            'total_words': stats['total'],
            'error_words': stats['substitutions'] + stats['deletions'] + stats['insertions']
        }

        comparison['simple_report'] = simple_report
        return comparison

    def evaluate_asr_batch(self, asr_results: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        批量评估ASR结果

        Args:
            asr_results: 包含reference_text和asr_text的字典列表

        Returns:
            批量评估结果
        """
        individual_results = []
        total_stats = {
            'total_words': 0,
            'correct_words': 0,
            'substitutions': 0,
            'deletions': 0,
            'insertions': 0
        }

        for result in asr_results:
            ref_text = result.get('reference_text', '')
            asr_text = result.get('asr_text', '')
            audio_path = result.get('audio_path', None)

            evaluation = self.evaluate_asr_result(ref_text, asr_text, audio_path)
            individual_results.append(evaluation)

            # 累计统计
            stats = evaluation['alignment']['statistics']
            total_stats['total_words'] += stats['total']
            total_stats['correct_words'] += stats['correct']
            total_stats['substitutions'] += stats['substitutions']
            total_stats['deletions'] += stats['deletions']
            total_stats['insertions'] += stats['insertions']

        # 计算总体指标
        total_errors = (total_stats['substitutions'] +
                       total_stats['deletions'] +
                       total_stats['insertions'])

        batch_summary = {
            'total_samples': len(asr_results),
            'total_words': total_stats['total_words'],
            'overall_wer': total_errors / max(total_stats['total_words'], 1),
            'overall_accuracy': total_stats['correct_words'] / max(total_stats['total_words'], 1),
            'error_breakdown': {
                'substitutions': total_stats['substitutions'],
                'deletions': total_stats['deletions'],
                'insertions': total_stats['insertions']
            },
            'individual_results': individual_results
        }

        return batch_summary