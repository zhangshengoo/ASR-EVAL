"""
热词评估指标模块 - 简洁核心版
基于文本对齐结果计算召回率和精确率
复用TextAligner接口
"""

from typing import List, Dict, Tuple
from evaluation.text_alignment import TextAligner


class HotwordMetricsCalculator:
    """热词指标计算器 - 核心逻辑版"""

    def __init__(self):
        self.hotwords = []
        self.aligner = TextAligner()

    def load_hotwords(self, hotwords: List[str]) -> None:
        """加载热词列表"""
        self.hotwords = [hw.strip() for hw in hotwords if hw.strip()]

    def calculate_metrics(self, reference: str, hypothesis: str, include_alignment: bool = False) -> Dict:
        """
        计算热词召回率和精确率

        Args:
            reference: 参考文本
            hypothesis: 预测文本
            include_alignment: 是否包含对齐详情（可选，保持向后兼容）

        Returns:
            {'recall': float, 'precision': float} 或
            {'recall': float, 'precision': float, 'alignment_details': dict}（如果include_alignment=True）
        """
        if not self.hotwords:
            if include_alignment:
                return {'recall': 0.0, 'precision': 0.0, 'alignment_details': {}}
            return {'recall': 0.0, 'precision': 0.0}

        # 文本预处理
        ref_text = self._normalize_text(reference)
        hyp_text = self._normalize_text(hypothesis)

        # 基于对齐计算指标
        metrics = self._calculate_aligned_metrics(ref_text, hyp_text)

        if include_alignment:
            # 获取对齐详情用于外部使用
            alignment = self._align_text(ref_text, hyp_text)
            alignment_details = {
                'alignment': alignment,
                'reference_text': ref_text,
                'hypothesis_text': hyp_text
            }
            metrics['alignment_details'] = alignment_details

        return metrics

    def _normalize_text(self, text: str) -> str:
        """文本规范化"""
        return text.lower().strip()

    def _calculate_aligned_metrics(self, ref_text: str, hyp_text: str) -> Dict[str, float]:
        """基于对齐结果计算指标"""
        # 文本对齐
        alignment = self._align_text(ref_text, hyp_text)

        # 统计热词匹配
        hotword_stats = self._count_hotword_matches(ref_text, hyp_text, alignment)

        # 计算指标
        total_ref = sum(stats['ref_count'] for stats in hotword_stats.values())
        total_hyp = sum(stats['hyp_count'] for stats in hotword_stats.values())
        total_matches = sum(stats['matches'] for stats in hotword_stats.values())

        recall = total_matches / total_ref if total_ref > 0 else 0.0
        precision = total_matches / total_hyp if total_hyp > 0 else 0.0

        return {'recall': recall, 'precision': precision}

    def _align_text(self, ref: str, hyp: str) -> List[Tuple[str, str]]:
        """使用TextAligner进行文本对齐"""
        return self.aligner.align_text(ref, hyp)

    def _count_hotword_matches(self, ref_text: str, hyp_text: str, alignment: List[Tuple[str, str]]) -> Dict[str, Dict[str, int]]:
        """统计热词匹配情况"""
        # 构建位置映射
        ref_to_hyp_map = self._build_position_map(alignment)

        stats = {}

        for hotword in self.hotwords:
            # 查找参考文本中的热词
            ref_positions = self._find_occurrences(ref_text, hotword)
            # 查找预测文本中的热词
            hyp_positions = self._find_occurrences(hyp_text, hotword)

            # 基于对齐匹配
            matches = self._match_occurrences(ref_positions, hyp_positions, ref_to_hyp_map)

            stats[hotword] = {
                'ref_count': len(ref_positions),
                'hyp_count': len(hyp_positions),
                'matches': matches
            }

        return stats

    def _build_position_map(self, alignment: List[Tuple[str, str]]) -> Dict[int, int]:
        """构建参考到预测的位置映射"""
        ref_pos, hyp_pos = 0, 0
        pos_map = {}

        for ref_char, hyp_char in alignment:
            if ref_char and hyp_char:  # 匹配
                pos_map[ref_pos] = hyp_pos
                ref_pos += 1
                hyp_pos += 1
            elif ref_char:  # 删除
                ref_pos += 1
            elif hyp_char:  # 插入
                hyp_pos += 1

        return pos_map

    def _find_occurrences(self, text: str, hotword: str) -> List[int]:
        """查找热词出现位置"""
        positions = []
        text_lower = text.lower()
        hw_lower = hotword.lower()
        start = 0

        while True:
            pos = text_lower.find(hw_lower, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1  # 允许重叠

        return positions

    def _match_occurrences(self, ref_pos: List[int], hyp_pos: List[int], pos_map: Dict[int, int]) -> int:
        """严格字符级匹配热词出现，返回匹配数量

        要求字符级别完全一致，不允许位置容差
        """
        matches = 0
        used_hyp = set()

        for ref_p in ref_pos:
            # 严格检查：必须存在精确的位置映射
            if ref_p in pos_map:
                mapped_hyp_p = pos_map[ref_p]

                # 在预测文本中查找完全相同的位置
                if mapped_hyp_p in hyp_pos and mapped_hyp_p not in used_hyp:
                    matches += 1
                    used_hyp.add(mapped_hyp_p)

        return matches

    def batch_calculate(self, test_data: List[Dict[str, str]]) -> Dict[str, float]:
        """批量计算热词指标"""
        total_recall = 0.0
        total_precision = 0.0

        for data in test_data:
            metrics = self.calculate_metrics(
                data.get('reference_text', ''),
                data.get('predicted_text', '')
            )
            total_recall += metrics['recall']
            total_precision += metrics['precision']

        count = len(test_data) if test_data else 1
        return {
            'avg_recall': total_recall / count,
            'avg_precision': total_precision / count
        }