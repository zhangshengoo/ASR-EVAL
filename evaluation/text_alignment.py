"""
基于编辑距离的文本对齐模块
使用动态规划算法进行文本对齐和比较
"""

from typing import List, Tuple, Dict


class TextAligner:
    """基于编辑距离的文本对齐器"""

    def __init__(self):
        """初始化文本对齐器"""
        pass

    def align_text(self, reference: str, hypothesis: str) -> List[Tuple[str, str]]:
        """
        使用编辑距离算法对齐两个文本

        Args:
            reference: 参考文本
            hypothesis: 假设文本

        Returns:
            对齐结果列表，每个元素是(reference_word, hypothesis_word)对
        """
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        return self._edit_distance_alignment(ref_words, hyp_words)

    def _edit_distance_alignment(self, ref_words: List[str], hyp_words: List[str]) -> List[Tuple[str, str]]:
        """
        使用动态规划计算编辑距离对齐

        Args:
            ref_words: 参考文本分词结果
            hyp_words: 假设文本分词结果

        Returns:
            对齐结果列表
        """
        m, n = len(ref_words), len(hyp_words)

        # 创建DP表
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # 初始化边界条件
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # 填充DP表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j] + 1,    # 删除
                        dp[i][j-1] + 1,    # 插入
                        dp[i-1][j-1] + 1   # 替换
                    )

        # 回溯构造对齐结果
        alignment = []
        i, j = m, n

        while i > 0 and j > 0:
            if ref_words[i-1] == hyp_words[j-1]:
                alignment.append((ref_words[i-1], hyp_words[j-1]))
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j-1] + 1:
                # 替换
                alignment.append((ref_words[i-1], hyp_words[j-1]))
                i -= 1
                j -= 1
            elif dp[i][j] == dp[i-1][j] + 1:
                # 删除
                alignment.append((ref_words[i-1], ""))
                i -= 1
            else:
                # 插入
                alignment.append(("", hyp_words[j-1]))
                j -= 1

        # 处理剩余部分
        while i > 0:
            alignment.append((ref_words[i-1], ""))
            i -= 1
        while j > 0:
            alignment.append(("", hyp_words[j-1]))
            j -= 1

        # 反转结果，因为我们是从后往前构造的
        alignment.reverse()

        return alignment

    def format_alignment(self, alignment: List[Tuple[str, str]]) -> str:
        """格式化对齐结果"""
        lines = []
        max_ref_len = max(len(pair[0]) for pair in alignment) if alignment else 0
        max_hyp_len = max(len(pair[1]) for pair in alignment) if alignment else 0

        lines.append("=" * (max_ref_len + max_hyp_len + 15))
        lines.append(f"{'Reference':<{max_ref_len}} | {'Hypothesis':<{max_hyp_len}} | Status")
        lines.append("=" * (max_ref_len + max_hyp_len + 15))

        for ref, hyp in alignment:
            status = "✓" if ref == hyp else "✗"
            lines.append(f"{ref:<{max_ref_len}} | {hyp:<{max_hyp_len}} | {status}")

        lines.append("=" * (max_ref_len + max_hyp_len + 15))

        return "\n".join(lines)

    def calculate_error_statistics(self, alignment: List[Tuple[str, str]]) -> Dict[str, int]:
        """计算错误统计"""
        total = len(alignment)
        correct = sum(1 for ref, hyp in alignment if ref == hyp and ref != "")
        substitutions = sum(1 for ref, hyp in alignment if ref and hyp and ref != hyp)
        deletions = sum(1 for ref, hyp in alignment if ref and not hyp)
        insertions = sum(1 for ref, hyp in alignment if not ref and hyp)

        return {
            "total": total,
            "correct": correct,
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions
        }

    def print_alignment_demo(self, reference: str, hypothesis: str) -> None:
        """打印对齐结果的演示格式"""
        alignment = self.align_text(reference, hypothesis)

        ref_line = "ref : "
        hyp_line = "hyp : "

        for ref, hyp in alignment:
            # 处理空字符串显示
            ref_display = ref if ref else "**"
            hyp_display = hyp if hyp else "**"

            ref_line += f"{ref_display} "
            hyp_line += f"{hyp_display} "

        print(ref_line.rstrip())
        print(hyp_line.rstrip())

        # 显示统计信息
        stats = self.calculate_error_statistics(alignment)
        print(f"\n对齐统计:")
        print(f"总词数: {stats['total']}")
        print(f"正确: {stats['correct']}")
        print(f"替换: {stats['substitutions']}")
        print(f"删除: {stats['deletions']}")
        print(f"插入: {stats['insertions']}")

    def generate_diff_report(self, reference: str, hypothesis: str) -> Dict[str, any]:
        """生成差异报告"""
        alignment = self.align_text(reference, hypothesis)
        stats = self.calculate_error_statistics(alignment)
        formatted_output = self.format_alignment(alignment)

        return {
            'alignment': alignment,
            'statistics': stats,
            'formatted_output': formatted_output,
            'reference': reference,
            'hypothesis': hypothesis
        }


def demo_edit_distance_alignment():
    """演示编辑距离对齐功能"""
    aligner = TextAligner()

    # 示例文本
    reference = "今 天 天 气 很 好"
    hypothesis = "今 天 天 天 很 号"

    print("=== 编辑距离文本对齐演示 ===")
    print()

    # 使用print_alignment_demo方法打印对齐结果
    aligner.print_alignment_demo(reference, hypothesis)

    print()
    print("=== 详细对齐信息 ===")
    result = aligner.generate_diff_report(reference, hypothesis)
    print(result['formatted_output'])

    return result


if __name__ == "__main__":
    demo_edit_distance_alignment()