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

    def generate_diff_report(self, reference: str, hypothesis: str) -> Dict[str, any]:
        """生成文本差异报告，包含基于编辑距离的可视化"""
        # 创建对齐结果
        alignment = self.align_text(reference, hypothesis)

        stats = self.calculate_error_statistics(alignment)
        formatted = self.format_alignment(alignment)

        # 使用编辑距离结果生成可视化
        visualizer = TextDiffVisualizer()
        ref_colored, hyp_colored = visualizer.color_diff_from_alignment(alignment)
        side_by_side = visualizer.side_by_side_diff_from_alignment(alignment)
        word_level = visualizer.word_level_diff(alignment)

        return {
            "reference": reference,
            "hypothesis": hypothesis,
            "alignment": alignment,
            "statistics": stats,
            "formatted_output": formatted,
            "visualization": {
                "colored_reference": ref_colored,
                "colored_hypothesis": hyp_colored,
                "side_by_side": side_by_side,
                "word_level_diff": word_level
            }
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


class TextDiffVisualizer:
    """基于编辑距离结果的文本差异可视化器"""

    @staticmethod
    def color_diff_from_alignment(alignment: List[Tuple[str, str]]) -> Tuple[str, str]:
        """基于align-text对齐结果生成彩色差异显示"""
        ref_colored = []
        hyp_colored = []

        for ref_word, hyp_word in alignment:
            if ref_word == hyp_word and ref_word != "":
                # 匹配的词 - 正常显示
                ref_colored.append(ref_word)
                hyp_colored.append(hyp_word)
            elif ref_word and hyp_word and ref_word != hyp_word:
                # 替换错误 - 红色(参考)和绿色(假设)
                ref_colored.append(f"\033[91m{ref_word}\033[0m")
                hyp_colored.append(f"\033[92m{hyp_word}\033[0m")
            elif ref_word and not hyp_word:
                # 删除错误 - 红色删除线
                ref_colored.append(f"\033[91m~~{ref_word}~~\033[0m")
                hyp_colored.append("\033[90m[DEL]\033[0m")
            elif not ref_word and hyp_word:
                # 插入错误 - 绿色下划线
                ref_colored.append("\033[90m[INS]\033[0m")
                hyp_colored.append(f"\033[92m_{hyp_word}_\033[0m")

            # 添加空格分隔
            ref_colored.append(" ")
            hyp_colored.append(" ")

        return "".join(ref_colored).strip(), "".join(hyp_colored).strip()

    @staticmethod
    def side_by_side_diff_from_alignment(alignment: List[Tuple[str, str]], width: int = 30) -> str:
        """基于align-text对齐结果生成并排差异显示"""
        lines = []
        lines.append("=" * (width * 2 + 15))
        lines.append(f"{'Reference Text':<{width}} | {'Hypothesis Text':<{width}} | Status")
        lines.append("=" * (width * 2 + 15))

        for ref_word, hyp_word in alignment:
            status = ""
            ref_display = ref_word if ref_word else ""
            hyp_display = hyp_word if hyp_word else ""

            if ref_word == hyp_word and ref_word != "":
                status = "✓"
            elif ref_word and hyp_word and ref_word != hyp_word:
                status = "替换"
                ref_display = f"[{ref_word}]"
                hyp_display = f"[{hyp_word}]"
            elif ref_word and not hyp_word:
                status = "删除"
                ref_display = f"~~{ref_word}~~"
                hyp_display = ""
            elif not ref_word and hyp_word:
                status = "插入"
                ref_display = ""
                hyp_display = f"++{hyp_word}++"

            lines.append(f"{ref_display:<{width}} | {hyp_display:<{width}} | {status}")

        lines.append("=" * (width * 2 + 15))
        return "\n".join(lines)


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

    @staticmethod
    def word_level_diff(alignment: List[Tuple[str, str]]) -> str:
        """基于align-text结果生成词级差异分析"""
        lines = []
        lines.append("\n【词级差异分析】")
        lines.append("-" * 40)

        for i, (ref_word, hyp_word) in enumerate(alignment, 1):
            if ref_word == hyp_word and ref_word != "":
                lines.append(f"{i:2d}. ✓ {ref_word}")
            elif ref_word and hyp_word and ref_word != hyp_word:
                lines.append(f"{i:2d}. ✗ {ref_word} → {hyp_word} (替换)")
            elif ref_word and not hyp_word:
                lines.append(f"{i:2d}. ✗ {ref_word} → [删除]")
            elif not ref_word and hyp_word:
                lines.append(f"{i:2d}. ✗ [插入] → {hyp_word}")

        return "\n".join(lines)


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

    @staticmethod
    def color_diff(reference: str, hypothesis: str) -> Tuple[str, str]:
        """保持向后兼容 - 使用备用对齐方法"""
        from difflib import SequenceMatcher

        matcher = SequenceMatcher(None, reference, hypothesis)
        ref_colored = []
        hyp_colored = []

        for opcode, ref_start, ref_end, hyp_start, hyp_end in matcher.get_opcodes():
            ref_part = reference[ref_start:ref_end]
            hyp_part = hypothesis[hyp_start:hyp_end]

            if opcode == 'equal':
                ref_colored.append(ref_part)
                hyp_colored.append(hyp_part)
            elif opcode == 'delete':
                ref_colored.append(f"\033[91m{ref_part}\033[0m")
                hyp_colored.append("" * len(ref_part))
            elif opcode == 'insert':
                ref_colored.append("" * len(hyp_part))
                hyp_colored.append(f"\033[92m{hyp_part}\033[0m")
            elif opcode == 'replace':
                ref_colored.append(f"\033[91m{ref_part}\033[0m")
                hyp_colored.append(f"\033[92m{hyp_part}\033[0m")

        return "".join(ref_colored), "".join(hyp_colored)

    @staticmethod
    def side_by_side_diff(reference: str, hypothesis: str, width: int = 50) -> str:
        """保持向后兼容 - 使用备用对齐方法"""
        from difflib import ndiff

        diff_lines = list(ndiff(reference.split(), hypothesis.split()))

        lines = []
        lines.append("=" * (width * 2 + 10))
        lines.append(f"{'Reference':<{width}} | {'Hypothesis':<{width}}")
        lines.append("=" * (width * 2 + 10))

        ref_line = []
        hyp_line = []

        for line in diff_lines:
            if line.startswith("  "):
                ref_line.append(line[2:])
                hyp_line.append(line[2:])
            elif line.startswith("- "):
                ref_line.append(f"[{line[2:]}]")
            elif line.startswith("+ "):
                hyp_line.append(f"[{line[2:]}]")

                if ref_line or hyp_line:
                    lines.append(f"{' '.join(ref_line):<{width}} | {' '.join(hyp_line):<{width}}")
                    ref_line = []
                    hyp_line = []

        if ref_line or hyp_line:
            lines.append(f"{' '.join(ref_line):<{width}} | {' '.join(hyp_line):<{width}}")

        lines.append("=" * (width * 2 + 10))

        return "\n".join(lines)


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