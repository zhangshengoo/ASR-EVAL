"""
Kaldi文本对齐工具集成模块
使用Kaldi的align-text工具进行文本对齐和比较
"""

import subprocess
import tempfile
import os
from typing import List, Tuple, Dict, Optional


class TextAligner:
    """Kaldi文本对齐器"""

    def __init__(self, kaldi_path: Optional[str] = None):
        """
        初始化文本对齐器

        Args:
            kaldi_path: Kaldi工具的路径，如果为None则使用系统PATH
        """
        self.kaldi_path = kaldi_path or ""
        self.align_text_cmd = self._find_kaldi_tool("align-text")

    def _find_kaldi_tool(self, tool_name: str) -> str:
        """查找Kaldi工具路径"""
        if self.kaldi_path:
            tool_path = os.path.join(self.kaldi_path, "src", "bin", tool_name)
            if os.path.exists(tool_path):
                return tool_path

        # 尝试系统PATH
        try:
            result = subprocess.run(["which", tool_name],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass

        return tool_name

    def align_text(self, reference: str, hypothesis: str) -> List[Tuple[str, str]]:
        """
        使用align-text工具对齐两个文本

        Args:
            reference: 参考文本
            hypothesis: 假设文本

        Returns:
            对齐结果列表，每个元素是(reference_word, hypothesis_word)对
        """
        if not self._check_kaldi_available():
            return self._fallback_alignment(reference, hypothesis)

        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ref', delete=False) as ref_file:
            ref_file.write(reference)
            ref_path = ref_file.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.hyp', delete=False) as hyp_file:
            hyp_file.write(hypothesis)
            hyp_path = hyp_file.name

        try:
            # 执行align-text命令
            cmd = [self.align_text_cmd, "ark:" + ref_path, "ark:" + hyp_path, "ark:-"]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Kaldi对齐失败: {result.stderr}")
                return self._fallback_alignment(reference, hypothesis)

            return self._parse_alignment_output(result.stdout)

        finally:
            # 清理临时文件
            os.unlink(ref_path)
            os.unlink(hyp_path)

    def _check_kaldi_available(self) -> bool:
        """检查Kaldi工具是否可用"""
        try:
            result = subprocess.run([self.align_text_cmd, "--help"],
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except:
            return False

    def _fallback_alignment(self, reference: str, hypothesis: str) -> List[Tuple[str, str]]:
        """备用对齐方法（当Kaldi不可用时使用）"""
        ref_words = reference.split()
        hyp_words = hypothesis.split()

        # 简单的逐词对齐
        alignment = []
        max_len = max(len(ref_words), len(hyp_words))

        for i in range(max_len):
            ref_word = ref_words[i] if i < len(ref_words) else ""
            hyp_word = hyp_words[i] if i < len(hyp_words) else ""
            alignment.append((ref_word, hyp_word))

        return alignment

    def _parse_alignment_output(self, output: str) -> List[Tuple[str, str]]:
        """解析align-text输出"""
        alignment = []
        lines = output.strip().split('\n')

        for line in lines:
            if not line.strip():
                continue

            # 解析格式: utt-id ref-word hyp-word
            parts = line.strip().split()
            if len(parts) >= 3:
                # 跳过utt-id，取ref和hyp词
                ref_word = parts[1] if parts[1] != "*" else ""
                hyp_word = parts[2] if parts[2] != "*" else ""
                alignment.append((ref_word, hyp_word))

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
        correct = sum(1 for ref, hyp in alignment if ref == hyp)
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
        """生成文本差异报告，包含基于align-text的可视化"""
        # 创建对齐结果
        alignment = self.align_text(reference, hypothesis)

        # 如果Kaldi不可用，使用逐词对齐
        if not self._check_kaldi_available():
            alignment = self._fallback_alignment(reference, hypothesis)

        stats = self.calculate_error_statistics(alignment)
        formatted = self.format_alignment(alignment)

        # 使用align-text结果生成可视化
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


class TextDiffVisualizer:
    """基于Kaldi align-text结果的文本差异可视化器"""

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