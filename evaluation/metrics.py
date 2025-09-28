"""
评估指标计算模块
计算各种ASR相关的评估指标
"""

import jiwer
import editdistance
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass

from core.models import TestResult, Metrics
from core.enums import MetricType


@dataclass
class MetricConfig:
    """指标配置"""
    calculate_wer: bool = True
    calculate_cer: bool = True
    calculate_ser: bool = True
    calculate_rtf: bool = True
    calculate_confidence: bool = True
    language: str = "zh"
    normalize_text: bool = True


class TextNormalizer:
    """文本规范化器"""

    def __init__(self, language: str = "zh"):
        self.language = language

    def normalize(self, text: str) -> str:
        """规范化文本"""
        if not text:
            return ""

        # 转换为小写
        text = text.lower()

        # 移除标点符号
        import re
        text = re.sub(r'[^\w\s]', '', text)

        # 移除多余空格
        text = ' '.join(text.split())

        # 中文特殊处理
        if self.language == "zh":
            # 移除空格
            text = text.replace(' ', '')
            # 数字转中文
            text = self._convert_numbers_to_chinese(text)

        return text.strip()

    def _convert_numbers_to_chinese(self, text: str) -> str:
        """将数字转换为中文"""
        # 简化的数字转换
        digit_map = {
            '0': '零', '1': '一', '2': '二', '3': '三', '4': '四',
            '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'
        }

        for digit, chinese in digit_map.items():
            text = text.replace(digit, chinese)

        return text


class WERCalculator:
    """词错误率计算器"""

    def __init__(self, normalizer: TextNormalizer):
        self.normalizer = normalizer

    def calculate(self, results: List[TestResult]) -> float:
        """计算WER"""
        if not results:
            return 0.0

        references = []
        hypotheses = []

        for result in results:
            ref_text = self.normalizer.normalize(result.reference_text)
            hyp_text = self.normalizer.normalize(result.predicted_text)

            if ref_text and hyp_text:
                references.append(ref_text)
                hypotheses.append(hyp_text)

        if not references:
            return 0.0

        try:
            wer = jiwer.wer(references, hypotheses)
            return float(wer)
        except Exception as e:
            print(f"计算WER时出错: {e}")
            return 0.0

    def calculate_per_item(self, result: TestResult) -> float:
        """计算单个结果的WER"""
        ref_text = self.normalizer.normalize(result.reference_text)
        hyp_text = self.normalizer.normalize(result.predicted_text)

        if not ref_text or not hyp_text:
            return 0.0

        try:
            return float(jiwer.wer([ref_text], [hyp_text]))
        except Exception:
            return 0.0


class CERCalculator:
    """字符错误率计算器"""

    def __init__(self, normalizer: TextNormalizer):
        self.normalizer = normalizer

    def calculate(self, results: List[TestResult]) -> float:
        """计算CER"""
        if not results:
            return 0.0

        total_chars = 0
        total_errors = 0

        for result in results:
            ref_text = self.normalizer.normalize(result.reference_text)
            hyp_text = self.normalizer.normalize(result.predicted_text)

            if not ref_text or not hyp_text:
                continue

            # 移除空格计算CER
            ref_chars = list(ref_text.replace(' ', ''))
            hyp_chars = list(hyp_text.replace(' ', ''))

            if not ref_chars:
                continue

            distance = editdistance.eval(ref_chars, hyp_chars)
            total_errors += distance
            total_chars += len(ref_chars)

        return total_errors / total_chars if total_chars > 0 else 0.0

    def calculate_per_item(self, result: TestResult) -> float:
        """计算单个结果的CER"""
        ref_text = self.normalizer.normalize(result.reference_text)
        hyp_text = self.normalizer.normalize(result.predicted_text)

        if not ref_text or not hyp_text:
            return 0.0

        ref_chars = list(ref_text.replace(' ', ''))
        hyp_chars = list(hyp_text.replace(' ', ''))

        if not ref_chars:
            return 0.0

        distance = editdistance.eval(ref_chars, hyp_chars)
        return distance / len(ref_chars)


class SERCalculator:
    """句错误率计算器"""

    def __init__(self, normalizer: TextNormalizer):
        self.normalizer = normalizer

    def calculate(self, results: List[TestResult]) -> float:
        """计算SER"""
        if not results:
            return 0.0

        total_sentences = len(results)
        error_sentences = 0

        for result in results:
            ref_text = self.normalizer.normalize(result.reference_text)
            hyp_text = self.normalizer.normalize(result.predicted_text)

            if ref_text != hyp_text:
                error_sentences += 1

        return error_sentences / total_sentences


class RTFCalculator:
    """实时因子计算器"""

    def calculate(self, results: List[TestResult], audio_durations: Optional[List[float]] = None) -> float:
        """计算实时因子"""
        if not results:
            return 0.0

        total_processing_time = sum(r.processing_time for r in results)

        if audio_durations:
            total_audio_duration = sum(audio_durations)
        else:
            # 如果没有提供音频时长，使用处理时间的平均值作为估计
            total_audio_duration = total_processing_time * 0.8  # 假设处理时间比音频时长短

        return total_processing_time / total_audio_duration if total_audio_duration > 0 else 0.0


class ConfidenceCalculator:
    """置信度计算器"""

    def calculate(self, results: List[TestResult]) -> Dict[str, float]:
        """计算置信度统计"""
        confidences = [r.confidence_score for r in results if r.confidence_score is not None]

        if not confidences:
            return {
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0
            }

        return {
            "avg": float(np.mean(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
            "std": float(np.std(confidences))
        }


class MetricsCalculator:
    """综合指标计算器"""

    def __init__(self, config: Optional[MetricConfig] = None):
        self.config = config or MetricConfig()
        self.normalizer = TextNormalizer(self.config.language)

        self.wer_calculator = WERCalculator(self.normalizer)
        self.cer_calculator = CERCalculator(self.normalizer)
        self.ser_calculator = SERCalculator(self.normalizer)
        self.rtf_calculator = RTFCalculator()
        self.confidence_calculator = ConfidenceCalculator()

    def calculate_metrics(self, results: List[TestResult],
                         audio_durations: Optional[List[float]] = None) -> Metrics:
        """计算所有指标"""
        if not results:
            return Metrics()

        metrics = Metrics()

        # 计算WER
        if self.config.calculate_wer:
            metrics.wer = self.wer_calculator.calculate(results)

        # 计算CER
        if self.config.calculate_cer:
            metrics.cer = self.cer_calculator.calculate(results)

        # 计算SER
        if self.config.calculate_ser:
            metrics.ser = self.ser_calculator.calculate(results)

        # 计算处理时间相关指标
        processing_times = [r.processing_time for r in results]
        metrics.processing_time_avg = float(np.mean(processing_times))

        # 计算实时因子
        if self.config.calculate_rtf:
            metrics.realtime_factor = self.rtf_calculator.calculate(results, audio_durations)

        # 计算置信度指标
        if self.config.calculate_confidence:
            confidence_stats = self.confidence_calculator.calculate(results)
            metrics.confidence_avg = confidence_stats["avg"]
            metrics.confidence_min = confidence_stats["min"]
            metrics.confidence_max = confidence_stats["max"]

        # 计算音频时长平均值（如果提供）
        if audio_durations:
            metrics.audio_duration_avg = float(np.mean(audio_durations))

        return metrics

    def calculate_detailed_metrics(self, results: List[TestResult],
                                 audio_durations: Optional[List[float]] = None) -> Dict[str, Any]:
        """计算详细指标"""
        basic_metrics = self.calculate_metrics(results, audio_durations)

        detailed = {
            "basic_metrics": {
                "wer": basic_metrics.wer,
                "cer": basic_metrics.cer,
                "ser": basic_metrics.ser,
                "processing_time_avg": basic_metrics.processing_time_avg,
                "realtime_factor": basic_metrics.realtime_factor,
                "confidence_avg": basic_metrics.confidence_avg,
                "confidence_min": basic_metrics.confidence_min,
                "confidence_max": basic_metrics.confidence_max,
            },
            "per_item_metrics": [],
            "statistics": {}
        }

        # 计算每个测试项的指标
        for i, result in enumerate(results):
            item_metrics = {}

            if self.config.calculate_wer:
                item_metrics["wer"] = self.wer_calculator.calculate_per_item(result)

            if self.config.calculate_cer:
                item_metrics["cer"] = self.cer_calculator.calculate_per_item(result)

            item_metrics["processing_time"] = result.processing_time
            item_metrics["confidence"] = result.confidence_score

            detailed["per_item_metrics"].append({
                "audio_path": result.audio_path,
                "metrics": item_metrics
            })

        # 计算统计信息
        if results:
            processing_times = [r.processing_time for r in results]
            detailed["statistics"] = {
                "total_samples": len(results),
                "processing_time": {
                    "min": float(np.min(processing_times)),
                    "max": float(np.max(processing_times)),
                    "std": float(np.std(processing_times))
                }
            }

        return detailed