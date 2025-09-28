"""
测试结果管理器
负责管理ASR模型测试结果的输出和格式化处理
"""

import os
import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.models import TestItem, TestResult


class TestResultsManager:
    """测试结果管理器"""

    def __init__(self, base_path: str = "results"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save_results(self, results: List[TestResult], model_name: str,
                    dataset_name: str, metadata: Dict[str, Any] = None) -> str:
        """保存测试结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = self.base_path / f"{model_name}_{dataset_name}_{timestamp}"
        result_dir.mkdir(parents=True, exist_ok=True)

        # 保存基础结果
        basic_results = self._format_basic_results(results, metadata)

        # 保存不同格式的结果
        self._save_json_results(results, result_dir / "results.json")
        self._save_csv_results(results, result_dir / "results.csv")
        self._save_detailed_results(results, result_dir / "detailed.json")
        self._save_list_results(results, result_dir / "results.list", model_name, dataset_name)

        # 保存热词测试结果（如果有）
        if metadata and metadata.get("test_type") == "hotword":
            self._save_hotword_results(results, result_dir)
            # 同时保存到对应的模型目录结构
            self._save_to_model_directory(results, model_name, dataset_name, timestamp)

        self.logger.info(f"测试结果已保存到: {result_dir}")
        return str(result_dir)

    def _format_basic_results(self, results: List[TestResult],
                            metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """格式化基础结果"""
        total_items = len(results)
        if total_items == 0:
            return {"error": "无测试结果"}

        # 计算基础指标 - 使用字符串匹配作为简单评估
        correct_count = sum(1 for r in results if r.reference_text == r.predicted_text)

        # 计算简单错误率（基于字符串匹配）
        error_rates = []
        for r in results:
            if r.reference_text and r.predicted_text:
                ref_words = r.reference_text.split()
                pred_words = r.predicted_text.split()
                if ref_words:
                    # 简单的错误率计算
                    correct_words = sum(1 for w in ref_words if w in pred_words)
                    error_rate = 1 - (correct_words / len(ref_words))
                    error_rates.append(max(0, min(1, error_rate)))

        avg_wer = sum(error_rates) / len(error_rates) if error_rates else 0

        # 计算RTF（假设音频时长=处理时间*10）
        total_rtf = sum(r.processing_time * 10 for r in results if r.processing_time is not None)
        avg_rtf = total_rtf / total_items if total_items > 0 else 0

        # 计算置信度统计
        confidences = [r.confidence_score for r in results if r.confidence_score is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0

        basic_results = {
            "summary": {
                "total_items": total_items,
                "correct_count": correct_count,
                "accuracy": correct_count / total_items,
                "average_wer": avg_wer,
                "average_rtf": avg_rtf,
                "average_confidence": avg_confidence,
                "model_name": metadata.get("model_name", "unknown") if metadata else "unknown",
                "dataset_name": metadata.get("dataset_name", "unknown") if metadata else "unknown",
                "timestamp": datetime.now().isoformat()
            },
            "results_count": total_items
        }

        return basic_results

    def _save_json_results(self, results: List[TestResult], output_path: Path):
        """保存JSON格式的结果"""
        json_data = []
        for result in results:
            # 计算简单的错误率
            ref_words = result.reference_text.split() if result.reference_text else []
            pred_words = result.predicted_text.split() if result.predicted_text else []

            correct_words = 0
            if ref_words and pred_words:
                correct_words = sum(1 for w in ref_words if w in pred_words)

            wer = 1 - (correct_words / len(ref_words)) if ref_words else 0

            # 计算字符错误率 (CER)
            ref_chars = list(result.reference_text) if result.reference_text else []
            pred_chars = list(result.predicted_text) if result.predicted_text else []

            correct_chars = 0
            if ref_chars and pred_chars:
                correct_chars = sum(1 for c in ref_chars if c in pred_chars)

            cer = 1 - (correct_chars / len(ref_chars)) if ref_chars else 0

            json_data.append({
                "audio_path": str(result.audio_path),
                "reference_text": result.reference_text,
                "predicted_text": result.predicted_text,
                "wer": round(wer, 4),
                "cer": round(cer, 4),
                "confidence": result.confidence_score,
                "processing_time": result.processing_time
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)

    def _save_csv_results(self, results: List[TestResult], output_path: Path):
        """保存CSV格式的结果"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "audio_path", "reference_text", "predicted_text",
                "wer", "cer", "confidence", "processing_time", "rtf"
            ])

            for result in results:
                # 计算WER和CER
                ref_words = result.reference_text.split() if result.reference_text else []
                pred_words = result.predicted_text.split() if result.predicted_text else []

                correct_words = 0
                if ref_words and pred_words:
                    correct_words = sum(1 for w in ref_words if w in pred_words)
                wer = 1 - (correct_words / len(ref_words)) if ref_words else 0

                ref_chars = list(result.reference_text) if result.reference_text else []
                pred_chars = list(result.predicted_text) if result.predicted_text else []

                correct_chars = 0
                if ref_chars and pred_chars:
                    correct_chars = sum(1 for c in ref_chars if c in pred_chars)
                cer = 1 - (correct_chars / len(ref_chars)) if ref_chars else 0

                # 计算RTF
                rtf = result.processing_time * 10 if result.processing_time else 0

                writer.writerow([
                    str(result.audio_path),
                    result.reference_text,
                    result.predicted_text,
                    round(wer, 4),
                    round(cer, 4),
                    result.confidence_score,
                    result.processing_time,
                    round(rtf, 4)
                ])

    def _save_detailed_results(self, results: List[TestResult], output_path: Path):
        """保存详细结果（包含分析）"""
        detailed_data = {
            "metadata": {
                "total_items": len(results),
                "timestamp": datetime.now().isoformat(),
                "model_info": self._extract_model_info(results)
            },
            "summary": self._generate_summary(results),
            "detailed_results": []
        }

        for result in results:
            # 计算WER和CER
            ref_words = result.reference_text.split() if result.reference_text else []
            pred_words = result.predicted_text.split() if result.predicted_text else []

            correct_words = 0
            if ref_words and pred_words:
                correct_words = sum(1 for w in ref_words if w in pred_words)
            wer = 1 - (correct_words / len(ref_words)) if ref_words else 0

            ref_chars = list(result.reference_text) if result.reference_text else []
            pred_chars = list(result.predicted_text) if result.predicted_text else []

            correct_chars = 0
            if ref_chars and pred_chars:
                correct_chars = sum(1 for c in ref_chars if c in pred_chars)
            cer = 1 - (correct_chars / len(ref_chars)) if ref_chars else 0

            # 计算RTF
            rtf = result.processing_time * 10 if result.processing_time else 0

            detailed_result = {
                "file_info": {
                    "path": str(result.audio_path),
                    "filename": Path(result.audio_path).name
                },
                "text_info": {
                    "reference": result.reference_text,
                    "predicted": result.predicted_text,
                    "match": result.reference_text == result.predicted_text
                },
                "metrics": {
                    "wer": round(wer, 4),
                    "cer": round(cer, 4),
                    "confidence": result.confidence_score,
                    "rtf": round(rtf, 4),
                    "processing_time": result.processing_time
                },
                "metadata": {}
            }
            detailed_data["detailed_results"].append(detailed_result)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)

    def _save_to_model_directory(self, results: List[TestResult], model_name: str, dataset_name: str, timestamp: str):
        """保存结果到模型专用目录结构"""
        # 转换模型名称为小写并标准化
        model_dir_name = model_name.lower().replace("-", "").replace("_", "")

        # 解析数据集名称 - 假设格式为 "场景1_热词测试" 或 "场景1"
        if "_" in dataset_name:
            parts = dataset_name.split("_", 1)
            scene_name = parts[0]  # "场景1"
            test_type = parts[1] if len(parts) > 1 else "测试"  # "热词测试"
        else:
            scene_name = dataset_name
            test_type = "测试"

        # 创建目录结构: results/{model_name}/{test_type}/{scene_name}/
        model_result_dir = self.base_path / model_dir_name / test_type / scene_name
        model_result_dir.mkdir(parents=True, exist_ok=True)

        # 保存结果文件 - 使用固定名称（不带时间戳）
        self._save_list_results(results, model_result_dir / f"{scene_name}.list", model_name, dataset_name)
        self._save_json_results(results, model_result_dir / f"{scene_name}.json")
        self._save_csv_results(results, model_result_dir / f"{scene_name}.csv")

        # 保存元数据信息
        metadata = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "scene_name": scene_name,
            "test_type": test_type,
            "timestamp": timestamp,
            "total_items": len(results),
            "results_summary": self._format_basic_results(results)
        }

        with open(model_result_dir / f"{scene_name}_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self.logger.info(f"模型结果已保存到: {model_result_dir}")

    def _save_list_results(self, results: List[TestResult], output_path: Path, model_name: str, dataset_name: str):
        """保存.list格式的结果，匹配参考格式"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                # 提取文件名（不含路径和扩展名）
                filename = Path(result.audio_path).stem
                # 使用预测文本
                predicted_text = result.predicted_text.strip()
                # 写入格式：filename | predicted_text
                f.write(f"{filename} | {predicted_text}\n")

    def _save_hotword_results(self, results: List[TestResult], result_dir: Path):
        """保存热词测试结果 - 简化版本"""
        hotword_data = {
            "hotword_analysis": {},
            "scene_summary": {
                "default": {
                    "total_items": len(results),
                    "correct_items": 0,
                    "accuracy": 0.0,
                    "average_wer": 0.0
                }
            },
            "results": []
        }

        if not results:
            with open(result_dir / "hotword_analysis.json", 'w', encoding='utf-8') as f:
                json.dump(hotword_data, f, ensure_ascii=False, indent=2)
            return

        # 计算WER并填充结果
        wers = []
        for result in results:
            ref_words = result.reference_text.split() if result.reference_text else []
            pred_words = result.predicted_text.split() if result.predicted_text else []
            correct_words = 0
            if ref_words and pred_words:
                correct_words = sum(1 for w in ref_words if w in pred_words)
            wer = 1 - (correct_words / len(ref_words)) if ref_words else 0
            wers.append(wer)

            hotword_data["results"].append({
                "audio_path": str(result.audio_path),
                "reference": result.reference_text,
                "predicted": result.predicted_text,
                "wer": round(wer, 4),
                "correct": wer == 0
            })

        correct = sum(1 for wer in wers if wer == 0)
        avg_wer = sum(wers) / len(results)

        hotword_data["scene_summary"]["default"]["correct_items"] = correct
        hotword_data["scene_summary"]["default"]["accuracy"] = correct / len(results)
        hotword_data["scene_summary"]["default"]["average_wer"] = avg_wer

        # 保存热词结果
        with open(result_dir / "hotword_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(hotword_data, f, ensure_ascii=False, indent=2)

    def _extract_model_info(self, results: List[TestResult]) -> Dict[str, Any]:
        """提取模型信息"""
        if not results:
            return {}

        # 从metadata参数或默认信息中获取模型信息
        return {
            "model_name": "Step-Audio-2-mini",
            "model_type": "StepAudio2",
            "hotwords": []
        }

    def _generate_summary(self, results: List[TestResult]) -> Dict[str, Any]:
        """生成结果摘要"""
        if not results:
            return {}

        # 默认场景分组
        scene_summary = {
            "default": {
                "total_items": len(results),
                "correct_items": 0,
                "accuracy": 0.0,
                "average_wer": 0.0
            }
        }

        if not results:
            return scene_summary

        # 计算WER
        wers = []
        for r in results:
            ref_words = r.reference_text.split() if r.reference_text else []
            pred_words = r.predicted_text.split() if r.predicted_text else []
            correct_words = 0
            if ref_words and pred_words:
                correct_words = sum(1 for w in ref_words if w in pred_words)
            wer = 1 - (correct_words / len(ref_words)) if ref_words else 0
            wers.append(wer)

        correct = sum(1 for wer in wers if wer == 0)
        avg_wer = sum(wers) / len(results)

        scene_summary["default"] = {
            "total_items": len(results),
            "correct_items": correct,
            "accuracy": correct / len(results),
            "average_wer": avg_wer
        }

        return scene_summary

    def load_results(self, result_path: str) -> List[Dict[str, Any]]:
        """加载已保存的结果"""
        result_path = Path(result_path)
        json_file = result_path / "results.json"

        if not json_file.exists():
            self.logger.warning(f"结果文件不存在: {json_file}")
            return []

        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def generate_report(self, results: List[TestResult], output_path: str,
                       template: str = "basic") -> str:
        """生成测试报告"""
        report_path = Path(output_path) / "report.html"

        html_template = self._get_html_template(template)
        report_content = html_template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_items=len(results),
            summary=self._format_summary_for_report(results),
            detailed_results=self._format_detailed_results_for_report(results)
        )

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        return str(report_path)

    def _get_html_template(self, template: str) -> str:
        """获取HTML模板"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>ASR测试结果报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ background: #f0f0f0; padding: 15px; margin: 10px 0; }}
        .result {{ border: 1px solid #ddd; margin: 10px 0; padding: 10px; }}
        .correct {{ background: #d4edda; }}
        .incorrect {{ background: #f8d7da; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>ASR测试结果报告</h1>
    <div class="summary">
        <h2>测试摘要</h2>
        <p>测试时间: {timestamp}</p>
        <p>总测试项: {total_items}</p>
        {summary}
    </div>
    <div class="detailed-results">
        {detailed_results}
    </div>
</body>
</html>
        """

    def _format_summary_for_report(self, results: List[TestResult]) -> str:
        """格式化摘要用于报告"""
        if not results:
            return "<p>无测试结果</p>"

        total = len(results)

        # 计算WER
        wers = []
        for r in results:
            ref_words = r.reference_text.split() if r.reference_text else []
            pred_words = r.predicted_text.split() if r.predicted_text else []
            correct_words = 0
            if ref_words and pred_words:
                correct_words = sum(1 for w in ref_words if w in pred_words)
            wer = 1 - (correct_words / len(ref_words)) if ref_words else 0
            wers.append(wer)

        correct = sum(1 for wer in wers if wer == 0)
        avg_wer = sum(wers) / total

        return f"""
        <p>准确率: {(correct/total)*100:.2f}%</p>
        <p>平均词错误率: {avg_wer:.4f}</p>
        <p>正确识别: {correct}/{total}</p>
        """

    def _format_detailed_results_for_report(self, results: List[TestResult]) -> str:
        """格式化详细结果用于报告"""
        if not results:
            return "<p>无详细结果</p>"

        html = "<table><tr><th>音频文件</th><th>参考文本</th><th>识别文本</th><th>WER</th><th>状态</th></tr>"

        for result in results:
            # 计算WER
            ref_words = result.reference_text.split() if result.reference_text else []
            pred_words = result.predicted_text.split() if result.predicted_text else []
            correct_words = 0
            if ref_words and pred_words:
                correct_words = sum(1 for w in ref_words if w in pred_words)
            wer = 1 - (correct_words / len(ref_words)) if ref_words else 0

            status = "正确" if wer == 0 else "错误"
            css_class = "correct" if wer == 0 else "incorrect"

            html += f"""
            <tr class="{css_class}">
                <td>{Path(result.audio_path).name}</td>
                <td>{result.reference_text}</td>
                <td>{result.predicted_text}</td>
                <td>{wer:.4f}</td>
                <td>{status}</td>
            </tr>
            """

        html += "</table>"
        return html