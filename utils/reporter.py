"""
测试报告生成器
生成各种格式的测试报告
"""

import json
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

from core.models import TestReport, ComparativeReport


class TestReporter:
    """测试报告生成器"""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_test_report(self, report: TestReport, output_path: str = None):
        """保存单个测试报告"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_report_{report.model_type.value}_{report.dataset_type.value}_{timestamp}.json"
            output_path = self.output_dir / filename

        # 转换为字典格式
        report_dict = self._report_to_dict(report)

        # 保存JSON格式
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

        # 保存CSV格式
        csv_path = output_path.with_suffix('.csv')
        self._save_csv_report(report, csv_path)

        print(f"测试报告已保存: {output_path}")

    def save_comparative_report(self, report: ComparativeReport, output_path: str = None):
        """保存对比测试报告"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparative_report_{report.dataset_type.value}_{timestamp}.json"
            output_path = self.output_dir / filename

        # 转换为字典格式
        report_dict = self._comparative_report_to_dict(report)

        # 保存JSON格式
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

        # 生成可视化报告
        self._generate_visualization(report, output_path.with_suffix('.png'))

        print(f"对比测试报告已保存: {output_path}")

    def _report_to_dict(self, report: TestReport) -> Dict[str, Any]:
        """将测试报告转换为字典"""
        return {
            "model_type": report.model_type.value,
            "dataset_type": report.dataset_type.value,
            "test_count": report.test_count,
            "created_at": report.created_at.isoformat(),
            "metrics": {
                "wer": report.metrics.wer,
                "cer": report.metrics.cer,
                "ser": report.metrics.ser,
                "processing_time_avg": report.metrics.processing_time_avg,
                "realtime_factor": report.metrics.realtime_factor,
                "confidence_avg": report.metrics.confidence_avg,
                "confidence_min": report.metrics.confidence_min,
                "confidence_max": report.metrics.confidence_max,
                "audio_duration_avg": report.metrics.audio_duration_avg
            },
            "config": report.config,
            "results": [
                {
                    "audio_path": result.audio_path,
                    "model_type": result.model_type.value,
                    "reference_text": result.reference_text,
                    "predicted_text": result.predicted_text,
                    "processing_time": result.processing_time,
                    "confidence_score": result.confidence_score,
                    "word_timestamps": result.word_timestamps,
                    "char_timestamps": result.char_timestamps,
                    "error_details": result.error_details,
                    "created_at": result.created_at.isoformat()
                }
                for result in report.results
            ]
        }

    def _comparative_report_to_dict(self, report: ComparativeReport) -> Dict[str, Any]:
        """将对比测试报告转换为字典"""
        model_reports_dict = {}
        for model_type, model_report in report.model_reports.items():
            model_reports_dict[model_type.value] = self._report_to_dict(model_report)

        return {
            "dataset_type": report.dataset_type.value,
            "test_count": report.test_count,
            "created_at": report.created_at.isoformat(),
            "model_reports": model_reports_dict,
            "comparative_metrics": {
                metric_name: {
                    model_type.value: value
                    for model_type, value in metrics.items()
                }
                for metric_name, metrics in report.comparative_metrics.items()
            }
        }

    def _save_csv_report(self, report: TestReport, csv_path: str):
        """保存CSV格式的报告"""
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # 写入标题行
            writer.writerow([
                'audio_path', 'reference_text', 'predicted_text',
                'processing_time', 'confidence_score', 'wer', 'cer'
            ])

            # 写入数据行
            for result in report.results:
                # 这里简化处理，实际应该计算每个样本的WER/CER
                writer.writerow([
                    result.audio_path,
                    result.reference_text,
                    result.predicted_text,
                    result.processing_time,
                    result.confidence_score or '',
                    '',  # WER
                    ''   # CER
                ])

    def _generate_visualization(self, report: ComparativeReport, output_path: str):
        """生成可视化图表"""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'ASR模型对比测试 - {report.dataset_type.value}', fontsize=16)

            # 准备数据
            models = []
            wers = []
            cers = []
            rtfs = []
            processing_times = []

            for model_type, model_report in report.model_reports.items():
                if model_report.metrics:
                    models.append(model_type.value)
                    wers.append(model_report.metrics.wer or 0)
                    cers.append(model_report.metrics.cer or 0)
                    rtfs.append(model_report.metrics.realtime_factor or 0)
                    processing_times.append(model_report.metrics.processing_time_avg or 0)

            if not models:
                plt.close(fig)
                return

            # WER对比
            axes[0, 0].bar(models, wers)
            axes[0, 0].set_title('Word Error Rate (WER)')
            axes[0, 0].set_ylabel('WER (%)')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # CER对比
            axes[0, 1].bar(models, cers)
            axes[0, 1].set_title('Character Error Rate (CER)')
            axes[0, 1].set_ylabel('CER (%)')
            axes[0, 1].tick_params(axis='x', rotation=45)

            # 实时因子对比
            axes[1, 0].bar(models, rtfs)
            axes[1, 0].set_title('Real-time Factor (RTF)')
            axes[1, 0].set_ylabel('RTF')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # 处理时间对比
            axes[1, 1].bar(models, processing_times)
            axes[1, 1].set_title('Average Processing Time')
            axes[1, 1].set_ylabel('Time (seconds)')
            axes[1, 1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"生成可视化图表失败: {e}")

    def generate_summary_report(self, reports: List[TestReport], output_path: str):
        """生成摘要报告"""
        summary = {
            "report_count": len(reports),
            "generated_at": datetime.now().isoformat(),
            "summary": {}
        }

        # 按模型分组统计
        model_summaries = {}
        for report in reports:
            model_type = report.model_type.value
            if model_type not in model_summaries:
                model_summaries[model_type] = {
                    "total_tests": 0,
                    "datasets": [],
                    "average_metrics": {}
                }

            model_summaries[model_type]["total_tests"] += report.test_count
            model_summaries[model_type]["datasets"].append({
                "dataset": report.dataset_type.value,
                "test_count": report.test_count,
                "metrics": {
                    "wer": report.metrics.wer,
                    "cer": report.metrics.cer,
                    "rtf": report.metrics.realtime_factor
                }
            })

        summary["summary"]["models"] = model_summaries

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"摘要报告已保存: {output_path}")

    def export_to_html(self, report: ComparativeReport, output_path: str):
        """导出为HTML格式"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ASR模型对比测试报告 - {report.dataset_type.value}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; }}
                .best {{ background-color: #d4edda; }}
                .worst {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <h1>ASR模型对比测试报告</h1>
            <p>数据集: {report.dataset_type.value}</p>
            <p>测试样本数: {report.test_count}</p>
            <p>生成时间: {report.created_at.strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>模型性能对比</h2>
            <table>
                <tr>
                    <th>模型</th>
                    <th>WER (%)</th>
                    <th>CER (%)</th>
                    <th>SER (%)</th>
                    <th>RTF</th>
                    <th>平均处理时间</th>
                </tr>
        """

        # 添加对比数据
        for model_type, model_report in report.model_reports.items():
            if model_report.metrics:
                html_content += f"""
                <tr>
                    <td>{model_type.value}</td>
                    <td>{model_report.metrics.wer:.2%}</td>
                    <td>{model_report.metrics.cer:.2%}</td>
                    <td>{model_report.metrics.ser:.2%}</td>
                    <td>{model_report.metrics.realtime_factor:.3f}</td>
                    <td>{model_report.metrics.processing_time_avg:.3f}s</td>
                </tr>
                """

        html_content += """
            </table>
        </body>
        </html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML报告已保存: {output_path}")