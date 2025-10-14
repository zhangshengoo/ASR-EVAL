"""
热词自动化测试系统
整合不同ASR模型、文本处理和评估工具，实现热词召回率和准确率的自动化测试
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
# import multiprocessing  # 注释掉，当前实现中未直接使用

from models.factory import ModelFactory
from models.base import BaseASRModel
from core.enums import ModelType
from core.models import ModelConfig
from evaluation.text_normalizer import TextNormalizer
from evaluation.text_alignment import TextAligner
from evaluation.hotword_metrics import HotwordMetricsCalculator


@dataclass
class HotwordTestSample:
    """热词测试样本"""
    filename: str
    target_text: str
    target_hotwords: List[str]
    configs: Dict[str, Dict[str, Any]]  # 不同热词库配置
    audio_path: Optional[str] = None


@dataclass
class HotwordTestResult:
    """热词测试结果"""
    sample_id: str
    model_name: str
    config_name: str
    hotword_library: List[str]
    target_text: str
    predicted_text: str
    normalized_target: str
    normalized_predicted: str
    recall: float
    precision: float
    f1_score: float
    processing_time: float
    alignment_details: Dict[str, Any]
    hotword_matches: Dict[str, Dict[str, int]]


class HotwordAutomationSystem:
    """热词自动化测试系统"""

    def __init__(self, config_path: str = "config/hotword_test_config.json"):
        """
        初始化热词自动化测试系统

        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.model_factory = ModelFactory()
        self.text_normalizer = TextNormalizer()
        self.text_aligner = TextAligner()
        self.hotword_metrics = HotwordMetricsCalculator()
        self.results_cache = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件并使其设置生效"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
        except FileNotFoundError:
            print(f"配置文件 {config_path} 不存在，使用默认配置")
            loaded_config = self._create_default_config()

        # 处理配置文件，使其设置生效
        return self._process_config(loaded_config)

    def _create_default_config(self) -> Dict[str, Any]:
        """创建默认配置"""
        return {
            "models": {
                "step_audio2": {
                    "model_path": "Model/Stepaudio2",
                    "device": "cuda",
                    "parallel_enabled": True,
                    "num_processes": 4,
                    "available_gpus": [0, 1],
                    "batch_size": 2
                }
            },
            "datasets": ["/Users/zhangsheng/code/ASR-Eval/datasets/热词测试/场景1"],
            "test_settings": {
                "parallel_enabled": True,
                "num_processes": 4,
                "available_gpus": [0, 1],
                "batch_size": 2
            },
            "text_normalization": {
                "enabled": True,
                "language": "auto"
            },
            "output": {
                "output_path": "results/hotword_tests"
            }
        }

    def _process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """处理配置，提取关键设置并使其生效"""
        # 提取测试设置
        test_settings = config.get("test_settings", {})
        self.parallel_enabled = test_settings.get("parallel_enabled", True)
        self.num_processes = test_settings.get("num_processes", 4)
        self.available_gpus = test_settings.get("available_gpus", [0])
        self.batch_size = test_settings.get("batch_size", 2)

        # 提取数据集配置
        datasets_config = config.get("datasets", [])
        self.available_datasets = datasets_config if isinstance(datasets_config, list) else []
        self.default_dataset = self.available_datasets[0] if self.available_datasets else "/Users/zhangsheng/code/ASR-Eval/datasets/热词测试/场景1"

        # 提取输出配置
        output_config = config.get("output", {})
        self.output_path = output_config.get("output_path", "results/hotword_tests")

        # 提取热词库配置选择
        self.enabled_configs = config.get("enabled_configs", ["lib3", "lib5", "lib10"])
        self.hotword_configurations = config.get("hotword_configurations", {})

        # 打印配置信息
        print(f"配置加载完成:")
        print(f"  测试设置: parallel={self.parallel_enabled}, processes={self.num_processes}, GPUs={self.available_gpus}, batch={self.batch_size}")
        print(f"  可用数据集: {len(self.available_datasets)} 个")
        print(f"  默认数据集: {self.default_dataset}")
        print(f"  输出路径: {self.output_path}")
        print(f"  启用热词库配置: {self.enabled_configs}")

        return config

    def load_hotword_dataset(self, dataset_path: str) -> List[HotwordTestSample]:
        """
        加载热词测试数据集

        Args:
            dataset_path: 数据集路径

        Returns:
            热词测试样本列表
        """
        dataset_file = Path(dataset_path) / "hotword_dataset.json"
        if not dataset_file.exists():
            raise FileNotFoundError(f"热词数据集文件不存在: {dataset_file}")

        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        samples = []
        for sample_data in data.get("samples", []):
            # 构建音频文件完整路径
            audio_path = str(Path(dataset_path) / f"{sample_data['filename']}.wav")
            if not os.path.exists(audio_path):
                print(f"警告: 音频文件不存在: {audio_path}")
                continue

            sample = HotwordTestSample(
                filename=sample_data["filename"],
                target_text=sample_data["target_text"],
                target_hotwords=sample_data["target_hotwords"],
                configs=sample_data.get("configs", {}),
                audio_path=audio_path
            )
            samples.append(sample)

        return samples

    def initialize_models(self, model_configs: Dict[str, Dict[str, Any]]) -> Dict[str, BaseASRModel]:
        """
        初始化多个ASR模型

        Args:
            model_configs: 模型配置字典

        Returns:
            初始化好的模型字典
        """
        models = {}

        for model_name, config in model_configs.items():
            print(f"正在初始化模型: {model_name}")
            print(f"模型配置: {config}")

            try:
                model_type = ModelType(model_name)
            except ValueError:
                print(f"错误: 不支持的模型类型 {model_name}")
                continue

            # 创建ModelConfig对象
            model_config = ModelConfig(
                model_type=model_type,
                model_path=config.get("model_path", f"Model/{model_name.title().replace('_', '')}"),
                config_file=config.get("config_file"),
                device=config.get("device", "cuda"),
                batch_size=config.get("batch_size", 1),
                language=config.get("language", "zh"),
                additional_params={
                    "parallel_enabled": config.get("parallel_enabled", False),
                    "num_processes": config.get("num_processes", 4),
                    "available_gpus": config.get("available_gpus", [0]),
                    **{k: v for k, v in config.items() if k not in ["model_path", "config_file", "device", "batch_size", "language", "parallel_enabled", "num_processes", "available_gpus"]}
                }
            )

            print(f"创建ModelConfig: {model_config}")
            model = self.model_factory.create_model(model_config)

            if model and model.load_model():
                models[model_name] = model
                print(f"模型 {model_name} 加载成功")
            else:
                print(f"模型 {model_name} 加载失败")

        return models

    def process_with_text_normalization(self, text: str, language: str = "auto") -> str:
        """
        使用文本规范化处理

        Args:
            text: 原始文本
            language: 语言代码

        Returns:
            规范化后的文本
        """
        if not self.config.get("text_normalization", {}).get("enabled", True):
            return text

        if language != "auto":
            self.text_normalizer.set_language(language)
        return self.text_normalizer.normalize(text)

    def perform_text_alignment(self, target_text: str, predicted_text: str) -> Dict[str, Any]:
        """
        执行文本对齐

        Args:
            target_text: 目标文本
            predicted_text: 预测文本

        Returns:
            对齐结果详情
        """
        alignment_result = self.text_aligner.generate_diff_report(target_text, predicted_text)
        return {
            "alignment": alignment_result.get("alignment", []),
            "differences": alignment_result.get("differences", {}),
            "similarity_score": alignment_result.get("similarity_score", 0.0)
        }

    def test_single_sample(self, model: BaseASRModel, sample: HotwordTestSample,
                          config_name: str, hotword_library: List[str]) -> HotwordTestResult:
        """
        测试单个样本

        Args:
            model: ASR模型
            sample: 测试样本
            config_name: 配置名称
            hotword_library: 热词库

        Returns:
            测试结果
        """
        start_time = time.time()

        # 设置热词库（如果模型支持）
        if hasattr(model, 'set_hotwords'):
            model.set_hotwords(hotword_library)

        # 执行ASR推理
        result = model.transcribe_audio(sample.audio_path)
        predicted_text = result.get("text", "")
        processing_time = result.get("processing_time", time.time() - start_time)

        # 设置热词计算器
        self.hotword_metrics.load_hotwords(hotword_library)

        # 计算热词指标 - 使用原始文本，calculate_metrics内部会进行规范化处理
        # 同时获取对齐详情以避免重复计算
        metrics = self.hotword_metrics.calculate_metrics(
            sample.target_text, predicted_text, include_alignment=True
        )

        # 从metrics中提取对齐详情，避免重复对齐计算
        alignment_details = metrics.get('alignment_details', {})

        # 直接使用metrics中的结果
        recall = metrics.get("recall", 0.0)
        precision = metrics.get("precision", 0.0)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # 简化热词匹配详情，直接使用计算结果
        hotword_matches = {
            "overall": {
                "recall": recall,
                "precision": precision,
                "f1_score": f1_score
            }
        }

        return HotwordTestResult(
            sample_id=sample.filename,
            model_name=model.model_name,
            config_name=config_name,
            hotword_library=hotword_library,
            target_text=sample.target_text,
            predicted_text=predicted_text,
            normalized_target=alignment_details.get('reference_text', sample.target_text),
            normalized_predicted=alignment_details.get('hypothesis_text', predicted_text),
            recall=recall,
            precision=precision,
            f1_score=f1_score,
            processing_time=processing_time,
            alignment_details=alignment_details,
            hotword_matches=hotword_matches
        )


    def run_automated_tests(self, models: Dict[str, BaseASRModel],
                          dataset_path: str) -> Dict[str, List[HotwordTestResult]]:
        """
        运行自动化测试

        Args:
            models: 模型字典
            dataset_path: 数据集路径

        Returns:
            测试结果字典
        """
        print("开始热词自动化测试...")

        # 加载数据集
        samples = self.load_hotword_dataset(dataset_path)
        print(f"加载了 {len(samples)} 个测试样本")

        all_results = {}

        # 遍历每个模型
        for model_name, model in models.items():
            print(f"\n测试模型: {model_name}")
            model_results = []

            # 遍历每个样本
            for sample in samples:
                # 只测试enabled_configs中指定的配置
                for config_name, config_data in sample.configs.items():
                    # 检查该配置是否在启用列表中
                    if config_name not in self.enabled_configs:
                        continue

                    hotword_library = config_data.get("hotwords", [])

                    print(f"\n  样本: {sample.filename}, 配置: {config_name}, 热词数: {len(hotword_library)}")

                    # 执行测试
                    result = self.test_single_sample(
                        model, sample, config_name, hotword_library
                    )

                    # 打印详细的样本输出信息
                    print(f"    目标文本: {sample.target_text}")
                    print(f"    模型输出: {result.predicted_text}")
                    print(f"    规范化目标: {result.normalized_target}")
                    print(f"    规范化输出: {result.normalized_predicted}")
                    print(f"    召回率: {result.recall:.3f}, 精确率: {result.precision:.3f}, F1: {result.f1_score:.3f}")
                    print(f"    处理时间: {result.processing_time:.3f}秒")
                    print(f"    {'-' * 60}")  # 分隔线

                    model_results.append(result)

            all_results[model_name] = model_results

        return all_results

    def run_parallel_tests(self, models: Dict[str, BaseASRModel],
                          dataset_path: str) -> Dict[str, List[HotwordTestResult]]:
        """
        真正的并行化测试 - 支持每个样本使用不同的热词库

        Args:
            models: 模型字典
            dataset_path: 数据集路径

        Returns:
            测试结果字典
        """
        print("开始真正的并行化热词测试（支持每个样本不同热词库）...")

        samples = self.load_hotword_dataset(dataset_path)
        print(f"加载了 {len(samples)} 个测试样本")

        all_results = {}

        # 为每个模型启用并行处理
        for model_name, model in models.items():
            print(f"\n并行测试模型: {model_name}")
            model_results = []

            # 配置模型并行参数
            if hasattr(model, 'parallel_enabled'):
                model.parallel_enabled = self.parallel_enabled
                model.num_processes = self.num_processes
                model.available_gpus = self.available_gpus
                model.parallel_batch_size = self.batch_size

            # 按热词库配置分组处理，每个配置单独批处理
            for config_name in self.enabled_configs:
                print(f"\n  处理热词库配置: {config_name}")

                # 收集该配置下的所有样本
                batch_items = []
                for sample in samples:
                    if config_name in sample.configs:
                        config_data = sample.configs[config_name]
                        hotword_library = config_data.get("hotwords", [])

                        # 创建批处理项 - 包含该样本的热词库信息
                        batch_item = {
                            "audio_path": sample.audio_path,
                            "reference_text": sample.target_text,
                            "sample_id": sample.filename,
                            "config_name": config_name,
                            "hotwords": hotword_library,  # 每个样本的热词库
                            "target_hotwords": sample.target_hotwords
                        }
                        batch_items.append(batch_item)

                if not batch_items:
                    print(f"    配置 {config_name} 没有样本，跳过")
                    continue

                print(f"    批处理 {len(batch_items)} 个样本")

                # 执行批量推理 - 现在支持每个样本不同的热词库
                batch_results = model.batch_inference(batch_items)

                # 处理批量推理结果
                for i, batch_result in enumerate(batch_results):
                    batch_item = batch_items[i]

                    # 计算热词指标（使用对应样本的热词库）
                    self.hotword_metrics.load_hotwords(batch_item["hotwords"])
                    metrics = self.hotword_metrics.calculate_metrics(
                        batch_item["reference_text"],
                        batch_result.predicted_text,
                        include_alignment=True
                    )

                    alignment_details = metrics.get('alignment_details', {})
                    recall = metrics.get("recall", 0.0)
                    precision = metrics.get("precision", 0.0)
                    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

                    # 创建热词测试结果
                    hotword_result = HotwordTestResult(
                        sample_id=batch_item["sample_id"],
                        model_name=model_name,
                        config_name=config_name,
                        hotword_library=batch_item["hotwords"],  # 使用样本特定的热词库
                        target_text=batch_item["reference_text"],
                        predicted_text=batch_result.predicted_text,
                        normalized_target=alignment_details.get('reference_text', batch_item["reference_text"]),
                        normalized_predicted=alignment_details.get('hypothesis_text', batch_result.predicted_text),
                        recall=recall,
                        precision=precision,
                        f1_score=f1_score,
                        processing_time=batch_result.processing_time,
                        alignment_details=alignment_details,
                        hotword_matches={"overall": {"recall": recall, "precision": precision, "f1_score": f1_score}}
                    )

                    model_results.append(hotword_result)

            all_results[model_name] = model_results
            print(f"模型 {model_name} 并行测试完成，共 {len(model_results)} 个结果")

        return all_results

    def generate_test_report(self, results: Dict[str, List[HotwordTestResult]]) -> Dict[str, Any]:
        """
        生成测试报告

        Args:
            results: 测试结果

        Returns:
            测试报告
        """
        report = {
            "summary": {},
            "model_comparison": {},
            "detailed_results": {},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # 汇总统计
        total_samples = 0
        total_tests = 0

        for model_name, model_results in results.items():
            total_tests += len(model_results)
            if model_results:
                total_samples += len(set(r.sample_id for r in model_results))

                # 模型统计
                avg_recall = sum(r.recall for r in model_results) / len(model_results)
                avg_precision = sum(r.precision for r in model_results) / len(model_results)
                avg_f1 = sum(r.f1_score for r in model_results) / len(model_results)
                avg_time = sum(r.processing_time for r in model_results) / len(model_results)

                report["model_comparison"][model_name] = {
                    "avg_recall": avg_recall,
                    "avg_precision": avg_precision,
                    "avg_f1_score": avg_f1,
                    "avg_processing_time": avg_time,
                    "total_tests": len(model_results)
                }

                # 详细结果
                report["detailed_results"][model_name] = [
                    {
                        "sample_id": r.sample_id,
                        "config_name": r.config_name,
                        "hotword_library": r.hotword_library,
                        "recall": r.recall,
                        "precision": r.precision,
                        "f1_score": r.f1_score,
                        "processing_time": r.processing_time,
                        "target_text": r.target_text,
                        "predicted_text": r.predicted_text,
                        "hotword_matches": r.hotword_matches
                    }
                    for r in model_results
                ]

        report["summary"] = {
            "total_models": len(results),
            "total_samples": total_samples,
            "total_tests": total_tests,
            "test_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        return report

    def save_results(self, report: Dict[str, Any], output_path: str = None):
        """
        保存测试结果

        Args:
            report: 测试报告
            output_path: 输出路径（如为None则使用配置文件中的路径）
        """
        # 使用配置文件中的输出路径或指定路径
        save_path = output_path or self.output_path
        os.makedirs(save_path, exist_ok=True)

        # 保存详细结果
        results_file = os.path.join(save_path, "hotword_test_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # 保存CSV格式（便于分析）
        csv_file = os.path.join(save_path, "hotword_test_results.csv")
        self._save_csv_results(report, csv_file)

        print(f"结果已保存到: {save_path}")

    def _save_csv_results(self, report: Dict[str, Any], csv_file: str):
        """保存CSV格式的结果"""
        import csv

        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "model_name", "sample_id", "config_name", "recall",
                "precision", "f1_score", "processing_time", "target_text", "predicted_text"
            ])

            for model_name, results in report["detailed_results"].items():
                for result in results:
                    writer.writerow([
                        model_name,
                        result["sample_id"],
                        result["config_name"],
                        result["recall"],
                        result["precision"],
                        result["f1_score"],
                        result["processing_time"],
                        result["target_text"],
                        result["predicted_text"]
                    ])


def main():
    """主函数 - 热词自动化测试"""
    import argparse

    parser = argparse.ArgumentParser(description="热词自动化测试系统")
    parser.add_argument("--config", default="config/hotword_test_config.json",
                       help="配置文件路径")
    parser.add_argument("--dataset",
                       help="数据集路径（如不指定则使用配置文件中的默认数据集）")
    parser.add_argument("--models", nargs="+",
                       default=["step_audio2", "kimi_audio", "fire_red_asr"],
                       help="要测试的模型列表")
    parser.add_argument("--parallel", action="store_true",
                       help="启用并行处理（覆盖配置文件设置）")
    parser.add_argument("--output",
                       help="输出路径（如不指定则使用配置文件中的默认路径）")

    args = parser.parse_args()

    # 创建测试系统
    test_system = HotwordAutomationSystem(args.config)

    # 准备模型配置 - 从配置文件读取模型设置
    model_configs = {}
    config_models = test_system.config.get("models", {})

    print(f"从配置文件加载模型设置，可用模型: {list(config_models.keys())}")

    for model_name in args.models:
        if model_name in config_models:
            # 使用配置文件中的模型设置
            model_configs[model_name] = config_models[model_name].copy()
            print(f"使用配置文件设置模型 {model_name}: {model_configs[model_name]}")
            # 命令行参数可以覆盖配置文件设置
            if args.parallel:
                model_configs[model_name]["parallel_enabled"] = args.parallel
                print(f"命令行参数覆盖并行设置: parallel_enabled = {args.parallel}")
        else:
            # 如果配置文件中没有该模型，使用默认配置
            print(f"警告: 配置文件中没有找到模型 {model_name} 的设置，使用默认配置")
            model_configs[model_name] = {
                "model_path": f"Model/{model_name.title().replace('_', '')}",
                "device": "cuda",
                "parallel_enabled": args.parallel,
                "num_processes": 4,
                "available_gpus": [0, 1]
            }
            print(f"使用默认配置模型 {model_name}: {model_configs[model_name]}")

    # 初始化模型
    print("正在初始化模型...")
    models = test_system.initialize_models(model_configs)

    if not models:
        print("没有成功加载的模型，测试终止")
        return

    # 使用配置文件中的设置或命令行参数
    dataset_path = args.dataset or test_system.default_dataset
    output_path = args.output or test_system.output_path
    use_parallel = args.parallel if args.parallel is not None else test_system.parallel_enabled

    print(f"使用数据集: {dataset_path}")
    print(f"输出路径: {output_path}")
    print(f"并行处理: {use_parallel}")

    # 运行测试
    if use_parallel:
        results = test_system.run_parallel_tests(models, dataset_path)
    else:
        results = test_system.run_automated_tests(models, dataset_path)

    # 生成报告
    report = test_system.generate_test_report(results)

    # 保存结果
    test_system.save_results(report, output_path)

    # 打印摘要
    print("\n=== 测试完成 ===")
    print(f"测试模型: {list(results.keys())}")
    print(f"总测试数: {report['summary']['total_tests']}")
    print(f"平均召回率: {sum(m['avg_recall'] for m in report['model_comparison'].values()) / len(report['model_comparison']):.3f}")
    print(f"平均精确率: {sum(m['avg_precision'] for m in report['model_comparison'].values()) / len(report['model_comparison']):.3f}")


if __name__ == "__main__":
    main()