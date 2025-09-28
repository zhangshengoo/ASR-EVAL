"""
ASR测试框架主类
协调整个测试流程
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.enums import ModelType, TestDatasetType
from core.models import TestReport, ComparativeReport, TestItem, TestResult
from models.factory import ModelFactory, load_and_initialize_models
from datasets.manager import TestDatasetManager
from evaluation.metrics import MetricsCalculator, MetricConfig
from utils.config_loader import ConfigLoader
from utils.reporter import TestReporter


logger = logging.getLogger(__name__)


class TestConfig:
    """测试配置"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config()

        # 模型配置
        self.model_configs = self._parse_model_configs()
        self.dataset_configs = self._parse_dataset_configs()
        self.evaluation_config = self._parse_evaluation_config()

    def _parse_model_configs(self) -> Dict[ModelType, Dict[str, Any]]:
        """解析模型配置"""
        model_configs = {}
        models_config = self.config.get("models", {})

        for model_type_str, config in models_config.items():
            try:
                model_type = ModelType(model_type_str)
                model_configs[model_type] = config
            except ValueError:
                logger.warning(f"未知的模型类型: {model_type_str}")
                continue

        return model_configs

    def _parse_dataset_configs(self) -> Dict[TestDatasetType, Dict[str, Any]]:
        """解析数据集配置"""
        dataset_configs = {}
        datasets_config = self.config.get("datasets", {})

        for dataset_type_str, config in datasets_config.items():
            try:
                dataset_type = TestDatasetType(dataset_type_str)
                dataset_configs[dataset_type] = config
            except ValueError:
                logger.warning(f"未知的数据集类型: {dataset_type_str}")
                continue

        return dataset_configs

    def _parse_evaluation_config(self) -> Dict[str, Any]:
        """解析评估配置"""
        return self.config.get("evaluation", {})


class ASRTestFramework:
    """ASR测试框架主类"""

    def __init__(self, config_path: Optional[str] = None):
        self.test_config = TestConfig(config_path)
        self.model_manager = ModelManager(self.test_config.model_configs)
        self.dataset_manager = TestDatasetManager()
        self.reporter = TestReporter()

        # 已加载的模型
        self.loaded_models: Dict[ModelType, Any] = {}

    def initialize(self) -> bool:
        """初始化框架"""
        try:
            logger.info("正在初始化ASR测试框架...")

            # 初始化模型
            self.loaded_models = self.model_manager.load_models()
            if not self.loaded_models:
                logger.error("没有成功加载任何模型")
                return False

            logger.info(f"成功加载 {len(self.loaded_models)} 个模型")
            logger.info("ASR测试框架初始化完成")
            return True

        except Exception as e:
            logger.error(f"初始化ASR测试框架失败: {e}")
            return False

    def run_single_model_test(self, model_type: ModelType,
                            dataset_type: TestDatasetType,
                            max_samples: Optional[int] = None) -> TestReport:
        """对单个模型运行测试"""
        logger.info(f"开始测试模型 {model_type.value} 在数据集 {dataset_type.value} 上")

        if model_type not in self.loaded_models:
            raise ValueError(f"模型 {model_type.value} 未加载")

        # 加载测试数据
        test_items = self.dataset_manager.load_dataset(dataset_type)
        if not test_items:
            logger.warning(f"数据集 {dataset_type.value} 为空")
            return self._create_empty_report(model_type, dataset_type)

        # 限制样本数量
        if max_samples and max_samples < len(test_items):
            test_items = test_items[:max_samples]
            logger.info(f"限制测试样本数量为: {max_samples}")

        # 运行测试
        model = self.loaded_models[model_type]
        test_results = self._run_model_inference(model, test_items)

        # 计算指标
        metrics_calculator = self._create_metrics_calculator()
        metrics = metrics_calculator.calculate_metrics(test_results)

        # 生成报告
        report = TestReport(
            model_type=model_type,
            dataset_type=dataset_type,
            test_count=len(test_results),
            metrics=metrics,
            results=test_results,
            config={
                "max_samples": max_samples,
                "model_config": self.test_config.model_configs.get(model_type, {}),
                "dataset_config": self.test_config.dataset_configs.get(dataset_type, {})
            }
        )

        logger.info(f"模型 {model_type.value} 测试完成: "
                   f"WER={metrics.wer:.2%}, CER={metrics.cer:.2%}, "
                   f"RTF={metrics.realtime_factor:.3f}")

        return report

    def run_comparative_test(self, dataset_type: TestDatasetType,
                           model_types: Optional[List[ModelType]] = None,
                           max_samples: Optional[int] = None) -> ComparativeReport:
        """运行对比测试"""
        if model_types is None:
            model_types = list(self.loaded_models.keys())

        logger.info(f"开始对比测试: 数据集={dataset_type.value}, 模型={len(model_types)}")

        model_reports = {}
        for model_type in model_types:
            if model_type in self.loaded_models:
                try:
                    report = self.run_single_model_test(model_type, dataset_type, max_samples)
                    model_reports[model_type] = report
                except Exception as e:
                    logger.error(f"测试模型 {model_type.value} 失败: {e}")
                    continue

        if not model_reports:
            raise RuntimeError("没有成功完成任何模型的测试")

        # 生成对比报告
        comparative_report = ComparativeReport(
            dataset_type=dataset_type,
            test_count=model_reports[list(model_reports.keys())[0]].test_count,
            model_reports=model_reports
        )

        # 计算对比指标
        self._calculate_comparative_metrics(comparative_report)

        logger.info("对比测试完成")
        return comparative_report

    def run_batch_tests(self, test_plan: Dict[str, Any]) -> Dict[str, Any]:
        """运行批量测试计划"""
        logger.info("开始执行批量测试计划")

        results = {
            "start_time": datetime.now().isoformat(),
            "test_plan": test_plan,
            "reports": {}
        }

        test_cases = test_plan.get("test_cases", [])

        for i, test_case in enumerate(test_cases):
            logger.info(f"执行测试用例 {i+1}/{len(test_cases)}")

            try:
                model_types = [ModelType(mt) for mt in test_case.get("models", [])]
                dataset_types = [TestDatasetType(dt) for dt in test_case.get("datasets", [])]
                max_samples = test_case.get("max_samples")

                case_results = {}
                for dataset_type in dataset_types:
                    report = self.run_comparative_test(dataset_type, model_types, max_samples)
                    case_results[dataset_type.value] = report

                results["reports"][f"test_case_{i+1}"] = {
                    "config": test_case,
                    "results": case_results,
                    "status": "completed"
                }

            except Exception as e:
                logger.error(f"测试用例 {i+1} 失败: {e}")
                results["reports"][f"test_case_{i+1}"] = {
                    "config": test_case,
                    "error": str(e),
                    "status": "failed"
                }

        results["end_time"] = datetime.now().isoformat()
        logger.info("批量测试计划执行完成")
        return results

    def _run_model_inference(self, model, test_items: List[TestItem]) -> List[TestResult]:
        """运行模型推理"""
        test_results = []
        batch_size = 1  # 可以根据模型支持调整批次大小

        logger.info(f"开始推理: 样本数量={len(test_items)}")

        for i, item in enumerate(test_items):
            try:
                logger.debug(f"推理进度: {i+1}/{len(test_items)}")

                # 单个推理
                result = model.run_inference(item.audio_path, item.reference_text)
                test_results.append(result)

            except Exception as e:
                logger.error(f"推理失败 {item.audio_path}: {e}")
                continue

        logger.info(f"推理完成: 成功={len(test_results)}/{len(test_items)}")
        return test_results

    def _create_metrics_calculator(self) -> MetricsCalculator:
        """创建指标计算器"""
        metric_config = MetricConfig(
            calculate_wer=self.test_config.evaluation_config.get("calculate_wer", True),
            calculate_cer=self.test_config.evaluation_config.get("calculate_cer", True),
            calculate_ser=self.test_config.evaluation_config.get("calculate_ser", True),
            calculate_rtf=self.test_config.evaluation_config.get("calculate_rtf", True),
            calculate_confidence=self.test_config.evaluation_config.get("calculate_confidence", True),
            language=self.test_config.evaluation_config.get("language", "zh"),
            normalize_text=self.test_config.evaluation_config.get("normalize_text", True)
        )
        return MetricsCalculator(metric_config)

    def _create_empty_report(self, model_type: ModelType, dataset_type: TestDatasetType) -> TestReport:
        """创建空报告"""
        return TestReport(
            model_type=model_type,
            dataset_type=dataset_type,
            test_count=0,
            metrics=None,
            results=[],
            config={}
        )

    def _calculate_comparative_metrics(self, report: ComparativeReport):
        """计算对比指标"""
        if not report.model_reports:
            return

        comparative_metrics = {}

        # 收集各模型的指标
        for metric_name in ["wer", "cer", "ser", "processing_time_avg", "realtime_factor"]:
            model_metrics = {}
            for model_type, model_report in report.model_reports.items():
                if model_report.metrics:
                    metric_value = getattr(model_report.metrics, metric_name, None)
                    if metric_value is not None:
                        model_metrics[model_type] = metric_value

            if model_metrics:
                comparative_metrics[metric_name] = model_metrics

        report.comparative_metrics = comparative_metrics

    def save_report(self, report: TestReport, output_path: str):
        """保存测试报告"""
        self.reporter.save_test_report(report, output_path)

    def save_comparative_report(self, report: ComparativeReport, output_path: str):
        """保存对比测试报告"""
        self.reporter.save_comparative_report(report, output_path)

    def cleanup(self):
        """清理资源"""
        logger.info("正在清理资源...")
        for model in self.loaded_models.values():
            try:
                model.cleanup()
            except Exception as e:
                logger.error(f"清理模型失败: {e}")


class ModelManager:
    """模型管理器"""

    def __init__(self, model_configs: Dict[ModelType, Dict[str, Any]]):
        self.model_configs = model_configs

    def load_models(self) -> Dict[ModelType, Any]:
        """加载所有配置的模型"""
        loaded_models = {}

        for model_type, config in self.model_configs.items():
            if not config.get("enabled", True):
                logger.info(f"跳过禁用的模型: {model_type.value}")
                continue

            try:
                logger.info(f"正在加载模型: {model_type.value}")
                model = self._load_single_model(model_type, config)
                if model:
                    loaded_models[model_type] = model
                    logger.info(f"模型加载成功: {model_type.value}")
                else:
                    logger.error(f"模型加载失败: {model_type.value}")

            except Exception as e:
                logger.error(f"加载模型 {model_type.value} 时出错: {e}")
                continue

        logger.info(f"成功加载 {len(loaded_models)}/{len(self.model_configs)} 个模型")
        return loaded_models

    def _load_single_model(self, model_type: ModelType, config: Dict[str, Any]) -> Optional[Any]:
        """加载单个模型"""
        try:
            from core.models import ModelConfig
            from models.factory import ModelFactory

            model_config = ModelConfig(
                model_type=model_type,
                model_path=config.get("model_path", f"Model/{model_type.value}"),
                config_file=config.get("config_file"),
                enabled=True,
                device=config.get("device", "cuda"),
                batch_size=config.get("batch_size", 1),
                language=config.get("language", "zh"),
                additional_params=config.get("additional_params", {})
            )

            model = ModelFactory.create_model(model_config)
            success = model.load_model()

            return model if success else None

        except Exception as e:
            logger.error(f"创建或加载模型失败 {model_type.value}: {e}")
            return None


# 使用示例
if __name__ == "__main__":
    # 初始化测试框架
    framework = ASRTestFramework("config/test_config.json")

    # 初始化
    if framework.initialize():
        # 运行对比测试
        report = framework.run_comparative_test(
            dataset_type=TestDatasetType.REGRESSION,
            model_types=[ModelType.KIMI_AUDIO, ModelType.FIRE_RED_ASR, ModelType.STEP_AUDIO2],
            max_samples=10
        )

        # 保存报告
        framework.save_comparative_report(report, "results/comparative_report.json")

        # 清理资源
        framework.cleanup()