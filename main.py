#!/usr/bin/env python3
"""
ASR模型测试系统主入口
"""

import argparse
import logging
import sys
from pathlib import Path

from core.enums import ModelType, TestDatasetType
from testing.framework import ASRTestFramework
from utils.config_loader import ConfigLoader


def setup_logging(config: dict):
    """设置日志"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper())
    log_file = log_config.get("file", "logs/asr_test.log")
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 创建日志目录
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # 配置日志
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ASR模型测试系统")
    parser.add_argument("--config", "-c", default="config/config.json", help="配置路径")
    parser.add_argument("--model", "-m", action="append", help="模型类型")
    parser.add_argument("--dataset", "-d", action="append", help="数据集类型")
    parser.add_argument("--max-samples", "-n", type=int, help="最大样本数")
    parser.add_argument("--output", "-o", help="输出目录")
    parser.add_argument("--compare", action="store_true", help="对比测试")
    parser.add_argument("--batch", help="批量测试文件")
    parser.add_argument("--create-sample-data", action="store_true", help="创建示例数据")
    parser.add_argument("--hotword", help="热词测试场景", choices=["场景1", "场景2", "场景3"])

    args = parser.parse_args()

    # 加载配置
    config_loader = ConfigLoader(args.config)
    config = config_loader.load_config()
    setup_logging(config)

    logger = logging.getLogger(__name__)
    logger.info("ASR模型测试系统启动")

    try:
        # 初始化测试框架
        framework = ASRTestFramework(args.config)

        if not framework.initialize():
            logger.error("测试框架初始化失败")
            return 1

        # 创建示例数据集
        if args.create_sample_data:
            logger.info("创建示例数据集...")
            for dataset_type in TestDatasetType:
                framework.dataset_manager.create_sample_dataset(dataset_type, 5)
            logger.info("示例数据集创建完成")
            return 0

        # 批量测试
        if args.batch:
            logger.info(f"执行批量测试计划: {args.batch}")
            with open(args.batch, 'r', encoding='utf-8') as f:
                test_plan = json.load(f)
            results = framework.run_batch_tests(test_plan)

            output_path = args.output or "results/batch_test_results.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"批量测试结果已保存: {output_path}")
            return 0

        # 确定测试的模型和数据集
        model_types = []
        if args.model:
            for model_str in args.model:
                try:
                    model_types.append(ModelType(model_str))
                except ValueError:
                    logger.warning(f"未知的模型类型: {model_str}")
        else:
            model_types = list(framework.loaded_models.keys())

        dataset_types = []
        if args.dataset:
            for dataset_str in args.dataset:
                try:
                    dataset_types.append(TestDatasetType(dataset_str))
                except ValueError:
                    logger.warning(f"未知的数据集类型: {dataset_str}")
        else:
            dataset_types = [TestDatasetType.REGRESSION]

        if not model_types:
            logger.error("没有可测试的模型")
            return 1

        if not dataset_types:
            logger.error("没有可测试的数据集")
            return 1

        # 运行测试
        if args.compare:
            # 对比测试
            for dataset_type in dataset_types:
                logger.info(f"运行对比测试: 数据集={dataset_type.value}, 模型={len(model_types)}")
                report = framework.run_comparative_test(
                    dataset_type=dataset_type,
                    model_types=model_types,
                    max_samples=args.max_samples
                )

                # 保存报告
                output_dir = args.output or "results"
                Path(output_dir).mkdir(parents=True, exist_ok=True)
                framework.save_comparative_report(report, f"{output_dir}/comparative_report.json")

                # 生成HTML报告
                reporter = framework.reporter
                html_path = f"{output_dir}/comparative_report.html"
                reporter.export_to_html(report, html_path)

                logger.info(f"对比测试报告已保存到: {output_dir}")
        else:
            # 单个模型测试
            for model_type in model_types:
                for dataset_type in dataset_types:
                    logger.info(f"测试模型 {model_type.value} 在数据集 {dataset_type.value} 上")
                    report = framework.run_single_model_test(
                        model_type=model_type,
                        dataset_type=dataset_type,
                        max_samples=args.max_samples
                    )

                    # 保存报告
                    output_dir = args.output or "results"
                    Path(output_dir).mkdir(parents=True, exist_ok=True)
                    framework.save_report(report, f"{output_dir}/test_report_{model_type.value}.json")

        logger.info("测试完成")
        framework.cleanup()
        return 0

    except KeyboardInterrupt:
        logger.info("用户中断测试")
        return 130
    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())