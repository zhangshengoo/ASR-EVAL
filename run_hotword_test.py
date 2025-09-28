#!/usr/bin/env python3
"""
热词测试运行示例
演示如何使用新的结果管理系统
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from datasets.hotword_manager import HotwordDatasetManager, HotwordEvaluator
from models.step_audio2 import StepAudio2Model
from core.config import load_model_config
from core.models import TestResult
from results.test_results_manager import TestResultsManager

def run_hotword_test():
    """运行热词测试示例"""
    logging.basicConfig(level=logging.INFO)

    # 创建结果管理器
    results_manager = TestResultsManager("results")

    # 加载热词测试数据
    print("加载热词测试数据...")
    hotword_manager = HotwordDatasetManager()
    test_items = hotword_manager.load_hotword_dataset("场景1")

    if not test_items:
        print("没有找到测试数据，创建示例数据...")
        hotword_manager.create_sample_hotword_dataset("场景1", 2)
        test_items = hotword_manager.load_hotword_dataset("场景1")

    print(f"加载了 {len(test_items)} 个测试项")

    # 显示测试数据格式
    print("\n测试数据格式：")
    for item in test_items:
        print(f"  音频: {Path(item.audio_path).name}")
        print(f"  参考文本: {item.reference_text}")
        print(f"  热词: {item.metadata.get('hotwords', [])}")
        print(f"  场景: {item.metadata.get('scene', 'unknown')}")
        print()

    # 初始化模型（使用后备实现）
    print("初始化StepAudio2模型...")
    config = load_model_config("step_audio2")
    model = StepAudio2Model(config)

    if not model.load_model():
        print("模型加载失败")
        return

    # 模拟测试结果
    print("生成模拟测试结果...")
    results = []

    for item in test_items:
        # 模拟StepAudio2识别结果
        result = model.transcribe_audio(item.audio_path,
                                      hotwords=item.metadata.get("hotwords", []))

        # 创建TestResult对象（使用正确的字段）
        test_result = TestResult(
            audio_path=item.audio_path,
            model_type=model.model_type,
            reference_text=item.reference_text,
            predicted_text=result["text"],
            processing_time=result["processing_time"],
            confidence_score=result["confidence"]
        )

        results.append(test_result)
        print(f"测试完成: {Path(item.audio_path).name} -> {result['text']}")

    # 评估热词性能
    print("评估热词性能...")
    evaluator = HotwordEvaluator()

    # 创建模拟结果对象
    class MockResult:
        def __init__(self, predicted_text):
            self.predicted_text = predicted_text

    mock_results = [MockResult(r.predicted_text) for r in results]
    hotword_performance = evaluator.evaluate_hotword_performance(mock_results, test_items)

    # 保存结果
    print("保存测试结果...")
    result_dir = results_manager.save_results(
        results=results,
        model_name="StepAudio2",  # 这将创建 stepaudio2 目录
        dataset_name="场景1_热词测试",  # 这将解析为 场景1/热词测试 子目录
        metadata={
            "model_name": "Step-Audio-2-mini",
            "dataset_name": "场景1",
            "test_type": "hotword"
        }
    )

    # 生成报告
    print("生成测试报告...")
    report_path = results_manager.generate_report(results, result_dir)

    # 输出热词分析
    print("\n=== 热词分析结果 ===")
    for hw, stats in hotword_performance["hotword_performance"].items():
        print(f"热词 '{hw}': 准确率 {stats['accuracy']:.2f}")

    print(f"\n=== 结果输出 ===")
    print(f"结果目录: {result_dir}")
    print(f"JSON结果: {result_dir}/results.json")
    print(f"CSV结果: {result_dir}/results.csv")
    print(f"热词分析: {result_dir}/hotword_analysis.json")
    print(f"HTML报告: {report_path}")

    # 显示结果目录结构
    print(f"\n结果目录结构:")
    for item in Path(result_dir).rglob("*"):
        if item.is_file():
            print(f"  {item.relative_to(Path(result_dir))}")

    # 显示模型专用目录结构
    model_result_base = Path("results") / "stepaudio2" / "热词测试" / "场景1"
    if model_result_base.exists():
        print(f"\n模型专用结果目录:")
        print(f"  {model_result_base}")
        for item in model_result_base.rglob("*"):
            if item.is_file():
                print(f"    {item.relative_to(model_result_base)}")

    # 检查 .list 文件内容
    list_file = model_result_base / "场景1.list"
    if list_file.exists():
        print(f"\n.list 文件内容:")
        with open(list_file, 'r', encoding='utf-8') as f:
            for line in f:
                print(f"  {line.strip()}")

    # 清理模型
    model.cleanup()

    print("\n热词测试完成！")

if __name__ == "__main__":
    run_hotword_test()