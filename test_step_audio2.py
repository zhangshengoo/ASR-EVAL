#!/usr/bin/env python3
"""
StepAudio2 测试脚本
演示如何使用StepAudio2模型进行热词测试
"""

import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from datasets.hotword_manager import HotwordDatasetManager, HotwordEvaluator
from models.step_audio2 import StepAudio2Model
from core.config import load_model_config

def test_step_audio2():
    """测试StepAudio2模型"""
    logging.basicConfig(level=logging.INFO)

    # 创建示例热词数据集
    print("创建示例热词数据集...")
    hotword_manager = HotwordDatasetManager()
    hotword_manager.create_sample_hotword_dataset("测试场景", 5)

    # 验证数据集
    print("验证数据集...")
    validation = hotword_manager.validate_dataset("测试场景")
    print("验证结果:", validation)

    # 加载数据集
    print("加载热词测试数据...")
    test_items = hotword_manager.load_hotword_dataset("测试场景")
    print(f"加载了 {len(test_items)} 个测试项")

    # 获取统计信息
    stats = hotword_manager.get_hotword_statistics("测试场景")
    print("热词统计:", stats)

    # 初始化模型
    print("初始化StepAudio2模型...")
    config = load_model_config("step_audio2")
    model = StepAudio2Model(config)

    # 模拟测试结果（真实测试需要实际音频文件）
    print("模拟热词测试结果...")
    evaluator = HotwordEvaluator()

    # 创建模拟结果
    class MockResult:
        def __init__(self, predicted_text):
            self.predicted_text = predicted_text

    mock_results = [
        MockResult("AI帮助很大"),
        MockResult("使用LLM技术"),
        MockResult("语音识别准确"),
        MockResult("热词测试场景"),
        MockResult("模型评估结果")
    ]

    # 评估热词性能
    performance = evaluator.evaluate_hotword_performance(mock_results, test_items)
    print("热词性能评估:")
    print(f"总热词数: {performance['overall_summary']['total_hotwords']}")
    print(f"平均准确率: {performance['overall_summary']['average_accuracy']:.2f}")

    for hw, stats in performance['hotword_performance'].items():
        print(f"热词 '{hw}': 准确率 {stats['accuracy']:.2f}")

    print("\nStepAudio2 测试系统已准备就绪!")
    print("使用方法:")
    print("1. 准备真实音频文件到 datasets/热词测试/测试场景/")
    print("2. 确保StepAudio2模型已下载到 Model/StepAudio2/Step-Audio-2-mini/")
    print("3. 运行: python test_step_audio2.py")

if __name__ == "__main__":
    test_step_audio2()