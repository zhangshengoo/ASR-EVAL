#!/usr/bin/env python3
"""
通用热词处理脚本
支持任意目录结构的热词测试数据集处理
"""

import os
import json
import random
from pathlib import Path


class HotwordProcessor:
    """通用热词处理器"""

    def __init__(self, hotwords: list = None):
        self.hotwords = hotwords or []
        self.samples = []

    def load_hotwords(self, hotword_file: str) -> list:
        """加载热词文件"""
        with open(hotword_file, 'r', encoding='utf-8') as f:
            self.hotwords = [line.strip() for line in f if line.strip()]
        return self.hotwords

    def load_dataset(self, dataset_file: str) -> list:
        """加载数据集文件"""
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.split('|')
                    if len(parts) >= 2:
                        self.samples.append({
                            'filename': Path(parts[0].strip()).stem,
                            'target_text': parts[1].strip()
                        })
        return self.samples

    def generate_configs(self, sample: dict, config_sizes: list) -> dict:
        """生成热词配置"""
        target = [hw for hw in self.hotwords
                 if hw.lower() in sample['target_text'].lower()]
        others = [hw for hw in self.hotwords if hw not in target]

        configs = {}

        for size in config_sizes:
            if size <= len(self.hotwords):
                if size <= len(target):
                    hotwords = target[:size]
                else:
                    needed = max(0, size - len(target))
                    non_target = random.sample(others, min(needed, len(others)))
                    hotwords = target + non_target

                configs[f'lib{size}'] = {
                    'hotwords': hotwords[:size],
                    'size': size
                }

        return configs

    def process_dataset(self, hotword_file: str, dataset_file: str,
                       config_sizes: list = None, output_file: str = None) -> str:
        """处理数据集"""
        if config_sizes is None:
            config_sizes = [3, 5, 7, 10]

        # 加载数据
        self.load_hotwords(hotword_file)
        self.load_dataset(dataset_file)

        # 处理样本
        results = []
        for sample in self.samples:
            target = [hw for hw in self.hotwords
                     if hw.lower() in sample['target_text'].lower()]

            processed = {
                'filename': sample['filename'],
                'target_text': sample['target_text'],
                'target_hotwords': target,
                'configs': self.generate_configs(sample, config_sizes)
            }
            results.append(processed)

        # 生成输出
        output = {
            'config_sizes': config_sizes,
            'hotwords': self.hotwords,
            'total_samples': len(results),
            'samples': results
        }

        # 保存结果
        if output_file:
            output_path = Path(output_file)
        else:
            output_path = Path(dataset_file).parent / "hotword_dataset.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return str(output_path)


def main():
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="通用热词处理工具")
    parser.add_argument("hotword_file", help="热词文件路径")
    parser.add_argument("dataset_file", help="数据集文件路径(.list)")
    parser.add_argument("--sizes", nargs="+", type=int, default=[3, 5, 10],
                       help="热词库大小配置 (默认: 3 5 10)")
    parser.add_argument("--output", help="输出文件路径")

    args = parser.parse_args()

    processor = HotwordProcessor()
    result = processor.process_dataset(
        args.hotword_file,
        args.dataset_file,
        args.sizes,
        args.output
    )

    print(result)


if __name__ == "__main__":
    main()