#!/usr/bin/env python3
"""
可配置热词处理脚本
支持自定义热词库大小和配置
"""

import os
import json
import random
from pathlib import Path


class ConfigurableHotwordProcessor:
    def __init__(self, base_path: str = "/Users/zhangsheng/code/ASR-Eval/datasets/热词测试/场景1"):
        self.base_path = Path(base_path)
        self.hotwords = []
        self.samples = []

    def load_data(self):
        with open(self.base_path / "hotword.txt", 'r') as f:
            self.hotwords = [line.strip() for line in f if line.strip()]

        with open(self.base_path / "场景1.list", 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    parts = line.split('|')
                    if len(parts) >= 2:
                        self.samples.append({
                            'filename': Path(parts[0].strip()).stem,
                            'target_text': parts[1].strip()
                        })

    def generate_configs(self, sample, config_sizes):
        target = [hw for hw in self.hotwords if hw.lower() in sample['target_text'].lower()]
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

        configs['full'] = {'hotwords': self.hotwords, 'size': len(self.hotwords)}
        return configs

    def process(self, config_sizes=None):
        if config_sizes is None:
            config_sizes = [3, 5, 7, 10]

        self.load_data()
        results = []

        for sample in self.samples:
            target = [hw for hw in self.hotwords if hw.lower() in sample['target_text'].lower()]
            processed = {
                'filename': sample['filename'],
                'target_text': sample['target_text'],
                'target_hotwords': target,
                'configs': self.generate_configs(sample, config_sizes)
            }
            results.append(processed)

        output = {
            'config_sizes': config_sizes,
            'hotwords': self.hotwords,
            'samples': results
        }

        output_file = self.base_path / "hotword_dataset.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        return str(output_file)


if __name__ == "__main__":
    processor = ConfigurableHotwordProcessor()
    result = processor.process([3, 5, 10])
    print(result)