"""
配置加载器
加载和管理系统配置文件
"""

import json
import os
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config/config.json"
        self.config = {}

    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            self._create_default_config()

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            return self.config
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            return self._get_default_config()

    def _create_default_config(self):
        """创建默认配置文件"""
        default_config = self._get_default_config()
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "models": {
                "kimi_audio": {
                    "enabled": True,
                    "model_path": "Model/KimiAudio",
                    "config_file": "config/models/kimi_audio.json",
                    "device": "cuda",
                    "batch_size": 1,
                    "language": "zh",
                    "additional_params": {}
                },
                "fire_red_asr": {
                    "enabled": True,
                    "model_path": "Model/FireRedASR",
                    "config_file": "config/models/fire_red_asr.json",
                    "device": "cuda",
                    "batch_size": 1,
                    "language": "zh",
                    "additional_params": {}
                },
                "step_audio2": {
                    "enabled": True,
                    "model_path": "Model/StepAudio2",
                    "config_file": "config/models/step_audio2.json",
                    "device": "cuda",
                    "batch_size": 1,
                    "language": "zh",
                    "additional_params": {}
                }
            },
            "datasets": {
                "regression": {
                    "path": "datasets/regression_test",
                    "description": "标准回归测试集"
                },
                "noise": {
                    "path": "datasets/noise_test",
                    "description": "噪音环境测试集"
                },
                "long_audio": {
                    "path": "datasets/long_audio_test",
                    "description": "长音频测试集"
                },
                "accent": {
                    "path": "datasets/accent_test",
                    "description": "方言口音测试集"
                },
                "multilingual": {
                    "path": "datasets/multilingual_test",
                    "description": "多语言测试集"
                }
            },
            "evaluation": {
                "calculate_wer": True,
                "calculate_cer": True,
                "calculate_ser": True,
                "calculate_rtf": True,
                "calculate_confidence": True,
                "language": "zh",
                "normalize_text": True
            },
            "output": {
                "save_detailed_results": True,
                "save_summary": True,
                "output_format": "json",
                "output_dir": "results",
                "create_timestamp_dir": True
            },
            "logging": {
                "level": "INFO",
                "file": "logs/asr_test.log",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config

    def save_config(self, config: Dict[str, Any]):
        """保存配置"""
        self.config = config
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)

        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """获取单个模型配置"""
        return self.config.get("models", {}).get(model_type, {})

    def get_dataset_config(self, dataset_type: str) -> Dict[str, Any]:
        """获取单个数据集配置"""
        return self.config.get("datasets", {}).get(dataset_type, {})

    def get_evaluation_config(self) -> Dict[str, Any]:
        """获取评估配置"""
        return self.config.get("evaluation", {})

    def get_output_config(self) -> Dict[str, Any]:
        """获取输出配置"""
        return self.config.get("output", {})

    def update_config(self, section: str, key: str, value: Any):
        """更新配置"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self.save_config(self.config)