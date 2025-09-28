"""
配置管理模块
负责加载和管理模型和测试配置
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any

def load_model_config(model_name: str) -> Dict[str, Any]:
    """加载模型配置"""
    config_path = Path("config/models") / f"{model_name}.json"

    if not config_path.exists():
        logging.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return get_default_model_config(model_name)

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"加载配置失败: {e}")
        return get_default_model_config(model_name)

def get_default_model_config(model_name: str) -> Dict[str, Any]:
    """获取默认模型配置"""
    return {
        "model_name": model_name,
        "model_path": f"Model/{model_name}",
        "device": "cuda",
        "batch_size": 1,
        "language": "zh",
        "max_length": 512,
        "sample_rate": 16000,
        "hotwords": [],
        "hotword_config": {
            "enabled": True,
            "confidence_threshold": 0.8,
            "max_hotword_length": 20,
            "case_sensitive": False
        },
        "audio_config": {
            "supported_formats": ["wav", "mp3", "flac", "m4a", "ogg"],
            "min_duration": 0.5,
            "max_duration": 300
        }
    }