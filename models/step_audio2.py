"""
StepAudio2模型实现 - 使用独立的API包装器
避免直接修改submodule，提供灵活的调用方式
"""

import logging
from typing import Dict, Any, List

from models.base import BaseASRModel
from core.models import ModelInfo
from core.enums import ModelType

# 导入独立的StepAudio2 API包装器
from models.step_audio2_api import StepAudio2API, StepAudio2Fallback


class StepAudio2Model(BaseASRModel):
    """StepAudio2模型实现 - 基于官方API"""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model_type = ModelType.STEP_AUDIO2
        self.api = None
        self.hotwords = model_config.get("hotwords", [])

    def load_model(self) -> bool:
        """加载StepAudio2模型 - 使用独立API包装器"""
        try:
            # 尝试使用独立API包装器
            api_config = {
                "model_path": self.config.get("model_path", "Model/StepAudio2/Step-Audio-2-mini"),
                "device": self.config.get("device", "cuda"),
                "api_url": self.config.get("api_url", None)
            }

            # 优先使用StepAudio2API
            self.api = StepAudio2API(api_config)
            if self.api.setup():
                self.is_loaded = True
                self.model_info = self.get_model_info()
                logging.info("StepAudio2 API设置成功")
                return True

            # 如果API不可用，使用后备实现
            logging.warning("使用StepAudio2后备实现")
            self.api = StepAudio2Fallback(api_config)
            if self.api.setup():
                self.is_loaded = True
                self.model_info = self.get_model_info()
                logging.info("StepAudio2后备实现加载成功")
                return True

            logging.error("StepAudio2加载失败")
            return False

        except Exception as e:
            logging.error(f"加载StepAudio2模型失败: {str(e)}")
            return False

    def transcribe_audio(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """使用StepAudio2进行音频转录"""
        if not self.is_loaded or self.api is None:
            raise RuntimeError("模型未加载")

        try:
            # 准备热词列表
            current_hotwords = kwargs.pop("hotwords", self.hotwords)

            # 调用StepAudio2 API进行转录
            result = self.api.transcribe_audio(
                audio_path,
                hotwords=current_hotwords,
                **kwargs
            )

            return {
                "text": result["text"],
                "confidence": result["confidence"],
                "language": result["language"],
                "processing_time": result["processing_time"],
                "timestamps": [],
                "model_info": {
                    "model_name": result.get("model_name", "Step-Audio-2-mini"),
                    "hotwords": current_hotwords
                }
            }

        except Exception as e:
            logging.error(f"StepAudio2 转录失败: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": "zh",
                "processing_time": 0.0,
                "error": str(e)
            }

    def get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        if self.api and self.is_loaded:
            info = self.api.get_model_info()
            mode = "API" if hasattr(self.api, 'model') else "后备"
            return ModelInfo(
                model_type=self.model_type,
                model_name=info["model_name"],
                version="2.0.0",
                description=f"StepAudio2 {mode}实现的高性能ASR模型",
                supported_languages=info["supported_languages"],
                supported_formats=info["supported_formats"],
                additional_info={
                    "model_size": "small",
                    "architecture": "transformer",
                    "sample_rate": info["sample_rate"],
                    "max_length": info["max_length"],
                    "hotword_support": True,
                    "mode": mode
                }
            )
        else:
            return ModelInfo(
                model_type=self.model_type,
                model_name="StepAudio2",
                version="2.0.0",
                description="StepAudio2 ASR模型",
                supported_languages=["zh", "en", "ja", "ko"],
                supported_formats=["wav", "mp3", "flac", "m4a", "ogg"],
                additional_info={
                    "model_size": "small",
                    "architecture": "transformer",
                    "hotword_support": True
                }
            )

    def preprocess_audio(self, audio_path: str) -> str:
        """音频预处理"""
        if self.api and self.is_loaded:
            if self.api.validate_audio(audio_path):
                return audio_path
            else:
                raise ValueError(f"无效的音频文件: {audio_path}")
        return audio_path

    def set_hotwords(self, hotwords: List[str]):
        """设置热词列表"""
        self.hotwords = hotwords
        logging.info(f"已设置热词: {hotwords}")

    def cleanup(self):
        """清理资源"""
        if self.api:
            self.api.cleanup()
        self.is_loaded = False
        logging.info("StepAudio2模型已清理")