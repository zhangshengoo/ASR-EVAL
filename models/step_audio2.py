"""
StepAudio2模型实现
基于BaseASRModel实现本地模型加载和推理
使用新的并行处理架构，模型只需实现特定的推理逻辑
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import time
import torch

# 添加StepAudio2模型路径到系统路径
stepaudio2_path = Path(__file__).parent.parent / "Model" / "StepAudio2"
sys.path.insert(0, str(stepaudio2_path))

# 直接导入，让ImportError直接抛出
from stepaudio2 import StepAudio2 as StepAudio2Core
from utils import load_audio, log_mel_spectrogram, padding_mels
STEP_AUDIO_AVAILABLE = True

from models.base import BaseASRModel
from core.models import ModelInfo
from core.enums import ModelType


class StepAudio2Model(BaseASRModel):
    """StepAudio2 ASR模型实现

    使用BaseASRModel的并行处理框架，只需实现模型特定的推理逻辑
    """

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.model_type = ModelType.STEP_AUDIO2
        self.model_path = model_config.get("model_path", "Step-Audio-2-mini")
        self.device = model_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sample_rate = 16000

        # 热词配置 - 简化版本
        self.hotwords = model_config.get("hotwords", [])

    def load_model(self) -> bool:
        """
        加载StepAudio2模型

        Returns:
            bool: 加载成功返回True，失败返回False
        """
        # 检查模型路径
        model_full_path = Path(self.model_path)
        if not model_full_path.exists():
            # 尝试相对路径
            relative_path = Path(__file__).parent.parent / "Model" / "StepAudio2" / self.model_path
            if relative_path.exists():
                model_full_path = relative_path
            else:
                raise FileNotFoundError(f"模型路径不存在: {self.model_path}")

        print(f"正在加载StepAudio2模型: {model_full_path}")

        # 初始化模型
        self.model = StepAudio2Core(str(model_full_path))
        self.is_loaded = True

        print("StepAudio2模型加载成功")
        return True

    def transcribe_audio(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        转录音频文件 - 单次推理

        Args:
            audio_path: 音频文件路径
            **kwargs: 额外参数

        Returns:
            Dict包含:
            - text: 转录文本
            - confidence: 置信度分数
            - timestamps: 时间戳信息
            - processing_time: 处理时间
            - language: 识别语言
        """
        start_time = time.time()

        # 构建消息格式（包含热词信息）
        messages = self._prepare_messages_with_hotwords(audio_path)

        # 调用模型进行推理
        tokens, output_text, _ = self.model(
            messages,
            max_new_tokens=kwargs.get("max_new_tokens", 256),
            temperature=kwargs.get("temperature", 0.7),
            repetition_penalty=kwargs.get("repetition_penalty", 1.05),
            top_p=kwargs.get("top_p", 0.9),
            do_sample=kwargs.get("do_sample", True)
        )

        processing_time = time.time() - start_time

        # 清理输出文本
        output_text = self._clean_transcription(output_text)

        return {
            "text": output_text,
            "confidence": self._calculate_confidence(tokens),
            "timestamps": None,  # StepAudio2不直接提供时间戳
            "processing_time": processing_time,
            "language": self._detect_language(output_text)
        }

    def transcribe_audio_batch(self, audio_paths: List[str], device: str, **kwargs) -> List[Dict[str, Any]]:
        """
        批量转录音频文件 - 用于并行处理

        Args:
            audio_paths: 音频文件路径列表
            device: 设备标识（如"cuda:0"）
            **kwargs: 推理参数

        Returns:
            List[Dict]: 每个音频的转录结果
        """
        # 注意：在并行处理中，每个进程会独立加载模型
        # 所以我们需要在这里重新加载模型到指定的设备

        # 重新导入StepAudio2模块（每个进程需要独立加载）
        stepaudio2_path = Path(__file__).parent.parent / "Model" / "StepAudio2"
        sys.path.insert(0, str(stepaudio2_path))

        from stepaudio2 import StepAudio2 as StepAudio2Core

        # 加载模型到指定设备
        model = StepAudio2Core(str(Path(self.model_path)))

        # 设置设备（如果支持）
        if torch.cuda.is_available():
            device_id = 0 if device.startswith("cuda:") else int(device.split(":")[-1]) if ":" in device else 0
            if device_id < torch.cuda.device_count():
                torch.cuda.set_device(device_id)

        # 处理音频批次
        results = []
        for audio_path in audio_paths:
            start_time = time.time()

            # 构建消息格式（包含热词信息）
            messages = self._prepare_messages_with_hotwords(audio_path)

            # 调用模型进行推理
            tokens, output_text, _ = model(
                messages,
                max_new_tokens=kwargs.get("max_new_tokens", 256),
                temperature=kwargs.get("temperature", 0.7),
                repetition_penalty=kwargs.get("repetition_penalty", 1.05),
                top_p=kwargs.get("top_p", 0.9),
                do_sample=kwargs.get("do_sample", True)
            )

            processing_time = time.time() - start_time

            # 清理输出文本
            output_text = self._clean_transcription(output_text)

            result = {
                "text": output_text,
                "confidence": self._calculate_confidence(tokens),
                "timestamps": None,
                "processing_time": processing_time,
                "language": self._detect_language(output_text),
                "audio_path": audio_path
            }

            results.append(result)

        # 清理模型资源
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        return ModelInfo(
            name="StepAudio2",
            version="2.0",
            description="StepAudio2多模态音频理解模型",
            supported_languages=["zh", "en", "ja"],
            sample_rate=self.sample_rate,
            parameters={
                "model_path": self.model_path,
                "device": self.device,
                "max_new_tokens": 256,
                "temperature": 0.7,
                "repetition_penalty": 1.05,
                "top_p": 0.9
            }
        )

    def preprocess_audio(self, audio_path: str) -> str:
        """
        音频预处理

        Args:
            audio_path: 原始音频路径

        Returns:
            处理后的音频路径（StepAudio2内部处理，直接返回原路径）
        """
        # StepAudio2内部会处理音频加载和预处理
        return audio_path

    def _clean_transcription(self, text: str) -> str:
        """清理转录文本"""
        # 移除特殊标记和多余空白
        text = text.strip()
        # 移除模型生成的特殊标记
        text = text.replace("<|BOT|>", "").replace("<|EOT|>", "")
        text = text.replace("human\n", "").replace("assistant\n", "")
        # 清理多余空白
        text = " ".join(text.split())
        return text

    def _calculate_confidence(self, tokens: list) -> float:
        """
        计算置信度分数

        Args:
            tokens: 输出的token列表

        Returns:
            置信度分数（0-1）
        """
        # 简化置信度计算：基于token的合理性
        if not tokens:
            return 0.0

        # 检查是否有异常token
        valid_tokens = [t for t in tokens if t < 151688]  # 文本token范围
        if len(valid_tokens) == 0:
            return 0.0

        return min(len(valid_tokens) / len(tokens), 1.0)

    def _detect_language(self, text: str) -> str:
        """
        简单语言检测

        Args:
            text: 转录文本

        Returns:
            语言代码
        """
        if not text:
            return "unknown"

        # 基于Unicode范围简单判断
        if any('\u4e00' <= c <= '\u9fff' for c in text):
            return "zh"
        elif any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text):
            return "ja"
        else:
            return "en"

    def cleanup(self):
        """清理资源"""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def set_hotwords(self, hotwords: List[str]) -> bool:
        """
        设置StepAudio2的热词库

        Args:
            hotwords: 热词列表

        Returns:
            bool: 设置成功返回True
        """
        self.current_hotwords = hotwords.copy() if hotwords else []
        return True

    def _prepare_messages_with_hotwords(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        准备包含热词信息的messages

        Args:
            audio_path: 音频文件路径

        Returns:
            messages列表
        """
        # 基础messages格式
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Transcribe the given audio accurately."
            },
            {
                "role": "human",
                "content": [{"type": "audio", "audio": audio_path}]
            },
            {
                "role": "assistant",
                "content": None
            }
        ]

        # 如果有热词，添加到系统提示中
        if self.current_hotwords:
            hotwords_text = ", ".join(self.current_hotwords)
            messages[0]["content"] = f"You are a helpful assistant. Transcribe the given audio accurately. Pay special attention to these keywords: {hotwords_text}"

        return messages