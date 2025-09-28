"""
ASR模型基类
定义所有ASR模型必须实现的接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
from pathlib import Path

from core.models import ModelInfo, TestResult
from core.enums import ModelType


class BaseASRModel(ABC):
    """ASR模型基类"""

    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.model_type = None
        self.model_name = self.__class__.__name__
        self.is_loaded = False
        self.model_info = None

    @abstractmethod
    def load_model(self) -> bool:
        """
        加载模型

        Returns:
            bool: 加载成功返回True，失败返回False
        """
        pass

    @abstractmethod
    def transcribe_audio(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """
        转录音频文件

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
        pass

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """获取模型信息"""
        pass

    def preprocess_audio(self, audio_path: str) -> str:
        """
        音频预处理

        Args:
            audio_path: 原始音频路径

        Returns:
            处理后的音频路径
        """
        # 默认不做处理，子类可重写
        return audio_path

    def postprocess_text(self, text: str) -> str:
        """
        文本后处理

        Args:
            text: 原始转录文本

        Returns:
            处理后的文本
        """
        # 默认不做处理，子类可重写
        return text.strip()

    def validate_audio(self, audio_path: str) -> bool:
        """
        验证音频文件

        Args:
            audio_path: 音频文件路径

        Returns:
            bool: 音频文件是否有效
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        if not audio_file.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
            raise ValueError(f"不支持的音频格式: {audio_file.suffix}")

        return True

    def run_inference(self, audio_path: str, reference_text: str = "") -> TestResult:
        """
        运行单个推理任务

        Args:
            audio_path: 音频文件路径
            reference_text: 参考文本（可选）

        Returns:
            TestResult: 测试结果
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用load_model()")

        # 验证音频
        self.validate_audio(audio_path)

        # 预处理
        processed_audio = self.preprocess_audio(audio_path)

        # 执行推理
        start_time = time.time()
        result = self.transcribe_audio(processed_audio)
        processing_time = time.time() - start_time

        # 后处理
        predicted_text = self.postprocess_text(result.get("text", ""))

        return TestResult(
            audio_path=audio_path,
            model_type=self.model_type,
            reference_text=reference_text,
            predicted_text=predicted_text,
            processing_time=processing_time,
            confidence_score=result.get("confidence"),
            word_timestamps=result.get("timestamps")
        )

    def batch_inference(self, audio_items: List[Dict[str, Any]]) -> List[TestResult]:
        """
        批量推理

        Args:
            audio_items: [{"audio_path": str, "reference_text": str}, ...]

        Returns:
            List[TestResult]: 测试结果列表
        """
        results = []
        for item in audio_items:
            try:
                result = self.run_inference(
                    item["audio_path"],
                    item.get("reference_text", "")
                )
                results.append(result)
            except Exception as e:
                print(f"推理失败: {item.get('audio_path', 'unknown')}, 错误: {str(e)}")
                continue

        return results

    def cleanup(self):
        """清理资源"""
        pass


class ModelRegistry:
    """模型注册器"""

    _models = {}

    @classmethod
    def register(cls, model_type: ModelType, model_class: type):
        """注册模型"""
        cls._models[model_type] = model_class

    @classmethod
    def get_model_class(cls, model_type: ModelType) -> Optional[type]:
        """获取模型类"""
        return cls._models.get(model_type)

    @classmethod
    def list_models(cls) -> List[ModelType]:
        """列出所有注册的模型类型"""
        return list(cls._models.keys())

    @classmethod
    def create_model(cls, model_type: ModelType, config: Dict[str, Any]) -> BaseASRModel:
        """创建模型实例"""
        model_class = cls.get_model_class(model_type)
        if model_class is None:
            raise ValueError(f"未注册的模型类型: {model_type}")

        return model_class(config)