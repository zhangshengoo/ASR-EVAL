"""
枚举定义模块
定义系统中使用的所有枚举类型
"""

from enum import Enum


class ModelType(Enum):
    """支持的ASR模型类型"""
    KIMI_AUDIO = "kimi_audio"
    FIRE_RED_ASR = "fire_red_asr"
    STEP_AUDIO2 = "step_audio2"
    WHISPER = "whisper"
    WENET = "wenet"


class TestDatasetType(Enum):
    """测试数据集类型"""
    REGRESSION = "regression"  # 回归测试
    NOISE = "noise"  # 噪音测试
    LONG_AUDIO = "long_audio"  # 长音频测试
    ACCENT = "accent"  # 方言测试
    MULTILINGUAL = "multilingual"  # 多语言测试


class MetricType(Enum):
    """支持的评估指标类型"""
    WER = "wer"  # Word Error Rate
    CER = "cer"  # Character Error Rate
    SER = "ser"  # Sentence Error Rate
    RTF = "rtf"  # Real-time Factor
    CONFIDENCE = "confidence"  # 置信度
    PROCESSING_TIME = "processing_time"  # 处理时间


class AudioFormat(Enum):
    """支持的音频格式"""
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    M4A = "m4a"
    OGG = "ogg"