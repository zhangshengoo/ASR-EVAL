"""
数据模型定义
定义系统中使用的所有数据类
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from .enums import ModelType, TestDatasetType, MetricType


@dataclass
class ModelConfig:
    """模型配置信息"""
    model_type: ModelType
    model_path: str
    config_file: Optional[str] = None
    enabled: bool = True
    device: str = "cuda"
    batch_size: int = 1
    language: str = "zh"
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestDatasetConfig:
    """测试数据集配置"""
    dataset_type: TestDatasetType
    path: str
    description: str = ""
    audio_format: str = "wav"
    sample_rate: int = 16000
    max_duration: Optional[float] = None
    min_duration: Optional[float] = None


@dataclass
class TestItem:
    """单个测试项"""
    audio_path: str
    reference_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    dataset_type: Optional[TestDatasetType] = None


@dataclass
class TestResult:
    """单个测试结果"""
    audio_path: str
    model_type: ModelType
    reference_text: str
    predicted_text: str
    processing_time: float
    confidence_score: Optional[float] = None
    word_timestamps: Optional[List[Dict[str, float]]] = None
    char_timestamps: Optional[List[Dict[str, float]]] = None
    error_details: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Metrics:
    """评估指标"""
    wer: Optional[float] = None  # Word Error Rate
    cer: Optional[float] = None  # Character Error Rate
    ser: Optional[float] = None  # Sentence Error Rate
    processing_time_avg: Optional[float] = None
    realtime_factor: Optional[float] = None
    confidence_avg: Optional[float] = None
    confidence_min: Optional[float] = None
    confidence_max: Optional[float] = None
    audio_duration_avg: Optional[float] = None
    additional_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class TestReport:
    """测试报告"""
    model_type: ModelType
    dataset_type: TestDatasetType
    test_count: int
    metrics: Metrics
    results: List[TestResult]
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ComparativeReport:
    """对比测试报告"""
    dataset_type: TestDatasetType
    test_count: int
    model_reports: Dict[ModelType, TestReport]
    comparative_metrics: Dict[str, Dict[ModelType, float]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ModelInfo:
    """模型信息"""
    model_type: ModelType
    model_name: str
    version: str
    description: str
    supported_languages: List[str]
    supported_formats: List[str]
    additional_info: Dict[str, Any] = field(default_factory=dict)