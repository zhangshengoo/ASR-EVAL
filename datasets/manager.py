"""
数据集管理模块
管理各种测试数据集
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core.models import TestItem, TestDatasetConfig
from core.enums import TestDatasetType


logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """数据集信息"""
    dataset_type: TestDatasetType
    name: str
    description: str
    total_files: int
    total_duration: float
    sample_rate: int
    file_formats: List[str]
    languages: List[str]


class DatasetValidator:
    """数据集验证器"""

    @staticmethod
    def validate_audio_file(file_path: str) -> bool:
        """验证音频文件"""
        path = Path(file_path)
        if not path.exists():
            return False

        if not path.is_file():
            return False

        valid_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        return path.suffix.lower() in valid_extensions

    @staticmethod
    def validate_transcription_file(file_path: str) -> bool:
        """验证转录文件"""
        path = Path(file_path)
        if not path.exists():
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                return len(content) > 0
        except Exception:
            return False


class TestDatasetManager:
    """测试数据集管理器"""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/datasets.json"
        self.datasets: Dict[TestDatasetType, TestDatasetConfig] = {}
        self._load_config()

    def _load_config(self):
        """加载数据集配置"""
        if not os.path.exists(self.config_path):
            self._create_default_config()
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            for dataset_type_str, config in config_data.items():
                try:
                    dataset_type = TestDatasetType(dataset_type_str)
                    self.datasets[dataset_type] = TestDatasetConfig(
                        dataset_type=dataset_type,
                        path=config["path"],
                        description=config.get("description", ""),
                        audio_format=config.get("audio_format", "wav"),
                        sample_rate=config.get("sample_rate", 16000),
                        max_duration=config.get("max_duration"),
                        min_duration=config.get("min_duration")
                    )
                except (KeyError, ValueError) as e:
                    logger.warning(f"跳过无效的数据集配置 {dataset_type_str}: {e}")

        except Exception as e:
            logger.error(f"加载数据集配置失败: {e}")
            self._create_default_config()

    def _create_default_config(self):
        """创建默认配置"""
        default_config = {
            TestDatasetType.REGRESSION: TestDatasetConfig(
                dataset_type=TestDatasetType.REGRESSION,
                path="datasets/regression_test",
                description="标准回归测试集，包含各种场景的语音样本",
                audio_format="wav",
                sample_rate=16000
            ),
            TestDatasetType.NOISE: TestDatasetConfig(
                dataset_type=TestDatasetType.NOISE,
                path="datasets/noise_test",
                description="噪音环境测试集，包含不同信噪比的语音样本",
                audio_format="wav",
                sample_rate=16000
            ),
            TestDatasetType.LONG_AUDIO: TestDatasetConfig(
                dataset_type=TestDatasetType.LONG_AUDIO,
                path="datasets/long_audio_test",
                description="长音频测试集，包含超过30秒的语音样本",
                audio_format="wav",
                sample_rate=16000,
                min_duration=30.0
            ),
            TestDatasetType.ACCENT: TestDatasetConfig(
                dataset_type=TestDatasetType.ACCENT,
                path="datasets/accent_test",
                description="方言口音测试集，包含各种方言的语音样本",
                audio_format="wav",
                sample_rate=16000
            ),
            TestDatasetType.MULTILINGUAL: TestDatasetConfig(
                dataset_type=TestDatasetType.MULTILINGUAL,
                path="datasets/multilingual_test",
                description="多语言测试集，包含中文、英文、日文等多种语言",
                audio_format="wav",
                sample_rate=16000
            )
        }

        self.datasets = default_config
        self.save_config()

    def save_config(self):
        """保存配置到文件"""
        config_data = {}
        for dataset_type, config in self.datasets.items():
            config_data[dataset_type.value] = {
                "path": config.path,
                "description": config.description,
                "audio_format": config.audio_format,
                "sample_rate": config.sample_rate,
                "max_duration": config.max_duration,
                "min_duration": config.min_duration
            }

        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存数据集配置失败: {e}")

    def load_dataset(self, dataset_type: TestDatasetType) -> List[TestItem]:
        """加载指定数据集"""
        if dataset_type not in self.datasets:
            raise ValueError(f"未配置的数据集类型: {dataset_type}")

        dataset_config = self.datasets[dataset_type]
        dataset_path = Path(dataset_config.path)

        if not dataset_path.exists():
            logger.warning(f"数据集路径不存在: {dataset_path}")
            return []

        return self._load_dataset_from_path(dataset_config)

    def _load_dataset_from_path(self, dataset_config: TestDatasetConfig) -> List[TestItem]:
        """从路径加载数据集"""
        dataset_path = Path(dataset_config.path)
        test_items = []

        # 查找音频文件
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
        audio_files = []

        for ext in audio_extensions:
            audio_files.extend(dataset_path.glob(f'*{ext}'))
            audio_files.extend(dataset_path.glob(f'*{ext.upper()}'))

        # 查找对应的转录文件
        for audio_file in audio_files:
            transcription_file = self._find_transcription_file(audio_file)
            if transcription_file:
                reference_text = self._load_transcription(transcription_file)
                if reference_text:
                    test_items.append(TestItem(
                        audio_path=str(audio_file),
                        reference_text=reference_text,
                        dataset_type=dataset_config.dataset_type,
                        metadata={
                            "file_name": audio_file.name,
                            "file_size": audio_file.stat().st_size,
                            "dataset_type": dataset_config.dataset_type.value
                        }
                    ))

        logger.info(f"从 {dataset_config.path} 加载了 {len(test_items)} 个测试项")
        return test_items

    def _find_transcription_file(self, audio_file: Path) -> Optional[Path]:
        """查找对应的转录文件"""
        base_name = audio_file.stem
        possible_files = [
            audio_file.with_suffix('.txt'),
            audio_file.with_suffix('.trans'),
            audio_file.parent / f"{base_name}.txt",
            audio_file.parent / f"{base_name}.trans"
        ]

        for file_path in possible_files:
            if file_path.exists() and DatasetValidator.validate_transcription_file(str(file_path)):
                return file_path

        return None

    def _load_transcription(self, transcription_file: Path) -> str:
        """加载转录文本"""
        try:
            with open(transcription_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"加载转录文件失败 {transcription_file}: {e}")
            return ""

    def get_dataset_info(self, dataset_type: TestDatasetType) -> Optional[DatasetInfo]:
        """获取数据集信息"""
        if dataset_type not in self.datasets:
            return None

        dataset_config = self.datasets[dataset_type]
        test_items = self.load_dataset(dataset_type)

        if not test_items:
            return DatasetInfo(
                dataset_type=dataset_type,
                name=dataset_type.value,
                description=dataset_config.description,
                total_files=0,
                total_duration=0.0,
                sample_rate=dataset_config.sample_rate,
                file_formats=[dataset_config.audio_format],
                languages=["unknown"]
            )

        # 计算总时长（需要音频处理库支持）
        total_duration = 0.0  # TODO: 实际计算音频时长

        return DatasetInfo(
            dataset_type=dataset_type,
            name=dataset_type.value,
            description=dataset_config.description,
            total_files=len(test_items),
            total_duration=total_duration,
            sample_rate=dataset_config.sample_rate,
            file_formats=[dataset_config.audio_format],
            languages=["zh"]  # TODO: 从音频中检测语言
        )

    def list_available_datasets(self) -> List[TestDatasetType]:
        """列出可用的数据集类型"""
        return list(self.datasets.keys())

    def validate_dataset(self, dataset_type: TestDatasetType) -> bool:
        """验证数据集完整性"""
        if dataset_type not in self.datasets:
            return False

        dataset_config = self.datasets[dataset_type]
        dataset_path = Path(dataset_config.path)

        if not dataset_path.exists():
            return False

        test_items = self.load_dataset(dataset_type)
        return len(test_items) > 0

    def create_sample_dataset(self, dataset_type: TestDatasetType, sample_size: int = 10):
        """创建示例数据集"""
        dataset_config = self.datasets[dataset_type]
        dataset_path = Path(dataset_config.path)

        try:
            os.makedirs(dataset_path, exist_ok=True)

            # 创建示例音频文件和转录文件
            for i in range(sample_size):
                audio_file = dataset_path / f"sample_{i+1:03d}.wav"
                transcription_file = dataset_path / f"sample_{i+1:03d}.txt"

                # TODO: 生成示例音频文件
                # 这里只是创建空文件作为占位符
                audio_file.touch()

                # 创建示例转录
                sample_texts = [
                    "这是一个示例语音文件",
                    "你好世界",
                    "语音识别测试",
                    "开源ASR模型评估",
                    "中文语音转录测试"
                ]
                text = sample_texts[i % len(sample_texts)]

                with open(transcription_file, 'w', encoding='utf-8') as f:
                    f.write(text)

            logger.info(f"已创建示例数据集: {dataset_type.value} ({sample_size} 个样本)")

        except Exception as e:
            logger.error(f"创建示例数据集失败: {e}")

    def get_dataset_config(self, dataset_type: TestDatasetType) -> Optional[TestDatasetConfig]:
        """获取数据集配置"""
        return self.datasets.get(dataset_type)