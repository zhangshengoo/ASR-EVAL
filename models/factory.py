"""
模型工厂
用于创建和管理ASR模型实例
"""

from typing import Dict, Any
import logging

from core.enums import ModelType
from core.models import ModelConfig
from models.base import BaseASRModel

# 导入具体的模型实现
from models.kimi_audio import KimiAudioModel
from models.fire_red_asr import FireRedASRModel
from models.step_audio2 import StepAudio2Model


logger = logging.getLogger(__name__)


class ModelFactory:
    """模型工厂类"""

    _model_registry = {
        ModelType.KIMI_AUDIO: KimiAudioModel,
        ModelType.FIRE_RED_ASR: FireRedASRModel,
        ModelType.STEP_AUDIO2: StepAudio2Model,
    }

    @classmethod
    def create_model(cls, model_config: ModelConfig) -> BaseASRModel:
        """
        创建模型实例

        Args:
            model_config: 模型配置

        Returns:
            BaseASRModel: 模型实例

        Raises:
            ValueError: 模型类型不支持
            RuntimeError: 模型创建失败
        """
        model_type = model_config.model_type

        if model_type not in cls._model_registry:
            raise ValueError(f"不支持的模型类型: {model_type}")

        model_class = cls._model_registry[model_type]

        try:
            # 转换配置为字典格式
            config_dict = {
                "model_path": model_config.model_path,
                "config_file": model_config.config_file,
                "device": model_config.device,
                "batch_size": model_config.batch_size,
                "language": model_config.language,
                **model_config.additional_params
            }

            model = model_class(config_dict)
            logger.info(f"成功创建模型实例: {model_type.value}")
            return model

        except Exception as e:
            logger.error(f"创建模型实例失败 {model_type.value}: {str(e)}")
            raise RuntimeError(f"创建模型实例失败: {str(e)}")

    @classmethod
    def register_model(cls, model_type: ModelType, model_class: type):
        """
        注册新的模型类型

        Args:
            model_type: 模型类型
            model_class: 模型类（必须继承BaseASRModel）
        """
        if not issubclass(model_class, BaseASRModel):
            raise ValueError("模型类必须继承BaseASRModel")

        cls._model_registry[model_type] = model_class
        logger.info(f"已注册模型类型: {model_type.value}")

    @classmethod
    def list_supported_models(cls) -> list[ModelType]:
        """获取支持的模型类型列表"""
        return list(cls._model_registry.keys())

    @classmethod
    def is_model_supported(cls, model_type: ModelType) -> bool:
        """检查是否支持指定模型类型"""
        return model_type in cls._model_registry


def create_model_pool(model_configs: Dict[ModelType, ModelConfig]) -> Dict[ModelType, BaseASRModel]:
    """
    创建模型池

    Args:
        model_configs: 模型配置字典

    Returns:
        Dict[ModelType, BaseASRModel]: 模型实例字典
    """
    model_pool = {}

    for model_type, config in model_configs.items():
        if not config.enabled:
            logger.info(f"跳过禁用的模型: {model_type.value}")
            continue

        try:
            model = ModelFactory.create_model(config)
            model_pool[model_type] = model
            logger.info(f"模型已添加到模型池: {model_type.value}")
        except Exception as e:
            logger.error(f"添加模型到池失败 {model_type.value}: {str(e)}")
            continue

    return model_pool


def load_and_initialize_models(model_configs: Dict[ModelType, ModelConfig]) -> Dict[ModelType, BaseASRModel]:
    """
    加载并初始化模型

    Args:
        model_configs: 模型配置字典

    Returns:
        Dict[ModelType, BaseASRModel]: 已加载的模型实例字典
    """
    loaded_models = {}
    model_pool = create_model_pool(model_configs)

    for model_type, model in model_pool.items():
        try:
            logger.info(f"正在加载模型: {model_type.value}")
            success = model.load_model()

            if success:
                loaded_models[model_type] = model
                logger.info(f"模型加载成功: {model_type.value}")
            else:
                logger.error(f"模型加载失败: {model_type.value}")

        except Exception as e:
            logger.error(f"加载模型时出错 {model_type.value}: {str(e)}")
            continue

    logger.info(f"成功加载 {len(loaded_models)}/{len(model_pool)} 个模型")
    return loaded_models