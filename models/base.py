"""
ASR模型基类
定义所有ASR模型必须实现的接口
新增多进程并行处理框架，供所有模型共用
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

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

        # 多进程并行处理配置
        self.parallel_enabled = model_config.get("parallel_enabled", False)
        self.num_processes = model_config.get("num_processes", None)  # None表示使用CPU核心数
        self.available_gpus = model_config.get("available_gpus", None)  # 指定可用GPU列表
        self.parallel_batch_size = model_config.get("parallel_batch_size", 1)  # 每批处理的音频数量

        # 热词支持
        self.hotwords_enabled = model_config.get("hotwords_enabled", True)
        self.current_hotwords = []

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
    def transcribe_audio_batch(self, audio_paths: List[str], device: str, **kwargs) -> List[Dict[str, Any]]:
        """
        模型特定的批量推理逻辑（用于并行处理）

        Args:
            audio_paths: 音频文件路径列表
            device: 设备标识（如"cuda:0"）
            **kwargs: 推理参数

        Returns:
            List[Dict]: 每个音频的转录结果
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
        批量推理 - 支持并行处理和每个样本的热词库

        Args:
            audio_items: [{
                "audio_path": str,
                "reference_text": str,
                "hotwords": List[str],  # 可选：该样本的热词库
                "sample_id": str,       # 可选：样本ID
                "config_name": str      # 可选：配置名称
            }, ...]

        Returns:
            List[TestResult]: 测试结果列表
        """
        if not self.is_loaded:
            raise RuntimeError("模型未加载，请先调用load_model()")

        # 检查是否所有样本都有热词信息
        has_per_sample_hotwords = any(item.get("hotwords") is not None for item in audio_items)

        if has_per_sample_hotwords:
            # 如果样本包含热词信息，使用支持每个样本热词的批处理方法
            return self._batch_inference_with_per_sample_hotwords(audio_items)
        else:
            # 使用标准批处理方法（所有样本使用相同的热词库或没有热词）
            audio_paths = [item["audio_path"] for item in audio_items]

            if self.parallel_enabled and len(audio_paths) > 1:
                # 使用并行推理
                parallel_results = self.transcribe_audio_parallel(audio_paths)

                # 转换为TestResult格式
                results = []
                for i, result in enumerate(parallel_results):
                    audio_item = audio_items[i]
                    test_result = TestResult(
                        audio_path=audio_item["audio_path"],
                        model_type=self.model_type,
                        reference_text=audio_item.get("reference_text", ""),
                        predicted_text=result.get("text", ""),
                        processing_time=result.get("processing_time", 0.0),
                        confidence_score=result.get("confidence", 0.0),
                        word_timestamps=result.get("timestamps", None)
                    )
                    results.append(test_result)

                return results
            else:
                # 使用串行处理
                return self._serial_batch_inference(audio_items)

    def _serial_batch_inference(self, audio_items: List[Dict[str, Any]]) -> List[TestResult]:
        """
        串行批量推理

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

    def _batch_inference_with_per_sample_hotwords(self, audio_items: List[Dict[str, Any]]) -> List[TestResult]:
        """
        支持每个样本不同热词库的批量推理

        Args:
            audio_items: 包含热词信息的音频项目列表

        Returns:
            List[TestResult]: 测试结果列表
        """
        results = []

        # 串行处理每个样本（因为每个样本可能有不同的热词）
        for item in audio_items:
            try:
                # 设置该样本的热词
                hotwords = item.get("hotwords", [])
                if hotwords:
                    self.set_hotwords(hotwords)
                else:
                    self.clear_hotwords()

                # 执行单个推理
                result = self.run_inference(
                    item["audio_path"],
                    item.get("reference_text", "")
                )
                results.append(result)

            except Exception as e:
                print(f"推理失败: {item.get('audio_path', 'unknown')}, 错误: {str(e)}")
                # 创建错误结果
                error_result = TestResult(
                    audio_path=item.get("audio_path", ""),
                    model_type=self.model_type,
                    reference_text=item.get("reference_text", ""),
                    predicted_text="",
                    processing_time=0.0,
                    confidence_score=0.0,
                    word_timestamps=None
                )
                results.append(error_result)
                continue

        return results

    def transcribe_audio_parallel(self, audio_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        多进程并行转录音频文件 - 通用框架

        Args:
            audio_paths: 音频文件路径列表
            **kwargs: 额外参数，包括:
                - batch_size: 每批处理的音频数量
                - gpu_ids: 指定使用的GPU ID列表
                - max_workers: 最大进程数

        Returns:
            List[Dict]: 转录结果列表，每个元素包含:
                - text: 转录文本
                - confidence: 置信度分数
                - timestamps: 时间戳信息
                - processing_time: 处理时间
                - language: 识别语言
                - audio_path: 音频文件路径
        """
        if not self.parallel_enabled:
            # 如果并行处理未启用，使用串行处理
            return [self.transcribe_audio(audio_path, **kwargs) for audio_path in audio_paths]

        # 获取GPU配置
        gpu_ids = kwargs.get('gpu_ids', self._get_available_gpus())
        max_workers = kwargs.get('max_workers', self.num_processes or mp.cpu_count())
        batch_size = kwargs.get('batch_size', self.parallel_batch_size)

        # 限制进程数不超过GPU数和音频文件数
        num_gpus = len(gpu_ids) if gpu_ids else 1
        max_workers = min(max_workers, num_gpus, len(audio_paths))

        print(f"启动多进程并行推理: 进程数={max_workers}, GPU数={num_gpus}, 音频数={len(audio_paths)}")

        # 准备进程参数
        start_time = time.time()

        # 将音频文件分组到不同的GPU
        audio_batches = self._distribute_audios_to_gpus(audio_paths, gpu_ids, batch_size)

        # 使用进程池进行并行处理
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp.get_context('spawn')) as executor:
            # 提交任务
            futures = []
            for gpu_id, audio_batch in audio_batches:
                future = executor.submit(
                    BaseASRModel._process_audio_batch_parallel_static,
                    audio_batch,
                    gpu_id,
                    self._get_model_config_for_parallel(),
                    kwargs,
                    self  # 传递模型实例
                )
                futures.append(future)

            # 收集结果
            results = []
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    print(f"批处理失败: {e}")
                    # 为失败的批次创建错误结果
                    for audio_path in audio_batch:
                        results.append({
                            "text": "",
                            "confidence": 0.0,
                            "timestamps": None,
                            "processing_time": 0.0,
                            "language": "unknown",
                            "audio_path": audio_path,
                            "error": str(e)
                        })

        total_time = time.time() - start_time
        print(f"多进程并行推理完成: 总时间={total_time:.2f}s, 平均每个音频={total_time/len(audio_paths):.2f}s")

        return results

    def _get_available_gpus(self) -> List[int]:
        """获取可用的GPU列表"""
        import torch

        if self.available_gpus is not None:
            return self.available_gpus

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            if gpu_count > 0:
                # 默认使用所有可用GPU
                available_gpus = list(range(gpu_count))
                print(f"检测到 {gpu_count} 个GPU，将使用所有GPU: {available_gpus}")
                return available_gpus
            else:
                print("未检测到GPU，使用CPU")
                return [0]  # 使用CPU
        else:
            print("PyTorch CUDA不可用，使用CPU")
            return [0]  # 使用CPU

    def _distribute_audios_to_gpus(self, audio_paths: List[str], gpu_ids: List[int], batch_size: int) -> List[tuple]:
        """
        将音频文件分配到不同的GPU上进行处理

        Args:
            audio_paths: 音频文件路径列表
            gpu_ids: GPU ID列表
            batch_size: 每批处理的音频数量

        Returns:
            List[tuple]: [(gpu_id, audio_batch), ...]
        """
        batches = []

        # 计算每个GPU应该处理的音频数量
        num_gpus = len(gpu_ids)
        audios_per_gpu = len(audio_paths) // num_gpus
        remainder = len(audio_paths) % num_gpus

        start_idx = 0
        for i, gpu_id in enumerate(gpu_ids):
            # 计算当前GPU的音频数量
            gpu_audio_count = audios_per_gpu + (1 if i < remainder else 0)

            if gpu_audio_count == 0:
                continue

            # 获取当前GPU的音频列表
            gpu_audios = audio_paths[start_idx:start_idx + gpu_audio_count]
            start_idx += gpu_audio_count

            # 将音频分成批次
            for j in range(0, len(gpu_audios), batch_size):
                batch = gpu_audios[j:j + batch_size]
                batches.append((gpu_id, batch))

        return batches

    def _get_model_config_for_parallel(self) -> Dict[str, Any]:
        """
        获取用于并行处理的模型配置

        Returns:
            Dict: 模型配置信息
        """
        return {
            "model_path": getattr(self, 'model_path', ''),
            "model_type": self.model_type.value if self.model_type else '',
            "device": getattr(self, 'device', 'cuda'),
            "sample_rate": getattr(self, 'sample_rate', 16000),
            # 可以添加其他模型特定的配置
        }

    def _process_audio_batch_parallel_wrapper(self, audio_batch: List[str], gpu_id: int, model_config: Dict[str, Any], inference_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        在指定GPU上处理一批音频文件 - 包装器方法

        Args:
            audio_batch: 音频文件路径列表
            gpu_id: GPU ID
            model_config: 模型配置
            inference_params: 推理参数

        Returns:
            List[Dict]: 转录结果列表
        """
        import torch
        import os
        # 设置当前进程的GPU设备
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"进程 {os.getpid()} 使用 GPU {gpu_id} 处理 {len(audio_batch)} 个音频")

        try:
            # 调用模型特定的批量推理逻辑
            return self._call_model_specific_batch_inference(
                audio_batch, gpu_id, model_config, inference_params
            )

        except Exception as e:
            print(f"批处理失败: {e}")
            # 返回错误结果
            return [{
                "text": "",
                "confidence": 0.0,
                "timestamps": None,
                "processing_time": 0.0,
                "language": "unknown",
                "audio_path": audio_path,
                "error": str(e)
            } for audio_path in audio_batch]

    @staticmethod
    def _process_audio_batch_parallel_static(audio_batch: List[str], gpu_id: int, model_config: Dict[str, Any], inference_params: Dict[str, Any], model_instance) -> List[Dict[str, Any]]:
        """
        静态方法版本，用于进程池调用
        """
        import torch
        import os

        # 设置当前进程的GPU设备
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(f"进程 {os.getpid()} 使用 GPU {gpu_id} 处理 {len(audio_batch)} 个音频")

        try:
            # 调用模型实例的批量推理逻辑
            return model_instance._call_model_specific_batch_inference(
                audio_batch, gpu_id, model_config, inference_params
            )

        except Exception as e:
            print(f"批处理失败: {e}")
            # 返回错误结果
            return [{
                "text": "",
                "confidence": 0.0,
                "timestamps": None,
                "processing_time": 0.0,
                "language": "unknown",
                "audio_path": audio_path,
                "error": str(e)
            } for audio_path in audio_batch]

    def _call_model_specific_batch_inference(self, audio_batch: List[str], gpu_id: int, model_config: Dict[str, Any], inference_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        调用模型特定的批量推理逻辑

        子类必须重写这个方法来提供模型特定的批量推理实现
        """
        # 获取设备字符串
        device = f"cuda:0" if torch.cuda.is_available() and gpu_id < torch.cuda.device_count() else "cpu"
        _ = model_config  # 避免未使用变量警告

        # 调用模型特定的批量推理逻辑
        return self.transcribe_audio_batch(audio_batch, device, **inference_params)

    def cleanup(self):
        """清理资源"""
        pass

    def set_hotwords(self, hotwords: List[str]) -> bool:
        """
        设置模型的热词库

        Args:
            hotwords: 热词列表

        Returns:
            bool: 设置成功返回True，失败返回False
        """
        if not self.hotwords_enabled:
            return False

        self.current_hotwords = hotwords.copy() if hotwords else []
        return True

    def get_hotwords(self) -> List[str]:
        """
        获取当前热词库

        Returns:
            List[str]: 当前热词列表
        """
        return self.current_hotwords.copy()

    def clear_hotwords(self) -> bool:
        """
        清除热词库

        Returns:
            bool: 清除成功返回True
        """
        self.current_hotwords = []
        return True


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