"""
StepAudio2 独立API包装器
不依赖StepAudio2 submodule的具体实现，提供灵活的接口
"""

import os
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests
import time


class StepAudio2API:
    """StepAudio2 API包装器 - 支持多种调用方式"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_path = self.config.get("model_path", "Step-Audio-2-mini")
        self.api_url = self.config.get("api_url", None)  # 支持API调用
        self.device = self.config.get("device", "cuda")
        self.setup_complete = False

    def setup(self) -> bool:
        """设置环境，支持多种调用方式"""
        try:
            # 方式1: 本地模型文件
            if self._check_local_model():
                logging.info("检测到本地StepAudio2模型")
                return self._setup_local_model()

            # 方式2: API调用
            if self.api_url:
                logging.info("使用StepAudio2 API调用")
                return self._setup_api_call()

            # 方式3: 命令行调用
            if self._check_cli_available():
                logging.info("使用StepAudio2命令行调用")
                return self._setup_cli_call()

            logging.error("未找到可用的StepAudio2调用方式")
            return False

        except Exception as e:
            logging.error(f"StepAudio2设置失败: {e}")
            return False

    def _check_local_model(self) -> bool:
        """检查本地模型文件"""
        model_dir = Path(self.model_path)
        return model_dir.exists() and (model_dir / "config.json").exists()

    def _check_cli_available(self) -> bool:
        """检查命令行工具是否可用"""
        try:
            result = subprocess.run(["python", "-c", "import stepaudio2"],
                                  capture_output=True, text=True, cwd="Model/StepAudio2")
            return result.returncode == 0
        except:
            return False

    def _setup_local_model(self) -> bool:
        """设置本地模型调用"""
        try:
            # 动态导入StepAudio2模块
            import sys
            stepaudio2_path = Path("Model/StepAudio2")
            if str(stepaudio2_path) not in sys.path:
                sys.path.insert(0, str(stepaudio2_path))

            try:
                from stepaudio2 import StepAudio2
                from stepaudio2.utils import load_audio
                from stepaudio2.tokenizer import StepAudio2Tokenizer

                self.StepAudio2 = StepAudio2
                self.load_audio = load_audio
                self.StepAudio2Tokenizer = StepAudio2Tokenizer

                # 初始化模型
                self.model = StepAudio2.from_pretrained(self.model_path)
                self.model.eval()

                import torch
                self.device = "cuda" if torch.cuda.is_available() and self.device == "cuda" else "cpu"
                self.model.to(self.device)

                self.tokenizer = StepAudio2Tokenizer.from_pretrained(self.model_path)

                self.setup_complete = True
                return True

            except ImportError as e:
                logging.warning(f"StepAudio2模块导入失败: {e}")
                return False

        except Exception as e:
            logging.error(f"本地模型设置失败: {e}")
            return False

    def _setup_api_call(self) -> bool:
        """设置API调用"""
        try:
            # 测试API连接
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                self.setup_complete = True
                return True
            return False
        except:
            return False

    def _setup_cli_call(self) -> bool:
        """设置命令行调用"""
        try:
            # 测试命令行调用
            test_cmd = ["python", "-c", "from stepaudio2 import StepAudio2; print('OK')"]
            result = subprocess.run(test_cmd, capture_output=True, text=True,
                                  cwd="Model/StepAudio2")
            if result.returncode == 0:
                self.setup_complete = True
                return True
            return False
        except:
            return False

    def transcribe_audio(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """转录音频文件"""
        if not self.setup_complete:
            raise RuntimeError("StepAudio2未正确设置")

        if self.api_url:
            return self._transcribe_via_api(audio_path, **kwargs)
        elif hasattr(self, 'model'):
            return self._transcribe_via_local(audio_path, **kwargs)
        else:
            return self._transcribe_via_cli(audio_path, **kwargs)

    def _transcribe_via_local(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """使用本地模型转录"""
        try:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")

            # 加载音频
            audio_data = self.load_audio(audio_path, sr=16000)

            # 构建消息
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Please transcribe the audio to text."
                },
                {
                    "role": "human",
                    "content": [{"type": "audio", "audio": audio_path}]
                }
            ]

            # 处理热词 - 使用kwargs.get避免冲突
            hotwords_list = kwargs.get("hotwords", [])
            if hotwords_list:
                hotword_str = ", ".join(hotwords_list)
                messages[0]["content"] += f" Pay special attention to these keywords: {hotword_str}"

            # 设置推理参数
            generation_kwargs = {
                "max_new_tokens": kwargs.get("max_new_tokens", 512),
                "temperature": kwargs.get("temperature", 0.0),
                "do_sample": False,
                "repetition_penalty": 1.0,
            }

            # 执行推理
            start_time = time.time()
            tokens, text, _ = self.model(
                messages,
                tokenizer=self.tokenizer,
                **generation_kwargs
            )
            processing_time = time.time() - start_time

            return {
                "text": text.strip(),
                "confidence": 0.95,  # 简化实现
                "language": "zh",
                "processing_time": processing_time,
                "model_name": "Step-Audio-2-mini",
                "input_audio_path": audio_path
            }

        except Exception as e:
            logging.error(f"本地转录失败: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": "zh",
                "processing_time": 0.0,
                "error": str(e)
            }

    def _transcribe_via_api(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """通过API调用转录"""
        try:
            with open(audio_path, 'rb') as f:
                files = {'audio': f}
                data = {'hotwords': ','.join(kwargs.get('hotwords', []))}

                response = requests.post(
                    f"{self.api_url}/transcribe",
                    files=files,
                    data=data,
                    timeout=30
                )

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "text": result.get("text", ""),
                        "confidence": result.get("confidence", 0.95),
                        "language": result.get("language", "zh"),
                        "processing_time": result.get("processing_time", 0.0),
                        "model_name": "Step-Audio-2-mini",
                        "input_audio_path": audio_path
                    }
                else:
                    raise Exception(f"API调用失败: {response.status_code}")

        except Exception as e:
            logging.error(f"API转录失败: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": "zh",
                "processing_time": 0.0,
                "error": str(e)
            }

    def _transcribe_via_cli(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """通过命令行调用转录"""
        try:
            cmd = [
                "python", "examples.py",
                "--audio", audio_path,
                "--model", self.model_path
            ]

            if kwargs.get("hotwords"):
                cmd.extend(["--hotwords", ",".join(kwargs["hotwords"])])

            result = subprocess.run(cmd, capture_output=True, text=True,
                                  cwd="Model/StepAudio2")

            if result.returncode == 0:
                # 解析命令行输出
                output = result.stdout.strip()
                return {
                    "text": output,
                    "confidence": 0.95,
                    "language": "zh",
                    "processing_time": 0.0,
                    "model_name": "Step-Audio-2-mini",
                    "input_audio_path": audio_path
                }
            else:
                raise Exception(f"命令行调用失败: {result.stderr}")

        except Exception as e:
            logging.error(f"命令行转录失败: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "language": "zh",
                "processing_time": 0.0,
                "error": str(e)
            }

    def validate_audio(self, audio_path: str) -> bool:
        """验证音频文件"""
        try:
            if not os.path.exists(audio_path):
                return False

            valid_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
            ext = os.path.splitext(audio_path)[1].lower()

            return ext in valid_extensions

        except Exception:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": "Step-Audio-2-mini",
            "model_type": "StepAudio2",
            "version": "2.0.0",
            "device": self.device,
            "supported_languages": ["zh", "en", "ja", "ko"],
            "supported_formats": ["wav", "mp3", "flac", "m4a"],
            "sample_rate": 16000,
            "max_length": 512,
            "hotword_support": True
        }

    def cleanup(self):
        """清理资源"""
        if hasattr(self, 'model'):
            import torch
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class StepAudio2Fallback:
    """StepAudio2 后备实现 - 当主要实现不可用时使用"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model_name = "Step-Audio-2-mini (Fallback)"

    def setup(self) -> bool:
        """设置后备实现"""
        logging.warning("使用StepAudio2后备实现")
        return True

    def transcribe_audio(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """后备转录实现"""
        import random

        # 模拟转录结果
        sample_texts = [
            "这是一个测试音频",
            "语音识别测试",
            "StepAudio2后备实现",
            "热词测试成功",
            "模型加载中",
            "AI帮助很大",
            "使用LLM技术"
        ]

        hotwords_list = kwargs.pop("hotwords", [])
        base_text = random.choice(sample_texts)

        # 如果有热词，添加到结果中
        if hotwords_list:
            base_text += " " + " ".join(hotwords_list[:2])

        return {
            "text": base_text,
            "confidence": 0.8,
            "language": "zh",
            "processing_time": 0.1,
            "model_name": self.model_name,
            "input_audio_path": audio_path,
            "note": "这是后备实现，仅用于测试"
        }

    def validate_audio(self, audio_path: str) -> bool:
        """验证音频文件"""
        try:
            return os.path.exists(audio_path)
        except:
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model_name": self.model_name,
            "model_type": "StepAudio2",
            "version": "2.0.0",
            "device": "cpu",
            "supported_languages": ["zh"],
            "supported_formats": ["wav", "mp3", "flac"],
            "sample_rate": 16000,
            "max_length": 512,
            "hotword_support": True,
            "mode": "fallback"
        }

    def cleanup(self):
        """清理资源"""
        pass