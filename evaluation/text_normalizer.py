"""
文本规范化模块 - 优化版
支持多语言处理，直接报错，无异常捕捉
"""

from typing import Optional, Dict, Any
import re


class TextNormalizer:
    """多语言文本规范化器 - 精简版"""

    def __init__(self, language: Optional[str] = None):
        self.language = language.lower() if language else None
        self._chinese_normalizer = None
        self._english_normalizer = None
        self._japanese_normalizer = None
        self._japanese_processor = None

        if self.language:
            self._initialize_normalizers()

    def _initialize_normalizers(self):
        """根据当前语言初始化对应的规范化器"""
        if self.language == "zh":
            self._init_chinese_normalizer()
        elif self.language == "en":
            self._init_english_normalizer()
        elif self.language in ["ja", "jp"]:
            self._init_japanese_normalizer()
            self._init_japanese_processor()

    def _detect_language(self, text: str) -> str:
        """自动检测文本语言 - 简化版"""
        if re.search(r'[\u4e00-\u9fff]', text):
            return 'zh'
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return 'ja'
        elif re.search(r'[a-zA-Z]', text):
            return 'en'
        else:
            return 'generic'

    def _init_chinese_normalizer(self):
        """初始化WeTextProcessing中文规范化器"""
        from itn.chinese.inverse_normalizer import InverseNormalizer
        from tn.chinese.normalizer import Normalizer

        self._chinese_normalizer = {
            'tn': Normalizer(),
            'itn': InverseNormalizer()
        }

    def _init_english_normalizer(self):
        """初始化NVIDIA NeMo英文规范化器"""
        from nemo_text_processing.text_normalization.normalize import Normalizer as NeMoNormalizer
        from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer as NeMoInverseNormalizer

        self._english_normalizer = {
            'tn': NeMoNormalizer(input_case='cased', lang='en'),
            'itn': NeMoInverseNormalizer(lang='en')
        }

    def _init_japanese_normalizer(self):
        """初始化NVIDIA NeMo日文规范化器"""
        from nemo_text_processing.text_normalization.normalize import Normalizer as NeMoNormalizer
        from nemo_text_processing.inverse_text_normalization.inverse_normalize import InverseNormalizer as NeMoInverseNormalizer

        self._japanese_normalizer = {
            'tn': NeMoNormalizer(input_case='cased', lang='ja'),
            'itn': NeMoInverseNormalizer(lang='ja')
        }

    def _init_japanese_processor(self):
        """初始化日文处理器"""
        from pyopenjtalk_plus import g2p
        import MeCab
        self._japanese_processor = {
            'g2p': g2p,
            'mecab': MeCab.Tagger('-Owakati')
        }

    def normalize(self, text: str) -> str:
        """规范化文本，根据语言选择不同策略"""
        if not text:
            return ""

        # 如果未指定语言，自动检测
        if self.language is None:
            self.language = self._detect_language(text)
            self._initialize_normalizers()

        # 根据语言选择规范化策略
        if self.language == "zh":
            return self._normalize_chinese(text)
        elif self.language == "en":
            return self._normalize_english(text)
        elif self.language in ["ja", "jp"]:
            return self._normalize_japanese(text)
        else:
            return self._normalize_generic(text)

    def _normalize_chinese(self, text: str) -> str:
        """中文文本规范化"""
        text = text.strip()
        if not text:
            return ""

        # 使用WeTextProcessing进行TN
        if self._chinese_normalizer:
            text = self._chinese_normalizer['tn'].normalize(text)

        # 基础清理
        text = self._basic_chinese_clean(text)
        return text

    def _normalize_english(self, text: str) -> str:
        """英文文本规范化"""
        text = text.strip()
        if not text:
            return ""

        # 使用NVIDIA NeMo进行TN
        if self._english_normalizer:
            text = self._english_normalizer['tn'].normalize(text, verbose=False)

        # 基础清理
        text = self._basic_english_clean(text)
        return text

    def _normalize_japanese(self, text: str) -> str:
        """日文文本规范化"""
        text = text.strip()
        if not text:
            return ""

        # 使用NVIDIA NeMo进行TN
        if self._japanese_normalizer:
            text = self._japanese_normalizer['tn'].normalize(text, verbose=False)

        # 日文特殊处理
        text = self._process_japanese_text(text)
        return text

    def _normalize_generic(self, text: str) -> str:
        """通用文本规范化"""
        text = text.strip()
        if not text:
            return ""

        # 转换为小写
        text = text.lower()
        # 移除标点符号
        text = re.sub(r'[^\w\s]', '', text)
        # 标准化空格
        text = ' '.join(text.split())
        return text

    def _basic_chinese_clean(self, text: str) -> str:
        """基础中文清理"""
        # 移除空格
        text = text.replace(' ', '')
        # 移除英文和标点
        text = re.sub(r'[a-zA-Z0-9\W]', '', text)
        return text

    def _basic_english_clean(self, text: str) -> str:
        """基础英文清理"""
        # 转换为小写
        text = text.lower()
        # 标准化空格
        text = ' '.join(text.split())
        # 移除特殊字符但保留字母数字
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def _process_japanese_text(self, text: str) -> str:
        """日文特殊处理"""
        # 英文转片假名
        text = self._convert_english_to_katakana(text)
        # 汉字转平假名
        text = self._convert_kanji_to_hiragana(text)
        # 移除连接符号
        text = re.sub(r'[-・ー]', '', text)
        # 标准化空格
        text = ' '.join(text.split())
        return text

    def _convert_english_to_katakana(self, text: str) -> str:
        """英文转片假名"""
        if not self._japanese_processor:
            return text

        from pykakasi import kakasi
        kakasi_instance = kakasi()
        kakasi_instance.setMode('H', 'K')  # 平假名->片假名
        kakasi_instance.setMode('K', 'K')  # 片假名->片假名
        kakasi_instance.setMode('J', 'K')  # 汉字->片假名
        kakasi_instance.setMode('a', 'K')  # 英文->片假名
        conv = kakasi_instance.getConverter()

        # 使用正则表达式找到英文单词并转换
        def replace_english(match):
            word = match.group(0)
            katakana = self._japanese_processor['g2p'](word, kana=True)
            return katakana

        return re.sub(r'[a-zA-Z]+', replace_english, text)

    def _convert_kanji_to_hiragana(self, text: str) -> str:
        """汉字转平假名"""
        if not self._japanese_processor:
            return text

        from pykakasi import kakasi
        kakasi_instance = kakasi()
        kakasi_instance.setMode('H', 'H')  # 平假名->平假名
        kakasi_instance.setMode('K', 'H')  # 片假名->平假名
        kakasi_instance.setMode('J', 'H')  # 汉字->平假名
        conv = kakasi_instance.getConverter()
        return conv.do(text)

    def set_language(self, language: Optional[str] = None):
        """设置语言，如果为None则启用自动检测"""
        if language is None:
            self.language = None
            return

        self.language = language.lower()
        self._initialize_normalizers()

    def get_supported_languages(self) -> list[str]:
        """获取支持的语言列表"""
        return ["zh", "en", "ja", "jp"]

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return {
            "language": self.language,
            "auto_detect": self.language is None,
            "supported_languages": self.get_supported_languages()
        }