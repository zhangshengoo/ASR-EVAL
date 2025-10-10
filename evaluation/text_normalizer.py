"""
文本规范化模块 - 支持多语言的高级文本处理
支持中文(WeTextProcessing)、英文(NVIDIA NeMo)、日文(NVIDIA NeMo + pyopenjtalk-plus)
"""

from typing import Optional, Dict, Any
import re
from langdetect import detect, LangDetectError


class TextNormalizer:
    """多语言文本规范化器"""

    def __init__(self, language: Optional[str] = None, use_wetextprocessing: bool = True, use_nemo: bool = True):
        self.use_wetextprocessing = use_wetextprocessing
        self.use_nemo = use_nemo
        self._chinese_normalizer = None
        self._english_normalizer = None
        self._japanese_normalizer = None
        self._japanese_processor = None

        # 如果未指定语言，初始化为None，将在normalize时自动检测
        if language:
            self.language = language.lower()
            self._initialize_normalizers()
        else:
            self.language = None

    def _initialize_normalizers(self):
        """根据当前语言初始化对应的规范化器"""
        self.use_wetextprocessing = self.use_wetextprocessing and self.language == "zh"
        self.use_nemo = self.use_nemo and self.language in ["en", "ja", "jp"]

        if self.use_wetextprocessing:
            self._init_chinese_normalizer()
        if self.use_nemo:
            if self.language == "en":
                self._init_english_normalizer()
            elif self.language in ["ja", "jp"]:
                self._init_japanese_normalizer()
                self._init_japanese_processor()

    def _detect_language(self, text: str) -> str:
        """自动检测文本语言"""
        try:
            detected_lang = detect(text)
            # 映射检测结果到支持的语言代码
            lang_map = {
                'zh': 'zh', 'zh-cn': 'zh', 'zh-tw': 'zh',
                'en': 'en',
                'ja': 'ja', 'jp': 'ja'
            }
            return lang_map.get(detected_lang, 'generic')
        except LangDetectError:
            # 如果检测失败，根据字符特征进行简单判断
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
        try:
            from pyopenjtalk_plus import g2p
            import MeCab
            self._japanese_processor = {
                'g2p': g2p,
                'mecab': MeCab.Tagger('-Owakati')
            }
        except ImportError:
            self._japanese_processor = None

    def normalize(self, text: str) -> str:
        """规范化文本，根据语言选择不同策略"""
        if not text:
            return ""

        # 如果未指定语言，自动检测
        if self.language is None:
            detected_lang = self._detect_language(text)
            self.language = detected_lang
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
        if self.use_wetextprocessing and self._chinese_normalizer:
            text = self._chinese_normalizer['tn'].normalize(text)

        # 基础清理
        text = self._basic_chinese_clean(text)

        return text

    def _normalize_english(self, text: str) -> str:
        """英文文本规范化 - 使用NVIDIA NeMo"""
        text = text.strip()
        if not text:
            return ""

        # 使用NVIDIA NeMo进行TN
        if self.use_nemo and self._english_normalizer:
            text = self._english_normalizer['tn'].normalize(text, verbose=False)

        # 基础清理
        text = self._basic_english_clean(text)

        return text

    def _normalize_japanese(self, text: str) -> str:
        """日文文本规范化 - 使用NVIDIA NeMo + pyopenjtalk-plus"""
        text = text.strip()
        if not text:
            return ""

        # 使用NVIDIA NeMo进行TN
        if self.use_nemo and self._japanese_normalizer:
            text = self._japanese_normalizer['tn'].normalize(text, verbose=False)

        # 日文特殊处理
        text = self._process_japanese_text(text)

        return text

    def _normalize_generic(self, text: str) -> str:
        """通用文本规范化"""
        import re

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
        import re

        # 移除空格
        text = text.replace(' ', '')

        # 移除英文和标点
        text = re.sub(r'[a-zA-Z0-9\W]', '', text)

        return text

    def _basic_english_clean(self, text: str) -> str:
        """基础英文清理"""
        import re

        # 转换为小写
        text = text.lower()

        # 标准化空格
        text = ' '.join(text.split())

        # 移除特殊字符但保留字母数字
        text = re.sub(r'[^\w\s]', '', text)

        return text

    def _process_japanese_text(self, text: str) -> str:
        """日文特殊处理"""
        import re

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

        try:
            import pykakasi
            kakasi = pykakasi.kakasi()
            kakasi.setMode('H', 'K')  # 平假名->片假名
            kakasi.setMode('K', 'K')  # 片假名->片假名
            kakasi.setMode('J', 'K')  # 汉字->片假名
            kakasi.setMode('a', 'K')  # 英文->片假名
            conv = kakasi.getConverter()

            # 使用正则表达式找到英文单词并转换
            def replace_english(match):
                word = match.group(0)
                try:
                    # 使用g2p将英文转为片假名
                    if self._japanese_processor and 'g2p' in self._japanese_processor:
                        katakana = self._japanese_processor['g2p'](word, kana=True)
                        return katakana
                    else:
                        return conv.do(word)
                except:
                    return conv.do(word)

            # 替换英文单词
            text = re.sub(r'[a-zA-Z]+', replace_english, text)

        except ImportError:
            # 如果pykakasi不可用，使用基础转换
            pass

        return text

    def _convert_kanji_to_hiragana(self, text: str) -> str:
        """汉字转平假名"""
        if not self._japanese_processor:
            return text

        try:
            import pykakasi
            kakasi = pykakasi.kakasi()
            kakasi.setMode('H', 'H')  # 平假名->平假名
            kakasi.setMode('K', 'H')  # 片假名->平假名
            kakasi.setMode('J', 'H')  # 汉字->平假名
            conv = kakasi.getConverter()
            return conv.do(text)
        except ImportError:
            # 如果pykakasi不可用，返回原文
            return text

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
            "use_wetextprocessing": self.use_wetextprocessing,
            "use_nemo": self.use_nemo,
            "supported_languages": self.get_supported_languages(),
            "langdetect_available": True
        }