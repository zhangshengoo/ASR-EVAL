"""
文本规范化模块 - 优化版
支持多语言处理，直接报错，无异常捕捉
"""

from typing import Optional, Dict, Any
import re
from langdetect import detect
from jiwer.transforms import Compose, ToLowerCase, RemovePunctuation


class TextNormalizer:
    """多语言文本规范化器 - 精简版"""

    def __init__(self, language: Optional[str] = None):
        self.language = language.lower() if language else None
        self._chinese_normalizer = None
        self._english_normalizer = None
        self._japanese_normalizer = None
        self._japanese_processor = None
        self._jiwer_transforms = None

        if self.language:
            self._initialize_normalizers()
        self._init_jiwer_transforms()

    def _init_jiwer_transforms(self):
        """初始化jiwer transforms用于中英文处理"""
        self._jiwer_transforms = Compose([
            ToLowerCase(),
            RemovePunctuation()
        ])

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
        """自动检测文本语言 - 使用langdetect库，仅支持中英日三种语言"""
        if not text.strip():
            return 'generic'

        # 使用langdetect进行语言检测
        detected_lang = detect(text)

        # 仅支持中英日三种语言，其他统一归为generic
        if detected_lang in ['zh', 'zh-cn', 'zh-tw']:
            return 'zh'
        elif detected_lang == 'en':
            return 'en'
        elif detected_lang == 'ja':
            return 'ja'
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
        """中文文本规范化 - 集成jiwer transforms"""
        text = text.strip()
        if not text:
            return ""

        # 使用WeTextProcessing进行TN
        if self._chinese_normalizer:
            text = self._chinese_normalizer['tn'].normalize(text)

        # 基础清理
        text = self._basic_chinese_clean(text)

        # 应用jiwer transforms - 中文也适用部分规则
        if self._jiwer_transforms:
            text = self._jiwer_transforms(text)

        return text

    def _normalize_english(self, text: str) -> str:
        """英文文本规范化 - 集成jiwer transforms"""
        text = text.strip()
        if not text:
            return ""

        # 使用NVIDIA NeMo进行TN
        if self._english_normalizer:
            text = self._english_normalizer['tn'].normalize(text, verbose=False)

        # 应用jiwer transforms - 英文处理的核心
        if self._jiwer_transforms:
            text = self._jiwer_transforms(text)

        # 基础清理（jiwer已处理部分，但保留额外清理）
        text = self._basic_english_clean(text)
        return text

    def _normalize_japanese(self, text: str) -> str:
        """日文文本规范化 - 优化版"""
        text = text.strip()
        if not text:
            return ""

        # 第一步：使用NVIDIA NeMo进行TN
        if self._japanese_normalizer:
            text = self._japanese_normalizer['tn'].normalize(text, verbose=False)

        # 第二步：提取英文单词，检查词典或转换为片假名，再转为平假名
        import jaconv
        import pykakasi

        # 英文词典（暂时为空，所有英文单词都使用g2p转换）
        english_dict = {}

        # 处理英文单词
        def process_english_word(match):
            word = match.group(0).lower()
            if word in english_dict:
                # 词典中有，直接使用片假名
                katakana = english_dict[word]
            else:
                # 词典中没有，使用g2p转换为片假名
                if self._japanese_processor:
                    katakana = self._japanese_processor['g2p'](word, kana=True)
                else:
                    katakana = word
            # 片假名转平假名
            return jaconv.kata2hira(katakana)

        text = re.sub(r'[a-zA-Z]+', process_english_word, text)

        # 第三步：所有汉字转换为平假名
        kakasi = pykakasi.kakasi()
        kakasi.setMode('J', 'H')  # 汉字->平假名
        kakasi.setMode('K', 'H')  # 片假名->平假名
        kakasi.setMode('H', 'H')  # 平假名->平假名
        conv = kakasi.getConverter()
        text = conv.do(text)

        # 第四步：移除所有非平假名字符（主要是标点符号和语气词）
        text = re.sub(r'[^\u3040-\u309F\s]', '', text)

        # 第五步：转换为小写
        text = text.lower()

        # 标准化空格
        text = ' '.join(text.split())
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
        """基础中文清理 - jiwer处理后简化清理"""
        # jiwer已处理标点，这里主要处理中文特有字符
        # 移除空格（中文通常不需要）
        text = text.replace(' ', '')
        # 移除数字（如果需要保留数字可注释此行）
        text = re.sub(r'[0-9]', '', text)
        return text

    def _basic_english_clean(self, text: str) -> str:
        """基础英文清理 - jiwer处理后简化清理"""
        # jiwer已处理小写转换和标点移除，这里主要做空格标准化
        # 标准化空格
        text = ' '.join(text.split())
        # 确保文本干净（jiwer可能未处理的边界情况）
        text = text.strip()
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
        self._init_jiwer_transforms()

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