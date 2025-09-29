"""
测试TextNormalizer的多语言文本规范化功能
"""

import pytest
from evaluation.metrics import TextNormalizer, MetricConfig


class TestTextNormalizer:
    """测试TextNormalizer类"""

    def test_chinese_normalization_basic(self):
        """测试基础中文规范化"""
        normalizer = TextNormalizer(language="zh", use_wetextprocessing=False)

        # 测试空格移除
        assert normalizer.normalize("你好 世界") == "你好世界"

        # 测试标点符号移除
        assert normalizer.normalize("你好，世界！") == "你好世界"

        # 测试英文和数字移除
        assert normalizer.normalize("hello你好123world") == "你好"

    def test_english_normalization_basic(self):
        """测试英文规范化 - 基础模式"""
        normalizer = TextNormalizer(language="en", use_nemo=False)

        # 测试小写转换
        assert normalizer.normalize("Hello World") == "hello world"

        # 测试标点符号移除
        assert normalizer.normalize("Hello, World!") == "hello world"

        # 测试空格标准化
        assert normalizer.normalize("Hello   World") == "hello world"

    def test_english_normalization_nemo(self):
        """测试英文规范化 - NeMo模式"""
        # 注意：此测试需要NeMo-text-processing安装
        normalizer = TextNormalizer(language="en", use_nemo=True)

        # 基础测试
        result = normalizer.normalize("Hello, World!")
        assert "hello" in result
        assert "world" in result

    def test_japanese_normalization(self):
        """测试日文规范化"""
        normalizer = TextNormalizer(language="ja")

        # 测试全角转半角
        text = "こんにちは！"
        result = normalizer.normalize(text)
        assert "こんにちは" in result

        # 测试空格标准化
        assert normalizer.normalize("こんにちは　世界") == "こんにちは 世界"

    def test_empty_text(self):
        """测试空文本处理"""
        normalizer = TextNormalizer(language="zh")

        assert normalizer.normalize("") == ""
        assert normalizer.normalize(None) == ""
        assert normalizer.normalize("   ") == ""

    def test_language_switching(self):
        """测试语言切换"""
        normalizer = TextNormalizer(language="en")

        # 初始为英文
        assert normalizer.normalize("Hello World") == "hello world"

        # 切换到中文
        normalizer.set_language("zh")
        assert normalizer.normalize("你好 世界") == "你好世界"

        # 切换到日文
        normalizer.set_language("ja")
        result = normalizer.normalize("こんにちは")
        assert "こんにちは" in result

    def test_supported_languages(self):
        """测试支持的语言列表"""
        normalizer = TextNormalizer()
        supported = normalizer.get_supported_languages()

        assert "zh" in supported
        assert "en" in supported
        assert "ja" in supported

    def test_generic_normalization(self):
        """测试通用规范化"""
        normalizer = TextNormalizer(language="unknown")

        # 应该使用通用规范化
        assert normalizer.normalize("Hello, World!") == "hello world"

    def test_config_integration(self):
        """测试与MetricConfig的集成"""
        config = MetricConfig(language="en")
        normalizer = TextNormalizer(language=config.language)

        assert normalizer.normalize("Hello World") == "hello world"

    @pytest.mark.parametrize("language,text,expected_contains", [
        ("zh", "你好，世界！", "你好世界"),
        ("en", "Hello, World!", "hello world"),
        ("ja", "こんにちは、世界！", "こんにちは"),
    ])
    def test_multi_language_normalization(self, language, text, expected_contains):
        """测试多语言规范化参数化"""
        normalizer = TextNormalizer(language=language)
        result = normalizer.normalize(text)

        # 对于中文，检查主要字符是否存在
        if language == "zh":
            assert "你好" in result or "世界" in result
        elif language == "en":
            assert "hello" in result and "world" in result
        elif language == "ja":
            assert "こんにちは" in result

    def test_wetextprocessing_fallback(self):
        """测试WeTextProcessing不可用时回退到基础规范化"""
        # 模拟WeTextProcessing不可用的情况
        normalizer = TextNormalizer(language="zh", use_wetextprocessing=True)

        # 强制回退到基础规范化
        normalizer.use_wetextprocessing = False
        normalizer._chinese_normalizer = None

        result = normalizer.normalize("你好，世界！")
        assert "你好世界" in result


def test_text_normalizer_integration():
    """测试TextNormalizer在实际场景中的集成使用"""
    from core.models import TestResult

    # 创建测试数据
    results = [
        TestResult(
            audio_path="test.wav",
            reference_text="你好，世界！",
            predicted_text="你好世界",
            processing_time=1.0,
            confidence_score=0.95
        ),
        TestResult(
            audio_path="test2.wav",
            reference_text="Hello, World!",
            predicted_text="hello world",
            processing_time=1.2,
            confidence_score=0.92
        )
    ]

    # 测试中文规范化
    zh_normalizer = TextNormalizer(language="zh")
    assert zh_normalizer.normalize(results[0].reference_text) == "你好世界"

    # 测试英文规范化
    en_normalizer = TextNormalizer(language="en")
    assert en_normalizer.normalize(results[1].reference_text) == "hello world"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])