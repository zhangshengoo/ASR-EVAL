# Kaldi文本对齐工具集成指南

本文档介绍如何在ASR-Eval框架中使用Kaldi的文本对齐工具进行详细的文本比较和分析。

## 功能概述

- **Kaldi文本对齐**: 使用Kaldi的`align-text`工具进行精确的文本对齐
- **多语言支持**: 支持中文、英文、日文的文本规范化和对齐
- **差异可视化**: 提供彩色差异显示和并排对比
- **详细统计**: 提供词级错误统计和分析报告

## 核心组件

### 1. TextAligner (evaluation/text_alignment.py)

Kaldi文本对齐工具的Python封装，提供以下功能：

- **文本对齐**: 使用Kaldi的`align-text`工具进行逐词对齐
- **错误统计**: 计算替换、删除、插入错误
- **格式化输出**: 生成易读的对齐结果表格
- **备用方案**: 当Kaldi不可用时使用简单的逐词对齐

#### 使用示例

```python
from evaluation.text_alignment import TextAligner

aligner = TextAligner(kaldi_path="/path/to/kaldi")
result = aligner.generate_diff_report("你好世界", "你好世界")
print(result['formatted_output'])
print("错误统计:", result['statistics'])
```

### 2. TextComparator (evaluation/text_comparison.py)

综合文本比较器，集成文本规范化和Kaldi对齐：

- **文本规范化**: 自动应用语言特定的文本规范化
- **差异分析**: 结合规范化和对齐结果
- **可视化**: 提供多种可视化格式
- **批量处理**: 支持批量文本比较

#### 使用示例

```python
from evaluation.text_comparison import TextComparator

comparator = TextComparator(language="zh")
result = comparator.compare_texts("今天天气很好", "今天天气很好")
print(comparator.format_comparison_report(result))
```

### 3. ASRTextEvaluator (evaluation/text_comparison.py)

专门用于ASR结果评估的类：

- **单个评估**: 评估单个ASR结果
- **批量评估**: 批量评估多个ASR结果
- **综合统计**: 提供WER、准确率等统计指标

#### 使用示例

```python
from evaluation.text_comparison import ASRTextEvaluator

evaluator = ASRTextEvaluator(language="zh")
asr_results = [
    {"reference_text": "你好", "asr_text": "你好"},
    {"reference_text": "世界", "asr_text": "地球"}
]
batch_result = evaluator.evaluate_asr_batch(asr_results)
```

## 安装和使用

### 1. 安装Kaldi（可选）

要获得最佳文本对齐效果，建议安装Kaldi：

```bash
# 克隆Kaldi仓库
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi

# 编译Kaldi
./extras/install_mkl.sh
make -j 8
```

### 2. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# 中文文本处理（可选）
pip install -r requirements-wetextprocessing.txt

# NVIDIA NeMo文本处理（可选）
pip install -r requirements-nemo.txt

# 日文文本处理（可选）
pip install -r requirements-japanese.txt
```

### 3. 运行示例

```bash
# 运行示例脚本
python examples/text_alignment_example.py

# 运行测试
pytest tests/test_text_alignment.py -v
```

## 使用场景

### 1. 简单的文本对齐

```python
from evaluation.text_alignment import TextAligner

aligner = TextAligner()
result = aligner.align_text("hello world", "hello word")
for ref, hyp in result:
    print(f"{ref} → {hyp}")
```

### 2. 带规范化的文本比较

```python
from evaluation.text_comparison import TextComparator

comparator = TextComparator(language="zh", use_wetextprocessing=True)
comparison = comparator.compare_texts(
    "我今天花了123.45元",
    "我今天花了123块4毛5",
    normalize=True
)
print(comparison['alignment']['formatted_output'])
```

### 3. 批量ASR评估

```python
from evaluation.text_comparison import ASRTextEvaluator

evaluator = ASRTextEvaluator(language="en")

# 准备ASR结果
results = [
    {
        "reference_text": "The weather is nice today",
        "asr_text": "The weather is nice today",
        "audio_path": "audio1.wav"
    },
    {
        "reference_text": "I want to go shopping",
        "asr_text": "I want to go shop",
        "audio_path": "audio2.wav"
    }
]

# 批量评估
batch_summary = evaluator.evaluate_asr_batch(results)
print(f"总体WER: {batch_summary['overall_wer']:.2%}")
```

## 输出格式

### 1. 对齐结果格式

```
=======================================================
Reference | Hypothesis | Status
=======================================================
hello     | hello      | ✓
world     | word       | ✗
today     |            | ✗
          | extra      | ✗
=======================================================
```

### 2. 错误统计

```python
{
    "total": 4,
    "correct": 1,
    "substitutions": 1,
    "deletions": 1,
    "insertions": 1
}
```

### 3. 彩色差异显示

- 红色：删除的文本
- 绿色：插入的文本
- 正常：匹配的文本

## 注意事项

1. **Kaldi依赖**: 如果没有安装Kaldi，系统会自动使用备用对齐方法
2. **语言支持**: 确保选择正确的语言类型以获得最佳规范化效果
3. **性能**: 大批量文本处理时建议使用批量接口
4. **内存**: 处理大量文本时注意内存使用

## 故障排除

### 常见问题

1. **Kaldi不可用**
   - 检查Kaldi安装路径是否正确
   - 确保`align-text`命令在PATH中

2. **文本规范化失败**
   - 检查语言设置是否正确
   - 确保安装了相应的文本处理依赖

3. **内存问题**
   - 对于大量文本，使用批量处理接口
   - 考虑分批处理数据

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from evaluation.text_alignment import TextAligner
aligner = TextAligner()
print("Kaldi可用:", aligner._check_kaldi_available())
```

## 扩展开发

### 添加新语言支持

1. 在`TextNormalizer`中添加新语言处理
2. 在`TextComparator`中更新语言配置
3. 添加相应的依赖包要求

### 自定义对齐算法

可以继承`TextAligner`类并重写相关方法：

```python
class CustomAligner(TextAligner):
    def _fallback_alignment(self, reference, hypothesis):
        # 实现自定义对齐算法
        pass
```