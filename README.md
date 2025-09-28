# ASR模型测试系统

一个开源的ASR（自动语音识别）模型测试和评估框架，支持多种开源ASR模型的统一测试和对比。

## 系统特性

- **多模型支持**: 支持KimiAudio、FireRedASR、StepAudio2等多种开源ASR模型
- **统一接口**: 提供统一的模型接口和测试流程
- **丰富测试集**: 包含回归测试、噪音测试、长音频测试、方言测试等多种测试数据集
- **全面指标**: 计算WER、CER、SER、实时因子等多种评估指标
- **可视化报告**: 生成JSON、CSV、HTML等多种格式的测试报告

## 项目结构

```
ASR-Eval/
├── core/                    # 核心模块
│   ├── enums.py            # 枚举定义
│   └── models.py           # 数据模型
├── models/                 # 模型实现
│   ├── base.py             # 模型基类
│   ├── kimi_audio.py       # KimiAudio模型实现
│   ├── fire_red_asr.py     # FireRedASR模型实现
│   ├── step_audio2.py      # StepAudio2模型实现
│   └── factory.py          # 模型工厂
├── datasets/               # 数据集管理
│   └── manager.py          # 数据集管理器
├── evaluation/             # 评估模块
│   └── metrics.py          # 指标计算
├── testing/                # 测试框架
│   └── framework.py        # 测试框架主类
├── utils/                  # 工具模块
│   ├── config_loader.py    # 配置加载器
│   └── reporter.py         # 报告生成器
├── config/                 # 配置文件
│   └── config.json         # 主配置文件
├── Model/                  # 模型目录（Git子模块）
│   ├── KimiAudio/         # KimiAudio模型子模块
│   ├── FireRedASR/        # FireRedASR模型子模块
│   └── StepAudio2/        # StepAudio2模型子模块
├── datasets/               # 测试数据集
├── results/                # 测试结果输出
├── main.py                 # 主入口
└── requirements.txt        # 依赖列表
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 初始化模型子模块

```bash
# 添加KimiAudio子模块（示例）
git submodule add https://github.com/KimiAudio/KimiAudio.git Model/KimiAudio

# 添加FireRedASR子模块（示例）
git submodule add https://github.com/FireRedASR/FireRedASR.git Model/FireRedASR

# 添加StepAudio2子模块（示例）
git submodule add https://github.com/StepAudio/StepAudio2.git Model/StepAudio2

# 更新子模块
git submodule update --init --recursive
```

### 3. 创建示例数据集

```bash
python main.py --create-sample-data
```

### 4. 运行测试

#### 单个模型测试
```bash
# 测试KimiAudio在回归测试集上的表现
python main.py --model kimi_audio --dataset regression

# 测试多个模型
python main.py --model kimi_audio --model fire_red_asr --dataset regression --max-samples 100
```

#### 对比测试
```bash
# 运行所有模型的对比测试
python main.py --compare --dataset regression --max-samples 50

# 指定输出目录
python main.py --compare --dataset regression --output results/comparison_2024
```

#### 批量测试
```bash
# 使用测试计划文件
python main.py --batch test_plan.json
```

## 配置文件

系统使用JSON格式的配置文件，主要配置在 `config/config.json`：

```json
{
  "models": {
    "kimi_audio": {
      "enabled": true,
      "model_path": "Model/KimiAudio",
      "device": "cuda"
    }
  },
  "datasets": {
    "regression": {
      "path": "datasets/regression_test"
    }
  },
  "evaluation": {
    "calculate_wer": true,
    "calculate_cer": true
  }
}
```

## 支持的测试数据集

1. **regression**: 标准回归测试集
2. **noise**: 噪音环境测试集
3. **long_audio**: 长音频测试集（>30秒）
4. **accent**: 方言口音测试集
5. **multilingual**: 多语言测试集

## 评估指标

- **WER (Word Error Rate)**: 词错误率
- **CER (Character Error Rate)**: 字符错误率
- **SER (Sentence Error Rate)**: 句错误率
- **RTF (Real-time Factor)**: 实时因子
- **Confidence**: 置信度
- **Processing Time**: 处理时间

## 命令行参数

```bash
python main.py [选项]

选项:
  -h, --help            显示帮助信息
  -c, --config CONFIG   配置文件路径 (默认: config/config.json)
  -m, --model MODEL     指定测试的模型类型 (可多次使用)
  -d, --dataset DATASET 指定测试的数据集类型 (可多次使用)
  -n, --max-samples N   最大测试样本数
  -o, --output DIR      输出目录
  --compare             运行对比测试
  --batch FILE          批量测试计划文件
  --create-sample-data  创建示例数据集
```

## 测试计划示例

创建 `test_plan.json`：

```json
{
  "test_cases": [
    {
      "models": ["kimi_audio", "fire_red_asr"],
      "datasets": ["regression", "noise"],
      "max_samples": 100
    },
    {
      "models": ["step_audio2"],
      "datasets": ["long_audio"],
      "max_samples": 50
    }
  ]
}
```

## 添加新模型

1. 创建新的模型类继承自 `BaseASRModel`
2. 在 `models/factory.py` 中注册新模型
3. 更新配置文件

## 开发指南

### 添加自定义指标

在 `evaluation/metrics.py` 中自定义指标计算类。

### 添加新的测试数据集

在 `datasets/manager.py` 中添加新的数据集类型。

### 扩展报告格式

在 `utils/reporter.py` 中添加新的报告格式。

## 注意事项

1. 确保模型子模块已正确初始化和更新
2. 检查模型所需的依赖是否已安装
3. 测试前确保数据集路径正确且包含音频文件和对应的转录文本
4. 大模型测试可能需要较多内存，建议在GPU环境下运行

## 许可证

MIT License