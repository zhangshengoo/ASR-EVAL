# 以StepAudio2 集成说明

## 子模块设置

要正确设置StepAudio2子模块，请执行以下命令：

```bash
git init
git submodule add https://github.com/stepfun-ai/Step-Audio2.git Model/StepAudio2
git submodule update --init --recursive
```

## 热词测试集成

本目录包含StepAudio2模型的热词测试专用实现，支持以下功能：

1. **热词识别测试**: 针对特定词汇的识别准确性测试
2. **场景化测试**: 支持不同场景下的热词识别验证
3. **性能评估**: 计算热词识别相关的特殊指标

## 文件结构

```
StepAudio2/
├── step_audio_model.py     # StepAudio2模型封装
├── hotword_tester.py       # 热词测试专用类
├── config.json            # 模型配置文件
└── requirements.txt       # 额外依赖
```

## 使用说明

### 1. 初始化子模块
```bash
# 在项目根目录执行
git submodule add https://github.com/stepfun-ai/Step-Audio2.git Model/StepAudio2
```

### 2. 安装依赖
```bash
cd Model/StepAudio2
pip install -r requirements.txt
```

### 3. 运行热词测试
```bash
# 使用测试框架运行热词测试
python main.py --model step_audio2 --dataset hotword --max-samples 50
```