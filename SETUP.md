# StepAudio2 测试系统设置指南

## 环境要求

- Python 3.10+
- PyTorch 2.3+ with CUDA支持
- 音频处理库

## 安装步骤

### 1. 安装依赖

```bash
pip install torch torchaudio librosa
pip install transformers==4.49.0 s3tokenizer hyperpyyaml
pip install gitpython  # 用于管理submodule
```

### 2. 设置StepAudio2子模块

```bash
# 如果尚未初始化子模块
git submodule init
git submodule update

# 或者手动克隆StepAudio2
cd Model
git clone https://github.com/stepfun-ai/Step-Audio2.git StepAudio2
cd StepAudio2
git lfs install
git clone https://huggingface.co/stepfun-ai/Step-Audio-2-mini
```

### 3. 验证模型文件

确保以下文件存在：
- `Model/StepAudio2/Step-Audio-2-mini/config.json`
- `Model/StepAudio2/Step-Audio-2-mini/pytorch_model.bin`

### 4. 创建测试数据

```bash
# 运行测试脚本创建示例数据
python test_step_audio2.py
```

### 5. 运行测试

```bash
# 运行完整测试
python main.py --model step_audio2 --dataset hotword --max-samples 10

# 或者运行简单的功能测试
python test_step_audio2.py
```

## 热词测试数据格式

数据目录结构：
```
datasets/热词测试/
├── 测试场景/
│   ├── 测试场景.list
│   ├── test1.wav
│   ├── test2.wav
│   └── ...
└── 其他场景/
    └── ...
```

`.list` 文件格式：
```
filename | reference_text | hotword1,hotword2,hotword3
```

## 配置说明

配置文件位置：
- `config/models/step_audio2.json` - StepAudio2模型配置
- 支持热词、置信度阈值、音频格式等参数

## 故障排除

### 模型加载失败
- 检查模型文件是否完整
- 验证CUDA环境
- 检查Python版本兼容性

### 音频文件问题
- 支持的格式：wav, mp3, flac, m4a, ogg
- 采样率：16kHz
- 时长范围：0.5-300秒

### 热词不生效
- 检查热词配置是否启用
- 验证置信度阈值设置
- 确保热词长度不超过20个字符