# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
ASR evaluation framework for comparing models (FireRedASR, KimiAudio, StepAudio2) with WER/CER/SER/RTF metrics and Chinese language support.

## High-Level Architecture

### Core Flow
**Model Factory** → **Dataset Manager** → **Test Framework** → **Metrics Engine** → **Report Generator**

### Key Design Patterns
- **Factory Pattern**: Model creation in `models/factory.py`
- **Strategy Pattern**: Different metrics calculators in `evaluation/`
- **Template Method**: BaseASRModel defines interface, models implement specifics

### Critical Integration Points
- **models/base.py**: All ASR models must inherit from BaseASRModel
- **core/enums.py**: Add new model/dataset types here
- **config/config.json**: Central configuration for all components

## Essential Commands

### Development Workflow
```bash
# Quick test with specific model
cd /Users/zhangsheng/code/ASR-Eval && python main.py --model step_audio2 --dataset regression --max-samples 10

# Comparative testing between models
cd /Users/zhangsheng/code/ASR-Eval && python main.py --compare --dataset regression --max-samples 50

# Batch testing with plan file
cd /Users/zhangsheng/code/ASR-Eval && python main.py --batch test_plan.json

# Run specific test
cd /Users/zhangsheng/code/ASR-Eval && pytest tests/test_hotword_metrics.py::test_basic_hotword_detection -v

# Run all tests
cd /Users/zhangsheng/code/ASR-Eval && pytest tests/ -v

# Test text alignment
cd /Users/zhangsheng/code/ASR-Eval && python -c "from evaluation.text_alignment import TextAligner; print(TextAligner().generate_diff_report('测试文本', '测试文件'))"

# Hotword testing with configurable library
cd /Users/zhangsheng/code/ASR-Eval && python script/configurable_hotword.py datasets/热词测试/场景1 "3,5,10"
```

### Model Development
```bash
# Add new model (example)
cd /Users/zhangsheng/code/ASR-Eval && git submodule add https://github.com/example/model.git Model/NewModel

# Test model integration
cd /Users/zhangsheng/code/ASR-Eval && python -c "from models.factory import ModelFactory; factory = ModelFactory(); model = factory.create_model('new_model', {}); print('Model loaded:', model is not None)"

# Check available models
cd /Users/zhangsheng/code/ASR-Eval && python -c "from models.factory import ModelFactory; f = ModelFactory(); print(f.get_available_models())"
```

## Key Technical Decisions

### Text Processing Pipeline
1. **Kaldi align-text**: Primary alignment algorithm for Chinese/English
2. **Priority-based edit distance**: Optimized for Chinese character alignment
3. **Jieba integration**: Chinese word segmentation for proper WER calculation
4. **Multi-language support**: zh, en, ja with language-specific normalization

### Model Integration Strategy
- Git submodules for external models
- Factory pattern for model instantiation
- Configuration-driven model parameters
- Batch processing with error isolation
- Multi-GPU parallel inference support

### Hotword Evaluation Architecture
- Semantic matching beyond exact string matching
- Configurable library sizes for different scenarios
- F1-score, precision, recall metrics
- Integration with main testing framework

### Parallel Processing
- Multi-process inference with GPU distribution
- Configurable batch sizes and worker counts
- Error isolation for failed samples
- Automatic GPU memory management

## Code Requirements (代码要求)
1. **精炼与单一职责**: Each function/class does one thing well
2. **可读性优先**: Code is written for humans first (KISS principle)
3. **逻辑核心优先**: Focus on core algorithms, omit temporary logging
4. **无副作用设计**: Prefer pure functions, isolate side effects
5. **功能接口**: Use Python third-party libraries, update requirements.txt

## Common Pitfalls
- **Model paths**: Always check Model/ submodules are initialized
- **Chinese text**: Use TextAligner for proper character-level alignment
- **Batch processing**: Failed samples don't stop entire batch
- **GPU memory**: Adjust batch_size in model config for GPU limits
- **Audio formats**: Framework supports wav/mp3/flac/m4a/ogg
- **Git submodules**: Run `git submodule update --init --recursive` after cloning

## Quick Debugging
```bash
# Check model loading
cd /Users/zhangsheng/code/ASR-Eval && python -c "from models.factory import ModelFactory; f = ModelFactory(); print(f.get_available_models())"

# Validate dataset
cd /Users/zhangsheng/code/ASR-Eval && python -c "from datasets.manager import TestDatasetManager; TestDatasetManager.validate_dataset('datasets/regression_test')"

# Test text normalization
cd /Users/zhangsheng/code/ASR-Eval && python -c "from evaluation.text_normalizer import TextNormalizer; n = TextNormalizer(); print(n.normalize('测试文本123'))"

# Check GPU availability
cd /Users/zhangsheng/code/ASR-Eval && python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

# Test audio loading
cd /Users/zhangsheng/code/ASR-Eval && python -c "import librosa; audio, sr = librosa.load('datasets/regression_test/sample.wav', sr=16000); print(f'Audio shape: {audio.shape}, Sample rate: {sr}')"
```

## Project Structure
```
core/                    # Core enums and data models
models/                  # ASR model implementations and factory
datasets/               # Dataset management
evaluation/             # Metrics calculation and text processing
testing/                # Test framework orchestration
utils/                  # Utility modules (config, reporting)
config/                 # Configuration files
Model/                  # Git submodules for external ASR models
datasets/               # Test datasets (regression, noise, accent, etc.)
script/                 # Automation scripts (hotword processing)
tests/                  # Unit tests
results/                # Test output directory
```



