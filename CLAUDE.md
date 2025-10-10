# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an ASR (Automatic Speech Recognition) evaluation framework designed to compare different ASR models including FireRedASR, KimiAudio, and StepAudio2. The framework provides comprehensive evaluation capabilities including WER, CER, SER, and RTF metrics with specialized support for Chinese language ASR testing.

## Architecture

### Core Components
- **core/**: System enumerations, data models, and configuration management
- **models/**: ASR model implementations with factory pattern
- **datasets/**: Audio dataset management and validation
- **evaluation/**: Metrics calculation (WER, CER, SER, RTF) and text alignment
- **testing/**: Main orchestration framework
- **utils/**: Configuration loading and report generation
- **results/**: Centralized results management
- **script/**: Utility scripts including configurable hotword processing

### Model Layer
- **models/base.py**: Abstract base class defining ASR model interface
- **models/factory.py**: Model factory for creating and managing model instances
- **models/kimi_audio.py**: KimiAudio model integration
- **models/fire_red_asr.py**: FireRedASR model implementation
- **models/step_audio2.py**: StepAudio2 model integration

### Hotword Evaluation Components
- **evaluation/hotword_evaluator.py**: Advanced hotword evaluator with semantic matching
- **evaluation/hotword_metrics.py**: Hotword metrics calculator (recall, precision, F1-score)
- **script/configurable_hotword.py**: Configurable hotword processing with flexible library sizes

### Data Processing Flow
1. **Dataset Loading** → Audio file discovery with transcription matching
2. **Model Inference** → Batch processing with timing and confidence
3. **Metrics Calculation** → Multi-dimensional evaluation (accuracy, speed, confidence)
4. **Results Reporting** → JSON, CSV, HTML formats with visualizations

## Development Commands

### Setup & Installation
```bash
pip install -r requirements.txt

# Initialize Git submodules for model repositories
git submodule update --init --recursive

# Optional: Install additional text normalization dependencies
pip install -r requirements-wetextprocessing.txt
pip install -r requirements-nemo.txt
pip install -r requirements-japanese.txt
```

### Testing Commands
```bash
pytest tests/                          # Run all tests
pytest tests/test_filename.py          # Run specific test file
pytest tests/test_filename.py::test_function_name  # Run specific test
pytest --cov=. tests/                  # Run with coverage
pytest -v tests/                       # Verbose output
```

### Model Evaluation Commands
```bash
# Single model evaluation
python main.py --model step_audio2 --dataset regression --output results/

# Comparative testing between models
python main.py --compare --models step_audio2,kimi_audio --datasets regression,noise

# Batch testing with test plan
python main.py --batch test_plan.json --output results/

# Hotword testing
python main.py --model step_audio2 --dataset hotword --hotword-config config/hotwords.json

# Advanced hotword testing with configurable library sizes
python script/configurable_hotword.py datasets/热词测试/场景1 "3,5,10"
python hotword_tester.py
python integrate_hotword_testing.py

# Text alignment and visualization
python -c "from evaluation.text_alignment import TextAligner, TextDiffVisualizer; aligner = TextAligner(); visualizer = TextDiffVisualizer(); result = aligner.generate_diff_report('reference text', 'hypothesis text'); print(visualizer.side_by_side_diff_from_alignment(result['alignment']))"
```

### Dataset Management
```bash
# Create sample dataset for testing
python main.py --create-sample-data --output sample_data/

# Validate dataset structure
python -c "from datasets.manager import TestDatasetManager; TestDatasetManager.validate_dataset('path/to/dataset')"
```

## Key Configuration Files

### Main Configuration
- `config/config.json`: Model paths, dataset configurations, evaluation settings
- Format: Hierarchical JSON with model-specific, dataset-specific, and evaluation sections

### Model Configuration Pattern
```json
{
  "models": {
    "step_audio2": {
      "model_path": "Model/StepAudio2",
      "device": "cuda",
      "batch_size": 8,
      "parameters": {}
    }
  }
}
```

## Key Dependencies
- **PyTorch ecosystem**: torch≥2.0.0, transformers≥4.30.0
- **Audio processing**: librosa≥0.10.0, soundfile≥0.12.0
- **Metrics**: jiwer≥3.0.0, editdistance≥0.6.0
- **Web**: fastapi≥0.100.0, uvicorn≥0.23.0
- **Chinese text**: jieba≥0.42.0
- **Testing**: pytest≥7.4.0

## Common Development Tasks

### Adding New ASR Model
1. Create new model class inheriting from `BaseASRModel` in `models/`
2. Register model in `models/factory.py` model registry
3. Add model configuration to `config/config.json`
4. Add model type to `core/enums.py`
5. Create model-specific config file in `config/models/[model_name].json`
6. Add Git submodule if model is external: `git submodule add [repo_url] Model/[ModelName]`

### Creating Custom Dataset
1. Prepare audio files with matching .txt transcription files
2. Place in datasets/ directory following naming convention
3. Add dataset configuration to config.json
4. Validate with TestDatasetManager

### Running Hotword Tests
1. Configure hotword settings in config/hotwords.json
2. Use --hotword-config parameter with main.py
3. Results include hotword-specific metrics and analysis

### Advanced Hotword Testing
```bash
# Configurable hotword processing with flexible library sizes
python script/configurable_hotword.py datasets/热词测试/场景1 "3,5,10"

# Complete hotword testing runner
python hotword_tester.py

# Integration testing with ASR framework
python integrate_hotword_testing.py
```

### Extending Metrics
1. Add new metric type to `core/enums.py`
2. Implement calculation in `evaluation/metrics.py`
3. Update reporting in `utils/reporter.py`

### Text Alignment and Visualization
The system includes Kaldi align-text based text difference visualization:
- **TextAligner**: Generates detailed alignment reports using Kaldi align-text
- **TextDiffVisualizer**: Provides word-level text comparison with color coding
- **Features**: Word-level alignment, substitution/deletion/insertion marking, side-by-side comparison
- **Recent Improvements**: Priority-based edit distance algorithm for Chinese text handling
- **Chinese Text Support**: Enhanced character-level alignment for Chinese text processing

### Git Submodule Management
```bash
# Add new model submodule
git submodule add [repository_url] Model/[ModelName]
git submodule update --init --recursive

# Update existing submodules
git submodule update --remote --merge

# Check submodule status
git submodule status
```

## File Formats Supported
- **Audio**: wav, mp3, flac, m4a, ogg
- **Transcriptions**: .txt, .trans files
- **Results**: JSON, CSV, HTML formats with visualizations
- **Configuration**: JSON configuration files
- **Hotword Config**: JSON configuration for hotword testing

## Results Structure
Results are organized in timestamped directories under `results/` with:
- `results.json`: Detailed per-sample metrics
- `results.csv`: Aggregated metrics
- `report.html`: Visual comparison charts
- `detailed.json`: Extended analysis data

## Key Development Notes

### Recent Development Focus
- **Text Alignment**: Priority-based edit distance algorithm for improved Chinese text handling
- **Hotword Testing**: Configurable hotword processing with flexible library sizes
- **Kaldi Integration**: Complete migration to Kaldi align-text based text visualization
- **Chinese Support**: Enhanced character-level alignment for Chinese text processing

### Testing Best Practices
- Always validate datasets before testing using TestDatasetManager
- Use batch testing for comprehensive model comparison
- Enable text normalization for Chinese language testing
- Check model configuration files for proper device and batch size settings

### Error Handling
- Framework includes comprehensive error handling and logging
- Failed model inferences are logged but don't stop batch processing
- Invalid audio files are skipped with detailed error reporting
- Configuration validation prevents common setup issues

## 代码要求
本文件定义协作生成代码时应遵循的核心原则和编码规范，旨在产出高质量、可维护、专业级的代码。
1. 精炼与单一职责
每个函数/类只做一件事，并把它做好。
接口设计力求简洁，避免不必要的参数和复杂性。
优先质量而非数量。
2. 可读性优先 
代码首先是写给人读的，其次才是给机器执行的。
清晰的逻辑结构和命名比晦涩的技巧更重要。
采用KISS原则。
3. 逻辑核心优先
初始代码生成应专注于核心算法和业务逻辑的实现。
省略临时的打印 (print)、日志 (logging) 和完整的测试脚手架，这些将在后续迭代中完善。
4. 无副作用设计
优先设计纯函数。
明确隔离产生副作用（如I/O操作、状态修改）的代码模块。
5. 功能接口
优先调用python三方库
调用不需要处理异常，无法import直接系统报错



