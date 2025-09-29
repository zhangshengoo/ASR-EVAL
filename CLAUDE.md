# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an ASR (Automatic Speech Recognition) evaluation framework designed to compare different ASR models including FireRedASR, KimiAudio, and StepAudio2. The framework provides comprehensive evaluation capabilities including WER, CER, SER, and RTF metrics with specialized support for Chinese language ASR testing.

## Architecture

### Core Components
- **core/**: System enumerations, data models, and configuration management
- **models/**: ASR model implementations with factory pattern
- **datasets/**: Audio dataset management and validation
- **evaluation/**: Metrics calculation (WER, CER, SER, RTF)
- **testing/**: Main orchestration framework
- **utils/**: Configuration loading and report generation
- **results/**: Centralized results management

### Model Layer
- **models/base.py**: Abstract base class defining ASR model interface
- **models/factory.py**: Model factory for creating and managing model instances
- **models/kimi_audio.py**: KimiAudio model integration
- **models/fire_red_asr.py**: FireRedASR model implementation
- **models/step_audio2.py**: StepAudio2 model integration

### Data Processing Flow
1. **Dataset Loading** → Audio file discovery with transcription matching
2. **Model Inference** → Batch processing with timing and confidence
3. **Metrics Calculation** → Multi-dimensional evaluation (accuracy, speed, confidence)
4. **Results Reporting** → JSON, CSV, HTML formats with visualizations

## Development Commands

### Setup & Installation
```bash
pip install -r requirements.txt
```

### Testing Commands
```bash
pytest tests/                          # Run all tests
pytest tests/test_filename.py          # Run specific test file
pytest tests/test_filename.py::test_function_name  # Run specific test
pytest --cov=. tests/                  # Run with coverage
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

### Creating Custom Dataset
1. Prepare audio files with matching .txt transcription files
2. Place in datasets/ directory following naming convention
3. Add dataset configuration to config.json
4. Validate with TestDatasetManager

### Running Hotword Tests
1. Configure hotword settings in config/hotwords.json
2. Use --hotword-config parameter with main.py
3. Results include hotword-specific metrics and analysis

### Extending Metrics
1. Add new metric type to `core/enums.py`
2. Implement calculation in `evaluation/metrics.py`
3. Update reporting in `utils/reporter.py`

## File Formats Supported
- **Audio**: wav, mp3, flac, m4a, ogg
- **Transcriptions**: .txt, .trans files
- **Results**: JSON, CSV, HTML formats with visualizations
- **Configuration**: JSON configuration files

## Results Structure
Results are organized in timestamped directories under `results/` with:
- `results.json`: Detailed per-sample metrics
- `results.csv`: Aggregated metrics
- `report.html`: Visual comparison charts
- `detailed.json`: Extended analysis data