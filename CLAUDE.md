# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is an ASR (Automatic Speech Recognition) evaluation framework designed to compare different ASR models including FireRedASR, KimiAudio, and StepAudio2.

## Architecture
- **Model/**: Contains model-specific implementations for different ASR systems
  - `FireRedASR/`: FireRedASR model integration
  - `KimiAudio/`: KimiAudio model integration
  - `StepAudio2/`: StepAudio2 model integration
- **config/**: Configuration files for models and evaluation settings
- **datasets/**: Audio datasets for evaluation
- **results/**: Evaluation results and metrics output
- **tests/**: Unit and integration tests

## Dependencies
The project uses PyTorch-based models with audio processing libraries. Key dependencies include:
- PyTorch ≥ 2.0.0
- Transformers ≥ 4.30.0
- Librosa for audio processing
- JIwer for WER/CER calculations
- FastAPI for web API endpoints
- Pytest for testing

## Development Commands
- Install dependencies: `pip install -r requirements.txt`
- Run tests: `pytest tests/`
- Run specific test: `pytest tests/test_filename.py::test_function_name`
- Run with coverage: `pytest --cov=. tests/`

## Common Tasks
- **Model Evaluation**: Run comparative analysis between ASR models
- **Dataset Processing**: Prepare audio datasets for evaluation
- **Metrics Calculation**: Compute WER, CER, and other ASR metrics
- **API Development**: Build REST API endpoints for model inference