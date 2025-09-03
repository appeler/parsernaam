# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Parsernaam is a Python package that uses machine learning to parse names, particularly for Indian names where traditional pattern matching fails. The project uses PyTorch LSTM models trained on voter registration data to classify name components as first names, last names, or combinations.

## Development Commands

### Testing
- Run all tests: `pytest`
- Run specific test: `pytest parsernaam/tests/test_010_name_parser.py`

### Installation and Setup
- Install package in development mode: `pip install -e .`
- Install with test dependencies: `pip install .[test]`
- Install with development dependencies: `pip install .[dev]`

### Package Building
- Build package: `python -m build`
- Check manifest: `check-manifest` (if dev dependencies installed)

### Running the Application
- CLI usage: `parse_names input.csv -o output.csv -n name_column`
- Gradio web interface: `python gradio_app.py`

## Code Architecture

### Core Components

1. **Parsernaam class** (`parsernaam/naam.py`): Base class with static `parse()` method that handles the ML inference pipeline
2. **ParseNames class** (`parsernaam/parse.py`): Main API class that extends Parsernaam, defines model file paths, and provides the public interface
3. **LSTM model** (`parsernaam/model.py`): PyTorch neural network with embedding, LSTM, and softmax layers
4. **Utilities** (`parsernaam/utils.py`): Command-line argument parsing

### Model Files
- `models/parsernaam.pt`: LSTM model for single names (first vs last)
- `models/parsernaam_pos.pt`: LSTM model for multi-word names (first_last vs last_first)
- `models/parsernaam.joblib`: Character vocabulary vectorizer

### Key Architecture Details
- Uses character-level tokenization with a fixed sequence length of 30
- Supports CUDA if available, falls back to CPU
- Single names use binary classification (first/last)
- Multi-word names use positional classification (first_last/last_first)
- Returns predictions with confidence probabilities

## Input/Output Format
- Input: DataFrame with 'name' column containing names to parse
- Output: DataFrame with added 'parsed_name' column containing dict with 'name', 'type', and 'prob' fields