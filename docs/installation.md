# Installation

## Requirements

Parsernaam requires Python 3.11-3.12 and has the following dependencies:

- **pandas**: For data manipulation and analysis
- **torch**: PyTorch for neural network models
- **joblib**: For model serialization
- **numpy**: For numerical operations

## Basic Installation

Install parsernaam from PyPI using pip:

```bash
pip install parsernaam
```

This will install the package and all required dependencies.

## Development Installation

For development purposes, you can install parsernaam in editable mode:

```bash
git clone https://github.com/appeler/parsernaam.git
cd parsernaam
pip install -e .
```

## Development Dependencies

To install additional development dependencies for testing and building:

```bash
# Install with test dependencies
pip install .[test]

# Install with development dependencies (includes testing tools)
pip install .[dev]

# Install all dependencies
pip install .[test,dev]
```

## Verifying Installation

To verify that parsernaam is installed correctly, run:

```python
import parsernaam
from parsernaam.parse import ParseNames

# Test with a simple name
import pandas as pd

df = pd.DataFrame({"name": ["John Smith"]})
result = ParseNames.parse(df)
print(result)
```

If the installation is successful, you should see output with the parsed name result.

## GPU Support

Parsernaam automatically detects and uses CUDA-compatible GPUs when available for faster inference. 
No additional installation steps are required - if PyTorch detects CUDA, the models will automatically use GPU acceleration.

To check if CUDA is available:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
```

## Command Line Interface

After installation, you can use the command line interface:

```bash
parse_names --help
```

## Troubleshooting

### Common Issues

**Import Error**: If you encounter import errors, ensure all dependencies are installed:

```bash
pip install --upgrade parsernaam
```

**CUDA Issues**: If you have CUDA installation issues, parsernaam will fall back to CPU processing automatically.

**Model Download**: Models are included in the package installation and should be available immediately after install.

### Getting Help

If you encounter issues:

1. Check the [GitHub Issues](https://github.com/appeler/parsernaam/issues) page
2. Ensure you're using a supported Python version (3.11-3.12)
3. Try reinstalling with `pip install --force-reinstall parsernaam`