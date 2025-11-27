# Parsernaam: ML-Assisted Name Parser

**Parsernaam** is a machine learning-based name parser that intelligently classifies name components 
as first names, last names, or combinations. Unlike traditional pattern-matching approaches, 
parsernaam uses LSTM neural networks trained on voter registration data to achieve high accuracy, 
particularly for Indian names where conventional methods often fail.

## Key Features

🤖 **Machine Learning Based**: Uses LSTM neural networks trained on real voter registration data

🌍 **Multi-language Support**: Handles Indian, Western, and other international name patterns  

🎯 **High Accuracy**: Provides confidence scores for each prediction

⚡ **Performance Optimized**: Model caching and batch processing support

🛡️ **Robust Error Handling**: Handles edge cases like empty names, special characters, etc.

## Quick Start

```python
import pandas as pd
from parsernaam.parse import ParseNames

# Create DataFrame with names to parse
df = pd.DataFrame({'name': ['Jan', 'Nicholas Turner', 'Petersen']})

# Parse names using ML models
results = ParseNames.parse(df)
print(results)
```

## Documentation Contents

```{toctree}
:maxdepth: 2
:caption: User Guide

installation
usage
api
```

```{toctree}
:maxdepth: 1
:caption: Additional Information

GitHub Repository <https://github.com/appeler/parsernaam>
PyPI Package <https://pypi.org/project/parsernaam/>
Gradio Demo <https://huggingface.co/spaces/sixtyfold/parsernaam>
```

## How It Works

Most common name parsers use crude pattern matching (e.g., "the last word is the last name"). 
This approach is limited and fragile, especially for Indian names. Parsernaam takes a 
machine-learning approach to the problem using large voter registration datasets from India and the US.

For Indian electoral rolls, we assume the last name is the word in the name that is shared by 
multiple family members, providing a more culturally-aware parsing approach.

## Model Architecture

- **Character-level tokenization** with fixed sequence length of 30
- **LSTM neural network** with embedding and softmax layers
- **Dual model approach**: 
  - Single names: binary classification (first/last)
  - Multi-word names: positional classification (first_last/last_first)
- **CUDA support** with CPU fallback

## Training Data

The models are trained on names from the Florida Voter Registration Data from early 2022. 
The data are available on the [Harvard Dataverse](http://dx.doi.org/10.7910/DVN/UBIG3F).

## Authors

Rajashekar Chintalapati and Gaurav Sood

## License

The package is released under the [MIT License](https://opensource.org/licenses/MIT).

## Indices and Tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`