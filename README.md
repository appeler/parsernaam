# Parsernaam: ML-Assisted Name Parser

[![image](https://github.com/appeler/parsernaam/workflows/test/badge.svg)](https://github.com/appeler/parsernaam/actions?query=workflow%3Atest)
[![image](https://img.shields.io/pypi/v/parsernaam.svg)](https://pypi.python.org/pypi/parsernaam)
[![image](https://static.pepy.tech/badge/parsernaam)](https://pepy.tech/project/parsernaam)

Most common name parsers use crude pattern matching and the sequence of
strings, e.g., the last word is the last name, to parse names. This
approach is limited and fragile, especially for Indian names. We take a
machine-learning approach to the problem. Using the large voter
registration data in India and the US, we build machine-learning-based name
parsers that predict whether the string is a first or last name.

For Indian electoral rolls, we assume the last name is the word in the
name that is shared by multiple family members. (We table the expansion
to include compound last names\-\--extremely rare in India\-\--till the
next iteration.)

# Gradio App.

[parsernaam on HF](https://huggingface.co/spaces/sixtyfold/parsernaam)

# Installation

``` bash
pip install parsernaam
```

# Usage

## Python API

```python
import pandas as pd
from parsernaam.parse import ParseNames

# Create DataFrame with names to parse
df = pd.DataFrame({'name': ['Jan', 'Nicholas Turner', 'Petersen', 'Nichols Richard', 'Piet',
                           'John Smith', 'Janssen', 'Kim Yeon']})

# Parse names using ML models
results = ParseNames.parse(df)
print(results.to_markdown())
```

**Output:**
```
|    | name            | parsed_name                                                                   |
|---:|:----------------|:------------------------------------------------------------------------------|
|  0 | Jan             | {'name': 'Jan', 'type': 'first', 'prob': 0.677}                            |
|  1 | Nicholas Turner | {'name': 'Nicholas Turner', 'type': 'first_last', 'prob': 0.999}           |
|  2 | Petersen        | {'name': 'Petersen', 'type': 'last', 'prob': 0.534}                        |
|  3 | Nichols Richard | {'name': 'Nichols Richard', 'type': 'last_first', 'prob': 0.999}           |
|  4 | Piet            | {'name': 'Piet', 'type': 'first', 'prob': 0.538}                           |
|  5 | John Smith      | {'name': 'John Smith', 'type': 'first_last', 'prob': 0.997}                |
|  6 | Janssen         | {'name': 'Janssen', 'type': 'first', 'prob': 0.593}                        |
|  7 | Kim Yeon        | {'name': 'Kim Yeon', 'type': 'last_first', 'prob': 0.999}                  |
```

## Command Line Interface

```bash
parse_names input.csv -o output.csv -n name_column
```

## Features

- **Machine Learning Based**: Uses LSTM neural networks trained on voter registration data
- **Multi-language Support**: Handles Indian, Western, and other international name patterns  
- **High Accuracy**: Confidence scores provided for each prediction
- **Performance Optimized**: Model caching and batch processing support
- **Robust Error Handling**: Handles edge cases like empty names, special characters, etc.

# Data

The model is trained on names from the Florida Voter Registration Data
from early 2022. The data are available on the [Harvard
Dataverse](http://dx.doi.org/10.7910/DVN/UBIG3F)

# Authors

Rajashekar Chintalapati and Gaurav Sood

# Contributing

Contributions are welcome. Please open an issue if you find a bug or
have a feature request.

## 🔗 Adjacent Repositories

- [appeler/naamkaran](https://github.com/appeler/naamkaran) — generative model for names
- [appeler/ethnicolr2](https://github.com/appeler/ethnicolr2) — Ethnicolr implementation with new models in pytorch
- [appeler/namesexdata](https://github.com/appeler/namesexdata) — Data on international first names and sex of people with that name
- [appeler/pranaam](https://github.com/appeler/pranaam) — pranaam: predict religion based on name
- [appeler/graphic_names](https://github.com/appeler/graphic_names) — Infer the gender of a person with a particular first name using Google image search and Clarifai

# License

The package is released under the [MIT
License](https://opensource.org/licenses/MIT).
