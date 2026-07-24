# Usage

Parsernaam provides both a Python API and a command-line interface for parsing names using machine learning models.

## Python API

### Basic Usage

The main entry point is the `ParseNames` class with its static `parse()` method:

```python
import pandas as pd
from parsernaam.parse import ParseNames

# Create DataFrame with names to parse
df = pd.DataFrame(
    {
        "name": [
            "Jan",
            "Nicholas Turner",
            "Petersen",
            "Nichols Richard",
            "Piet",
            "John Smith",
            "Janssen",
            "Kim Yeon",
        ]
    }
)

# Parse names using ML models
results = ParseNames.parse(df)
print(results)
```

### Output Format

The `parse()` method returns a DataFrame with an additional `parsed_name` column containing dictionaries with:

- **name**: The original name
- **type**: Classification result (`'first'`, `'last'`, `'first_last'`, `'last_first'`)
- **prob**: Confidence probability (0.0 to 1.0)

Example output:

| | name | parsed_name |
|---:|:---|:---|
| 0 | Jan | {'name': 'Jan', 'type': 'first', 'prob': 0.677} |
| 1 | Nicholas Turner | {'name': 'Nicholas Turner', 'type': 'first_last', 'prob': 0.999} |
| 2 | Petersen | {'name': 'Petersen', 'type': 'last', 'prob': 0.534} |
| 3 | Nichols Richard | {'name': 'Nichols Richard', 'type': 'last_first', 'prob': 0.999} |

### Working with Results

You can extract specific information from the parsed results:

```python
# Get just the classification types
df["name_type"] = df["parsed_name"].apply(lambda x: x["type"])

# Get confidence scores
df["confidence"] = df["parsed_name"].apply(lambda x: x["prob"])

# Filter high-confidence results
high_confidence = df[df["confidence"] > 0.9]
```

### Custom DataFrame Column

If your DataFrame uses a different column name for names, specify it:

```python
# DataFrame with 'full_name' column instead of 'name'
df = pd.DataFrame({"full_name": ["John Smith", "Jane Doe"]})

# Rename column or create a 'name' column
df_renamed = df.rename(columns={"full_name": "name"})
results = ParseNames.parse(df_renamed)
```

### Batch Processing

For large datasets, parsernaam efficiently processes names in batches:

```python
# Large dataset
import numpy as np

large_df = pd.DataFrame(
    {"name": np.random.choice(["John Smith", "Jane Doe", "Kim Yeon"], 10000)}
)

# Process efficiently with automatic batching
results = ParseNames.parse(large_df)
```

## Command Line Interface

### Basic Usage

Parse names from a CSV file:

```bash
parse_names input.csv -o output.csv -n name_column
```

### Arguments

- **input**: Input CSV file containing names to parse (required)
- **-o, --output**: Output CSV file (default: `parsernaam_output.csv`)
- **-n, --names-col**: Column name containing the names to parse (required)

### Examples

```bash
# Parse names from 'name' column in input.csv
parse_names data.csv -o results.csv -n name

# Parse names from 'full_name' column
parse_names employees.csv -o parsed_employees.csv -n full_name

# Use default output filename
parse_names names.csv -n participant_name
```

### Input File Format

The input CSV file should contain at least one column with names:

```text
id,name,age
1,John Smith,35
2,Kim Yeon,28
3,Nicholas Turner,42
```

### Output File Format

The output CSV includes all original columns plus a `parsed_name` column:

```text
id,name,age,parsed_name
1,John Smith,35,"{'name': 'John Smith', 'type': 'first_last', 'prob': 0.997}"
2,Kim Yeon,28,"{'name': 'Kim Yeon', 'type': 'last_first', 'prob': 0.999}"
3,Nicholas Turner,42,"{'name': 'Nicholas Turner', 'type': 'first_last', 'prob': 0.999}"
```

## Web Interface

### Gradio Demo

Try parsernaam interactively using the Gradio web interface:

[Parsernaam on Hugging Face](https://huggingface.co/spaces/sixtyfold/parsernaam)

### Running Locally

You can also run the Gradio interface locally:

```bash
python gradio_app.py
```

This will start a local web server where you can test name parsing interactively.

## Model Behavior

### Single Names

For single names, the model classifies them as either:

- **first**: Likely a first name (e.g., "John", "Sarah", "Raj")
- **last**: Likely a last name (e.g., "Smith", "Patel", "Johnson")

### Multi-word Names

For names with multiple words, the model determines the order:

- **first_last**: First name followed by last name (e.g., "John Smith")
- **last_first**: Last name followed by first name (e.g., "Smith John")

### Confidence Scores

All predictions include confidence scores:

- **0.9+**: Very high confidence
- **0.7-0.9**: High confidence  
- **0.5-0.7**: Moderate confidence
- **<0.5**: Low confidence (consider manual review)

## Cultural Considerations

The model is particularly effective for:

- **Indian names**: Trained on Indian voter registration data
- **Western names**: Trained on US voter registration data
- **Mixed cultural contexts**: Handles various naming conventions

For Indian names specifically, the model assumes that last names are typically shared among family members, providing culturally-aware parsing.

## Error Handling

Parsernaam gracefully handles various edge cases:

- **Empty names**: Returns appropriate default values
- **Special characters**: Processes names with Unicode characters
- **Very long names**: Truncates to model's sequence length limit
- **Missing data**: Handles NaN values in DataFrames

## Performance Tips

1. **Batch Processing**: Process multiple names at once for better performance
2. **GPU Acceleration**: Ensure CUDA is available for faster inference
3. **Model Caching**: Models are automatically cached after first load
4. **Memory Management**: For very large datasets, consider processing in chunks

## Troubleshooting

### Common Issues

**Low Confidence Scores**: Some names may be ambiguous. Consider:

- Checking if the name follows expected patterns
- Using additional context clues if available
- Manual review for confidence < 0.7

**Unexpected Classifications**: The model may occasionally misclassify:

- Very unusual or rare names
- Names from cultures not well-represented in training data
- Names with unconventional spellings

**Performance Issues**: For slow processing:

- Ensure you're using batch processing (passing DataFrames vs individual names)
- Check if CUDA is available and properly configured
- Consider processing large datasets in smaller chunks