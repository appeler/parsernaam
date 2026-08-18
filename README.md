# Parsernaam

[![CI](https://github.com/appeler/parsernaam/actions/workflows/ci.yml/badge.svg)](https://github.com/appeler/parsernaam/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/parsernaam.svg)](https://pypi.org/project/parsernaam/)
[![Downloads](https://static.pepy.tech/badge/parsernaam)](https://pepy.tech/project/parsernaam)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97-models-yellow)](https://huggingface.co/gojiberries/parsernaam)

Parsernaam uses two character-level LSTM classifiers to label a single token as
`first` or `last`, or a multi-token string as `first_last` or `last_first`. It
is useful when name fields were not collected separately and simple word-order
rules are inadequate.

These labels cannot represent every naming convention. Model scores are not
calibrated guarantees, and errors and population imbalance in the training
records can affect predictions. Do not use the output to infer ethnicity,
citizenship, religion, gender, eligibility, or identity, or as the sole input
to a consequential decision.

## Installation

```bash
pip install parsernaam
```

Install the optional Gradio interface with:

```bash
pip install "parsernaam[web]"
```

## Python API

```python
import pandas as pd

from parsernaam import parse_names

names = pd.DataFrame(
    {
        "full_name": [
            "Jan",
            "Nicholas Turner",
            "Nichols Richard",
            "Kim Yeon",
        ]
    },
    index=pd.Index([10, 20, 30, 40], name="row_id"),
)

result = parse_names(names, names_col="full_name")
print(result[["full_name", "parsed_name"]])
```

`parse_names` returns a copy, preserves the input index and other columns, and
adds `parsed_name`. Each value contains the original string, one of the four
model labels, and its model score. Existing `parsed_name` values are replaced
without merge suffixes.

Invalid or blank values receive the `unknown` label and a score of `0.0`.

## Command line

The command-line interface uses Parquet for typed input and output:

```bash
parse_names input.parquet --output output.parquet --names-col full_name
```

The name column defaults to `name`, and the output path defaults to
`output.parquet`.

## Model artifacts

The two PyTorch state dictionaries and non-null string vocabulary are published
at [gojiberries/parsernaam](https://huggingface.co/gojiberries/parsernaam).
Parsernaam downloads them from an immutable Hugging Face commit and verifies
their SHA-256 hashes against the packaged `model_manifest.json`. Set
`PARSERNAAM_MODEL_DIR` to use an explicitly managed local copy. The Hugging
Face client honors its standard authentication configuration, including
`HF_TOKEN`.

The repository documentation describes training records derived from Indian
and United States voter registrations and cites the early 2022 Florida voter
registration data at [Harvard Dataverse](https://doi.org/10.7910/DVN/UBIG3F).
A complete row-level training manifest is not available, so use the models for
exploration rather than population claims.

## Development

```bash
uv sync --all-groups --all-extras
make ci
make docs
```

## Authors

Rajashekar Chintalapati and Gaurav Sood

## Related projects

- [naamkaran](https://github.com/appeler/naamkaran) generates synthetic name-like strings.
- [ethnicolr](https://github.com/appeler/ethnicolr) is the canonical ethnicity-from-name package.
- [pranaam](https://github.com/appeler/pranaam) estimates aggregate religion patterns from names.

## License

Parsernaam is released under the
[MIT License](https://github.com/appeler/parsernaam/blob/main/LICENSE).
