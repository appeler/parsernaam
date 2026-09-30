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

Invalid, blank, or non-Latin-only values receive the `unknown` label and a
score of `0.0`. Input is normalized with Unicode accent removal, whitespace
collapse, and title-casing before batched inference. Case variants share the
same prediction and score. Apostrophes and hyphens are retained; unsupported
characters are removed. Long inputs are truncated at the model's manifest
sequence length. The result's `name` always preserves the original input.

## Command line

The command-line interface uses Parquet for typed input and output:

```bash
parse_names input.parquet --output output.parquet --names-col full_name
```

The name column defaults to `name`, and the output path defaults to
`output.parquet`.

## Model artifacts

The two SafeTensors weight files and string vocabulary are published
at [gojiberries/parsernaam](https://huggingface.co/gojiberries/parsernaam).
Parsernaam downloads them from an immutable Hugging Face commit and verifies
their SHA-256 hashes against the packaged `model_manifest.json`. Set
`PARSERNAAM_MODEL_DIR` to use an explicitly managed local copy. The Hugging
Face client honors its standard authentication configuration, including
`HF_TOKEN`.

The published notebooks use early 2022 Florida voter registrations at
[Harvard Dataverse](https://doi.org/10.7910/DVN/UBIG3F) and a US Census surname
list. They do not establish Indian training provenance. Their validation and
test splits overlap training, so their reported quality is not a held-out
estimate. Version 0.4 trains on FL registration fields only and evaluates geographic
transfer on NC registration fields. The weights are hosted on Hugging Face;
the package pins their exact revision.

Use local retrained artifacts with:

```bash
PARSERNAAM_MODEL_DIR=training/artifacts uv run parse_names input.parquet
```

A local `model_manifest.json` specifies architecture, encoding, sequence lengths,
labels, hashes, and evaluation provenance. Complete local manifests are checked
for missing or corrupted files before loading. Runtime inference uses the retrained packed models with padding 0 and unknown 1.
The original encoding is retained only in the historical evaluation harness.

## Evaluation

The [training pipeline](https://github.com/appeler/parsernaam/blob/8b35dae96de9401972c516c2fde78a73f4c539d5/training/README.md) documents the source fields,
surname-disjoint FL splits, seeded record samples, and independent baselines.
Published and fixed-weight FL comparisons remain contaminated; NC comparisons
measure transfer between states and can contain shared name strings. Raw scores
are evaluated for calibration, rather than treated as calibrated probabilities.

<!-- evaluation:start -->
Accuracy on as-is input (NC is all caps):

| source | task | majority | frequency | published | fixed_inference | retrained |
| --- | --- | --- | --- | --- | --- | --- |
| FL | ordering | 50.0% | 46.2% | 85.1% | 98.5% | 92.7% |
| FL | single_distinct_name | 49.8% | 45.2% | 79.4% | 86.5% | 81.6% |
| FL | single_record_weighted | 51.0% | 49.2% | 73.6% | 81.8% | 84.4% |
| NC | ordering | 50.0% | 87.1% | 53.2% | 98.4% | 96.2% |
| NC | single_distinct_name | 41.4% | 66.9% | 51.2% | 87.3% | 79.6% |
| NC | single_record_weighted | 50.0% | 84.9% | 54.2% | 80.9% | 90.9% |

Expected calibration error of raw scores (lower is better):

| source | task | majority | frequency | published | fixed_inference | retrained |
| --- | --- | --- | --- | --- | --- | --- |
| FL | ordering | 0.000 | 0.433 | 0.130 | 0.008 | 0.020 |
| FL | single_distinct_name | 0.013 | 0.303 | 0.069 | 0.056 | 0.063 |
| FL | single_record_weighted | 0.001 | 0.399 | 0.096 | 0.129 | 0.060 |
| NC | ordering | 0.000 | 0.100 | 0.430 | 0.007 | 0.003 |
| NC | single_distinct_name | 0.097 | 0.066 | 0.224 | 0.087 | 0.084 |
| NC | single_record_weighted | 0.010 | 0.087 | 0.151 | 0.142 | 0.008 |

Release gates: case_invariance=True, case_violations=0, retrained_beats_both_baselines=True, fixed_beats_published=True.
<!-- evaluation:end -->

The generated aggregate table is saved in
[training/reports/summary.md](https://github.com/appeler/parsernaam/blob/8b35dae96de9401972c516c2fde78a73f4c539d5/training/reports/summary.md). The local HTML report
includes score-band accuracy, ECE, support breakdowns, and 20 source-linked
sample rows per task and state. The release uses the retrained models shown
in the table. Record-weighted single-name accuracy improves over fixed original
weights, but distinct-name and ordering accuracy are lower. On lookup-supported
distinct NC names, frequency lookup scores 92.0% versus the retrained model's
81.3%. These tradeoffs matter when choosing whether the package suits a dataset.

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
