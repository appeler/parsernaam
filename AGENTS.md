# Development guidance

Parsernaam classifies Latin-script name strings as first, last, first_last, or
last_first. Its four labels describe source-field patterns and ordering, not
personal identity. Version 0.4 uses retrained packed LSTMs hosted on Hugging Face
at a pinned commit. Historical weights used FL registrations and Census surnames;
the reproducible pipeline uses FL registrations only and evaluates transfer to NC.

## Commands

```bash
uv sync --all-groups --all-extras
make ci
make docs
make build
PARSERNAAM_MODEL_DIR=training/artifacts uv run pytest -m live tests/test_010_name_parser.py tests/test_020_edge_cases.py --no-cov
parse_names input.parquet --output output.parquet --names-col full_name
```

The Python API takes a DataFrame and returns a copy with one `parsed_name` dict
per row. Each dict preserves `name` and returns `type` and a raw model `prob`.
Invalid, blank, or non-Latin-only input receives unknown with score zero.

## Layout

- `src/parsernaam/text.py`: normalization and batched character encoding.
- `src/parsernaam/naam.py`: manifest-driven batched inference.
- `src/parsernaam/model.py`: packed LSTM architecture.
- `src/parsernaam/_resources.py`: pinned HF downloads and artifact verification.
- `training/`: preparation, training, and frozen evaluation commands.
- `training/legacy.py`: historical model adapter for evaluation only.

Sequence length and architecture come from `model_manifest.json`. Runtime weights
use SafeTensors; vocabulary uses Parquet. Training chooses MPS, CUDA, or CPU.
Prepared records, checkpoints, and source-linked report samples remain local.
README content is included automatically in Sphinx. Evaluation generates the
README table between its evaluation markers. Run the local CI before release;
its coverage floor is 95%.
