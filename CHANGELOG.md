# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-09-29

### Fixed

- Share Latin-script normalization across training and batched inference,
  including case invariance, accent removal, and unsupported-input handling.
- Read per-model sequence lengths and architecture from artifact manifests.
- Verify complete local artifact manifests, including vocabulary integrity.
- Read the last real character through packed sequences in retrained models.

### Added

- Reproducible FL preparation and MPS training with surname-based splits,
  epoch checkpoints, validation-loss selection, and evidence hashes.
- Frozen NC and FL evaluation against original inference, fixed inference,
  majority class and FL frequency lookup; record and distinct-name views,
  score-band reliability, ECE, and source-linked examples.
- Generated evaluation tables and local-artifact quality tests.

### Changed

- Correct training provenance to FL registrations and US Census surnames for
  published weights; new weights use FL only. Remove the Indian-names keyword.
- Replace Colab-only training notebooks with the command-line pipeline.
- Enforce the 95% local coverage floor.
- Publish retrained weights as SafeTensors on Hugging Face and pin their commit;
  retain the original-weight adapter only in the evaluation harness.

### Evaluation

- On held-out NC data, retrained models achieve 90.9% record-weighted single-name
  accuracy, 79.6% distinct-name accuracy, and 96.2% ordering accuracy, beating
  majority and frequency baselines on each view.
- Fixed original weights remain stronger on distinct names (87.3%) and ordering
  (98.4%); retrained weights improve record-weighted single names from 80.9%.
- Record-weighted single-name ECE is 0.008, distinct-name ECE 0.084, and ordering
  ECE 0.003. Scores remain raw model outputs.

## [0.3.0] - 2026-08-17

### Changed

- Store both model state dictionaries and the typed Parquet vocabulary on
  Hugging Face at an immutable revision instead of shipping model artifacts in
  the wheel.
- Verify downloaded artifact hashes against a packaged model manifest.
- Adopt py-canon 1.0.1, the uv_build backend, current dependencies, reusable
  workflows, and the standard `src` layout.
- Use typed Parquet files for command-line input and output.
- Drive the documentation from the README while retaining complete autodoc API
  coverage.

### Fixed

- Honor the CLI and Python API `names_col` selection.
- Return a copy without mutating the caller's DataFrame, while preserving its
  columns and index.
- Cache vocabularies by artifact path so local model overrides cannot reuse a
  stale incompatible vocabulary.

## [0.2.0]

### Added

- LSTM-based name parser with single-name and positional models, a command-line
  interface, and an optional Gradio demo.
