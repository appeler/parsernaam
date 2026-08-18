# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
