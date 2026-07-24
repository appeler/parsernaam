# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Added `scikit-learn` to `[project].dependencies`; the model vocabulary
  pickle requires it to unpickle, and its absence caused `ModuleNotFoundError`
  at import time.

### Changed
- Adopted the [py-canon](https://github.com/gojiplus/py-canon) fleet standard
  for CI, docs, and dependency management.

## [0.2.0]

### Added
- LSTM-based ML name parser with single-name and positional models.
- Command-line interface (`parse_names`) and optional Gradio web demo.
