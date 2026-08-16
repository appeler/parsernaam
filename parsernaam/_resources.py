"""Resolve immutable name-parser artifacts."""

from __future__ import annotations

import os
from pathlib import Path

HF_REPO = "gojiberries/parsernaam"
HF_REVISION = "4061401e5cf9d5b903d0891fcc3c9381540883e5"
MODEL_DIR_ENV = "PARSERNAAM_MODEL_DIR"


def resolve_model(filename: str) -> str:
    """Return a local path for a pinned model artifact.

    Args:
        filename: Artifact name, optionally prefixed by ``models/``.

    Returns:
        Local artifact path.
    """
    filename = filename.removeprefix("models/")
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        candidate = Path(override) / filename
        if candidate.is_file():
            return str(candidate)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, filename, revision=HF_REVISION)
