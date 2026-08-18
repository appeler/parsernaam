"""Resolve immutable name-parser artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from importlib import resources
from pathlib import Path
from typing import Any, Final

MODEL_MANIFEST: Final[dict[str, Any]] = json.loads(
    resources.files("parsernaam")
    .joinpath("model_manifest.json")
    .read_text(encoding="utf-8")
)
HF_REPO: Final[str] = MODEL_MANIFEST["repository"]
HF_REVISION: Final[str] = MODEL_MANIFEST["revision"]
MODEL_DIR_ENV: Final[str] = "PARSERNAAM_MODEL_DIR"


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_model(filename: str) -> str:
    """Return a local path for a pinned model artifact.

    Args:
        filename: Artifact name, optionally prefixed by ``models/``.

    Returns:
        Local artifact path.

    Raises:
        ValueError: If a remote artifact is not declared in the manifest.
        RuntimeError: If a downloaded artifact fails integrity verification.
    """
    filename = filename.removeprefix("models/")
    override = os.environ.get(MODEL_DIR_ENV)
    if override:
        candidate = Path(override) / filename
        if candidate.is_file():
            return str(candidate)

    from huggingface_hub import hf_hub_download

    artifact = MODEL_MANIFEST["artifacts"].get(filename)
    if artifact is None:
        raise ValueError(f"Unknown model artifact: {filename}")

    downloaded = Path(hf_hub_download(HF_REPO, filename, revision=HF_REVISION))
    if _sha256(downloaded) != artifact["sha256"]:
        raise RuntimeError(f"Model artifact failed its integrity check: {filename}")
    return str(downloaded)
