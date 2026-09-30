"""Evidence hashes and deterministic data contracts."""

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    """Hash a file without reading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write deterministic human-readable metadata."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def surname_split(surname: str) -> str:
    """Assign normalized surnames to 80/10/5/5 train/val/calibration/test."""
    bucket = int.from_bytes(hashlib.sha256(surname.encode()).digest()[:8], "big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "calibration" if bucket < 95 else "test"


def verify_preparation(directory: Path, filenames: list[str] | None = None) -> dict:
    """Reject changed frozen inputs before training or evaluating models."""
    metadata = json.loads((directory / "preparation.json").read_text())
    selected = metadata["files"] if filenames is None else filenames
    for filename in selected:
        if sha256(directory / filename) != metadata["files"][filename]:
            raise ValueError(f"prepared input changed: {filename}")
    return metadata
