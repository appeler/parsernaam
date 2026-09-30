"""Contracts for published parsernaam artifacts."""

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest

from parsernaam._resources import HF_REPO, HF_REVISION, MODEL_MANIFEST, resolve_model


def test_local_override_avoids_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit model directory takes precedence over the Hub."""
    model = tmp_path / "parsernaam.safetensors"
    model.write_bytes(b"weights")
    monkeypatch.setenv("PARSERNAAM_MODEL_DIR", str(tmp_path))

    with patch("huggingface_hub.hf_hub_download") as download:
        assert resolve_model("models/parsernaam.safetensors") == str(model)
    download.assert_not_called()


def test_missing_local_artifact_uses_exact_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback download uses the declared repository and revision."""
    monkeypatch.setenv("PARSERNAAM_MODEL_DIR", str(tmp_path))

    downloaded = tmp_path / "downloaded.safetensors"
    downloaded.write_bytes(b"downloaded weights")
    expected_hash = MODEL_MANIFEST["artifacts"]["parsernaam.safetensors"]["sha256"]
    with (
        patch(
            "huggingface_hub.hf_hub_download", return_value=str(downloaded)
        ) as download,
        patch("parsernaam._resources._sha256", return_value=expected_hash),
    ):
        assert resolve_model("parsernaam.safetensors") == str(downloaded)
    download.assert_called_once_with(
        HF_REPO, "parsernaam.safetensors", revision=HF_REVISION
    )


def test_revision_is_an_immutable_commit() -> None:
    """Hub revision pins use the full hexadecimal commit identifier."""
    assert len(HF_REVISION) == 40
    assert set(HF_REVISION) <= set("0123456789abcdef")


def test_unknown_remote_artifact_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only artifacts declared in the package manifest may be downloaded."""
    monkeypatch.setenv("PARSERNAAM_MODEL_DIR", str(tmp_path))
    with (
        patch("huggingface_hub.hf_hub_download") as download,
        pytest.raises(ValueError, match="Unknown model artifact"),
    ):
        resolve_model("unknown.bin")
    download.assert_not_called()


def test_download_hash_mismatch_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt downloaded artifact cannot reach the model loader."""
    monkeypatch.delenv("PARSERNAAM_MODEL_DIR", raising=False)
    downloaded = tmp_path / "parsernaam.safetensors"
    downloaded.write_bytes(b"corrupt")
    with (
        patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
        pytest.raises(RuntimeError, match="integrity check"),
    ):
        resolve_model("parsernaam.safetensors")


@pytest.mark.live
def test_pinned_revision_contains_every_artifact() -> None:
    """The published snapshot contains both classifiers and the vocabulary."""
    from huggingface_hub import hf_hub_download, list_repo_files

    published = set(list_repo_files(HF_REPO, revision=HF_REVISION))
    assert {
        "parsernaam.safetensors",
        "parsernaam_pos.safetensors",
        "vocabulary.parquet",
    } <= published
    for filename, metadata in MODEL_MANIFEST["artifacts"].items():
        downloaded = Path(hf_hub_download(HF_REPO, filename, revision=HF_REVISION))
        assert hashlib.sha256(downloaded.read_bytes()).hexdigest() == metadata["sha256"]
        assert downloaded.stat().st_size == metadata["size"]


def test_local_manifest_integrity_and_missing_artifacts(tmp_path, monkeypatch):
    import json

    artifact = tmp_path / "parsernaam.safetensors"
    artifact.write_bytes(b"weights")
    metadata = {"sha256": hashlib.sha256(b"weights").hexdigest(), "size": 7}
    (tmp_path / "model_manifest.json").write_text(
        json.dumps({"artifacts": {"parsernaam.safetensors": metadata}})
    )
    monkeypatch.setenv("PARSERNAAM_MODEL_DIR", str(tmp_path))
    assert resolve_model("parsernaam.safetensors") == str(artifact)
    artifact.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="integrity check"):
        resolve_model("parsernaam.safetensors")
    artifact.unlink()
    with pytest.raises(RuntimeError, match="Missing local"):
        resolve_model("parsernaam.safetensors")
    with pytest.raises(ValueError, match="Unknown local"):
        resolve_model("unlisted.pt")
