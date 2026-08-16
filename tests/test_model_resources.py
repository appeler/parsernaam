"""Contracts for published parsernaam artifacts."""

from pathlib import Path
from unittest.mock import patch

import pytest

from parsernaam._resources import HF_REPO, HF_REVISION, resolve_model


def test_local_override_avoids_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit model directory takes precedence over the Hub."""
    model = tmp_path / "parsernaam.pt"
    model.write_bytes(b"weights")
    monkeypatch.setenv("PARSERNAAM_MODEL_DIR", str(tmp_path))

    with patch("huggingface_hub.hf_hub_download") as download:
        assert resolve_model("models/parsernaam.pt") == str(model)
    download.assert_not_called()


def test_missing_local_artifact_uses_exact_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback download uses the declared repository and revision."""
    monkeypatch.setenv("PARSERNAAM_MODEL_DIR", str(tmp_path))

    with patch(
        "huggingface_hub.hf_hub_download", return_value="/cache/parsernaam.pt"
    ) as download:
        assert resolve_model("parsernaam.pt") == "/cache/parsernaam.pt"
    download.assert_called_once_with(HF_REPO, "parsernaam.pt", revision=HF_REVISION)


def test_revision_is_an_immutable_commit() -> None:
    """Hub revision pins use the full hexadecimal commit identifier."""
    assert len(HF_REVISION) == 40
    assert set(HF_REVISION) <= set("0123456789abcdef")


@pytest.mark.live
def test_pinned_revision_contains_every_artifact() -> None:
    """The published snapshot contains both classifiers and the vocabulary."""
    from huggingface_hub import list_repo_files

    published = set(list_repo_files(HF_REPO, revision=HF_REVISION))
    assert {"parsernaam.pt", "parsernaam_pos.pt", "vocabulary.parquet"} <= published
