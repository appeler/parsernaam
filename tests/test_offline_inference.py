"""Offline inference tests with structurally valid deterministic artifacts."""

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from safetensors.torch import save_file

from parsernaam import parse_names
from parsernaam.config import ModelConfig
from parsernaam.model import LSTM
from parsernaam.parse import ParseNames


@pytest.fixture
def local_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create deterministic models and a typed vocabulary.

    Args:
        tmp_path: Temporary test directory.
        monkeypatch: Environment patch fixture.

    Returns:
        Directory containing the test artifacts.
    """
    tokens = list(" abcdefghijklmnopqrstuvwxyz")
    table = pa.Table.from_arrays(
        [pa.array(tokens, type=pa.string())], names=["token"]
    ).cast(pa.schema([pa.field("token", pa.string(), nullable=False)]))
    pq.write_table(table, tmp_path / "vocabulary.parquet")

    input_size = len(tokens) + 2
    manifest = {"artifacts": {}}
    for filename, categories in (
        ("parsernaam.safetensors", ModelConfig.CATEGORIES_SINGLE),
        ("parsernaam_pos.safetensors", ModelConfig.CATEGORIES_POSITIONAL),
    ):
        model = LSTM(
            input_size,
            16,
            len(categories),
            num_layers=1,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        save_file(model.state_dict(), tmp_path / filename)
        from parsernaam._resources import _sha256

        manifest["artifacts"][filename] = {
            "format": "safetensors",
            "encoding": "padding0_unknown1",
            "seq_len": 47,
            "hidden_size": 16,
            "num_layers": 1,
            "labels": categories,
            "sha256": _sha256(tmp_path / filename),
            "size": (tmp_path / filename).stat().st_size,
        }
    from parsernaam._resources import _sha256

    manifest["artifacts"]["vocabulary.parquet"] = {
        "sha256": _sha256(tmp_path / "vocabulary.parquet"),
        "size": (tmp_path / "vocabulary.parquet").stat().st_size,
    }
    (tmp_path / "model_manifest.json").write_text(json.dumps(manifest))

    monkeypatch.setenv("PARSERNAAM_MODEL_DIR", str(tmp_path))
    ParseNames._models_cache.clear()
    ParseNames._vocab_cache.clear()
    yield tmp_path
    ParseNames._models_cache.clear()
    ParseNames._vocab_cache.clear()


def test_inference_logic_without_network(local_models: Path) -> None:
    """Single, positional, and invalid names use the correct inference paths."""
    frame = pd.DataFrame({"name": ["Ada", "Ada Lovelace", "", None, 123]})

    result = parse_names(frame)

    assert result.loc[0, "parsed_name"] == {
        "name": "Ada",
        "type": "last",
        "prob": pytest.approx(0.5),
    }
    assert result.loc[1, "parsed_name"] == {
        "name": "Ada Lovelace",
        "type": "last_first",
        "prob": pytest.approx(0.5),
    }
    assert [entry["type"] for entry in result.loc[2:, "parsed_name"]] == [
        "unknown",
        "unknown",
        "unknown",
    ]


def test_local_artifacts_are_cached(local_models: Path) -> None:
    """Repeated inference reuses the vocabulary and both loaded models."""
    ParseNames.parse(pd.DataFrame({"name": ["Ada", "Ada Lovelace"]}))
    cache_ids = {key: id(value) for key, value in ParseNames._models_cache.items()}

    ParseNames.parse(pd.DataFrame({"name": ["Grace", "Grace Hopper"]}))

    assert ParseNames._vocab_cache
    assert {
        key: id(value) for key, value in ParseNames._models_cache.items()
    } == cache_ids


def test_custom_name_column_preserves_input_and_index(local_models: Path) -> None:
    """The selected column is honored without mutating the caller's frame."""
    frame = pd.DataFrame(
        {"full_name": ["Ada", "Ada Lovelace"], "group": [1, 2]},
        index=pd.Index([10, 20], name="row_id"),
    )

    result = ParseNames.parse(frame, names_col="full_name")

    assert "parsed_name" not in frame.columns
    assert result.index.equals(frame.index)
    assert result["group"].tolist() == [1, 2]
    assert [value["name"] for value in result["parsed_name"]] == [
        "Ada",
        "Ada Lovelace",
    ]


@pytest.mark.parametrize("names_col", ["", None])
def test_invalid_name_column_fails_before_loading_models(names_col) -> None:
    """Invalid selectors fail without resolving remote artifacts."""
    with pytest.raises(ValueError, match="names_col"):
        ParseNames.parse(pd.DataFrame({"name": ["Ada"]}), names_col=names_col)


def test_case_and_batch_invariance(local_models, monkeypatch):
    names = ["JOHN SMITH", "john smith", "John Smith", "JOSÉ", "josé", "Jose"]
    frame = pd.DataFrame({"name": names}, index=[1, 1, 2, 2, 3, 3])
    batch = parse_names(frame)["parsed_name"].tolist()
    monkeypatch.setattr(ParseNames, "batch_size", 1)
    singles = parse_names(frame)["parsed_name"].tolist()
    assert batch == singles
    for group in [batch[:3], batch[3:]]:
        assert len({row["type"] for row in group}) == 1
        assert len({row["prob"] for row in group}) == 1


def test_unsupported_input_does_not_load_models(monkeypatch):
    from unittest.mock import patch

    with patch("parsernaam.naam.resolve_model") as resolve:
        rows = parse_names(pd.DataFrame({"name": ["देवनागरी", "李明", "---", None]}))[
            "parsed_name"
        ]
    resolve.assert_not_called()
    assert all(row["type"] == "unknown" and row["prob"] == 0 for row in rows)


def test_manifest_model_sequence_lengths(local_models):
    from unittest.mock import patch

    from parsernaam.text import encode

    path = local_models / "model_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifacts"]["parsernaam.safetensors"]["seq_len"] = 30
    path.write_text(json.dumps(manifest))
    names = ["John", "John Smith"]
    with patch("parsernaam.naam.encode", wraps=encode) as encoder:
        batch = parse_names(pd.DataFrame({"name": names}))["parsed_name"].tolist()
    assert [row["type"] for row in batch] == ["last", "last_first"]
    assert [call.args[1] for call in encoder.call_args_list] == [30, 47]


def test_evaluator_rejects_corrupt_vocabulary(local_models):
    from training.evaluate import ModelRun

    ModelRun(local_models)
    (local_models / "vocabulary.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact integrity"):
        ModelRun(local_models)


def test_input_sensitive_models_preserve_case_and_batch_results(
    local_models, monkeypatch
):
    from parsernaam._resources import _sha256

    manifest_path = local_models / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for filename in ["parsernaam.safetensors", "parsernaam_pos.safetensors"]:
        torch.manual_seed(391)
        model = LSTM(29, 16, 2, num_layers=1)
        save_file(model.state_dict(), local_models / filename)
        manifest["artifacts"][filename]["sha256"] = _sha256(local_models / filename)
        manifest["artifacts"][filename]["size"] = (
            (local_models / filename).stat().st_size
        )
    manifest_path.write_text(json.dumps(manifest))
    frame = pd.DataFrame(
        {
            "name": [
                "JOHN SMITH",
                "john smith",
                "John Smith",
                "O'CONNOR",
                "o'connor",
                "Grace",
            ]
        }
    )
    batch = parse_names(frame)["parsed_name"].tolist()
    monkeypatch.setattr(ParseNames, "batch_size", 1)
    individual = parse_names(frame)["parsed_name"].tolist()
    assert [row["type"] for row in batch] == [row["type"] for row in individual]
    assert [row["prob"] for row in batch] == pytest.approx(
        [row["prob"] for row in individual], abs=1e-6
    )
    assert [row["prob"] for row in batch[:3]] == pytest.approx(
        [batch[0]["prob"]] * 3, abs=1e-6
    )
    assert batch[3]["prob"] == pytest.approx(batch[4]["prob"], abs=1e-6)
    assert batch[3]["prob"] != pytest.approx(batch[5]["prob"], abs=1e-6)
