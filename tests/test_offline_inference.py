"""Offline inference tests with structurally valid deterministic artifacts."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

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
    for filename, categories in (
        ("parsernaam.pt", ModelConfig.CATEGORIES_SINGLE),
        ("parsernaam_pos.pt", ModelConfig.CATEGORIES_POSITIONAL),
    ):
        model = LSTM(
            input_size,
            ModelConfig.HIDDEN_SIZE,
            len(categories),
            num_layers=ModelConfig.NUM_LAYERS,
        )
        for parameter in model.parameters():
            torch.nn.init.zeros_(parameter)
        torch.save(model.state_dict(), tmp_path / filename)

    monkeypatch.setenv("PARSERNAAM_MODEL_DIR", str(tmp_path))
    ParseNames._models_cache.clear()
    ParseNames._vocab_cache = None
    yield tmp_path
    ParseNames._models_cache.clear()
    ParseNames._vocab_cache = None


def test_inference_logic_without_network(local_models: Path) -> None:
    """Single, positional, and invalid names use the correct inference paths."""
    frame = pd.DataFrame({"name": ["Ada", "Ada Lovelace", "", None, 123]})

    result = ParseNames.parse(frame)

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

    assert ParseNames._vocab_cache is not None
    assert {
        key: id(value) for key, value in ParseNames._models_cache.items()
    } == cache_ids
