"""Batched ML inference for first/last name patterns."""

import json
from pathlib import Path
from typing import Any, ClassVar, TypedDict

import pandas as pd
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file

from ._resources import MODEL_MANIFEST, resolve_model
from .model import LSTM
from .text import encode, normalize


class ParsedNameResult(TypedDict):
    """One prediction, preserving the original input string."""

    name: str
    type: str
    prob: float


class Parsernaam:
    """Parse names with shared normalization and manifest-defined models."""

    _models_cache: ClassVar[dict[str, LSTM]] = {}
    _vocab_cache: ClassVar[dict[str, str]] = {}
    batch_size: ClassVar[int] = 512

    @classmethod
    def _parse_with_models(
        cls,
        df: pd.DataFrame,
        model_fn: str,
        model_fn_pos: str,
        vocab_fn: str,
        names_col: str,
    ) -> pd.DataFrame:
        """Classify normalized inputs in batches.

        Args:
            df: Input frame.
            model_fn: Single-word weights.
            model_fn_pos: Ordering weights.
            vocab_fn: Vocabulary artifact.
            names_col: Name column.

        Returns:
            Copied frame with one prediction dict per row.

        Raises:
            ValueError: If the input or name column is invalid.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if not isinstance(names_col, str) or not names_col:
            raise ValueError("names_col must be a non-empty string")
        if names_col not in df.columns:
            raise ValueError(f"DataFrame must contain {names_col!r} column")
        result = df.copy()
        originals = df[names_col].tolist()
        names = [
            normalize(value) if isinstance(value, str) else "" for value in originals
        ]
        predictions: list[ParsedNameResult] = [
            {
                "name": value
                if isinstance(value, str)
                else ""
                if value is None
                else str(value),
                "type": "unknown",
                "prob": 0.0,
            }
            for value in originals
        ]
        if any(names):
            vocabulary_path = resolve_model(vocab_fn)
            if vocabulary_path not in cls._vocab_cache:
                cls._vocab_cache[vocabulary_path] = "".join(
                    pq.read_table(vocabulary_path, columns=["token"])[
                        "token"
                    ].to_pylist()
                )
            vocabulary = cls._vocab_cache[vocabulary_path]
            for filename, single in ((model_fn, True), (model_fn_pos, False)):
                rows = [
                    i
                    for i, name in enumerate(names)
                    if name and (len(name.split()) == 1) == single
                ]
                if not rows:
                    continue
                path = resolve_model(filename)
                local_manifest = Path(path).parent / "model_manifest.json"
                manifest = (
                    json.loads(local_manifest.read_text())
                    if local_manifest.is_file()
                    else MODEL_MANIFEST
                )
                metadata: dict[str, Any] = manifest["artifacts"][Path(path).name]
                cache_key = f"{path}:{json.dumps(metadata, sort_keys=True)}"
                if cache_key not in cls._models_cache:
                    model = LSTM(
                        len(vocabulary) + 2,
                        metadata["hidden_size"],
                        len(metadata["labels"]),
                        metadata["num_layers"],
                    )
                    model.load_state_dict(load_file(path))
                    cls._models_cache[cache_key] = model.eval()
                model = cls._models_cache[cache_key]
                for start in range(0, len(rows), cls.batch_size):
                    batch_rows = rows[start : start + cls.batch_size]
                    batch_names = [names[i] for i in batch_rows]
                    tokens = encode(batch_names, metadata["seq_len"], vocabulary)
                    with torch.inference_mode():
                        probabilities = model(tokens).exp()
                    scores, classes = probabilities.max(dim=1)
                    for i, score, category in zip(
                        batch_rows, scores.tolist(), classes.tolist(), strict=True
                    ):
                        predictions[i]["type"] = metadata["labels"][category]
                        predictions[i]["prob"] = score
        result["parsed_name"] = pd.Series(predictions, index=result.index, dtype=object)
        return result
