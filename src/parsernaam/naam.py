"""Core ML inference pipeline for parsing names."""

import logging
from typing import ClassVar, TypedDict

import pandas as pd
import pyarrow.parquet as pq
import torch

from ._resources import resolve_model
from .config import ModelConfig
from .model import LSTM

logger = logging.getLogger(__name__)


class ParsedNameResult(TypedDict):
    """Structure for parsed name result."""

    name: str
    type: str
    prob: float


class VocabCache(TypedDict):
    """Structure for vocabulary cache."""

    all_letters: str
    n_letters: int


class Parsernaam:
    """Parse names."""

    _models_cache: ClassVar[dict[str, dict[str, torch.nn.Module]]] = {}
    _vocab_cache: ClassVar[dict[str, VocabCache]] = {}

    @classmethod
    def _parse_with_models(
        cls,
        df: pd.DataFrame,
        model_fn: str,
        model_fn_pos: str,
        vocab_fn: str,
        names_col: str,
    ) -> pd.DataFrame:
        """Parse names using ML models.

        Args:
            df: DataFrame containing names to parse.
            model_fn: Path to single name model file
            model_fn_pos: Path to positional name model file
            vocab_fn: Path to vocabulary file
            names_col: Column containing the name strings.

        Returns:
            DataFrame with added 'parsed_name' column

        Raises:
            ValueError: If the input or requested name column is invalid.
        """
        if not isinstance(df, pd.DataFrame):
            logger.error("Input is not a pandas DataFrame")
            raise ValueError("Input must be a pandas DataFrame")

        if not isinstance(names_col, str) or not names_col:
            raise ValueError("names_col must be a non-empty string")
        if names_col not in df.columns:
            logger.error("DataFrame missing requested name column: %s", names_col)
            raise ValueError(f"DataFrame must contain {names_col!r} column")

        logger.info("Starting name parsing for DataFrame with %d rows", len(df))
        result_df = df.copy()
        if df.empty:
            logger.info("Empty DataFrame provided, returning empty result")
            result_df["parsed_name"] = pd.Series(index=result_df.index, dtype=object)
            return result_df
        model_path = resolve_model(model_fn)
        model_pos_path = resolve_model(model_fn_pos)
        vocab_path = resolve_model(vocab_fn)

        if vocab_path not in cls._vocab_cache:
            logger.info("Loading vocabulary from %s", vocab_path)
            vocabulary_list = pq.read_table(vocab_path, columns=["token"])[
                "token"
            ].to_pylist()
            cls._vocab_cache[vocab_path] = {
                "all_letters": "".join(vocabulary_list),
                "n_letters": len(vocabulary_list),
            }
            logger.info("Loaded vocabulary with %d characters", len(vocabulary_list))

        vocabulary = cls._vocab_cache[vocab_path]
        all_letters_string = vocabulary["all_letters"]
        vocabulary_size = vocabulary["n_letters"]
        out_of_bounds_token = vocabulary_size + 1

        positional_categories = ModelConfig.CATEGORIES_POSITIONAL
        single_name_categories = ModelConfig.CATEGORIES_SINGLE
        num_single_categories = len(single_name_categories)

        hidden_layer_size = ModelConfig.HIDDEN_SIZE
        sequence_length = ModelConfig.SEQUENCE_LENGTH
        embedding_vocab_size = (
            vocabulary_size + 2
        )  # vocabulary + out_of_bounds + padding
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        cache_key = f"{model_path}_{model_pos_path}_{vocab_path}_{device}"
        if cache_key not in cls._models_cache:
            logger.info("Loading models on device: %s", device)

            logger.debug("Loading single name model from %s", model_path)
            single_name_model = LSTM(
                embedding_vocab_size,
                hidden_layer_size,
                num_single_categories,
                num_layers=ModelConfig.NUM_LAYERS,
            )
            single_name_model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
            single_name_model.to(device)
            single_name_model.eval()

            logger.debug("Loading positional name model from %s", model_pos_path)
            positional_name_model = LSTM(
                embedding_vocab_size,
                hidden_layer_size,
                len(positional_categories),
                num_layers=ModelConfig.NUM_LAYERS,
            )
            positional_name_model.load_state_dict(
                torch.load(model_pos_path, map_location=device, weights_only=True)
            )
            positional_name_model.to(device)
            positional_name_model.eval()

            cls._models_cache[cache_key] = {
                "single_name_model": single_name_model,
                "positional_name_model": positional_name_model,
            }
            logger.info("Models loaded and cached successfully")
        else:
            logger.debug("Using cached models")

        single_name_model = cls._models_cache[cache_key]["single_name_model"]
        positional_name_model = cls._models_cache[cache_key]["positional_name_model"]

        def get_character_index(character: str) -> int:
            character_index = all_letters_string.find(character)
            return character_index if character_index != -1 else out_of_bounds_token

        def convert_name_to_tensor(name_string: str) -> torch.Tensor:
            tensor = torch.full(
                (sequence_length,), out_of_bounds_token, dtype=torch.long
            )
            for position_index, character in enumerate(name_string[:sequence_length]):
                tensor[position_index] = get_character_index(character)
            return tensor

        def parse_single_name(name_input: str | None) -> ParsedNameResult:
            if not isinstance(name_input, str) or not name_input.strip():
                return {
                    "name": str(name_input) if name_input is not None else "",
                    "type": "unknown",
                    "prob": 0.0,
                }

            name_parts = name_input.split()
            name_tensor = convert_name_to_tensor(" ".join(name_parts))

            with torch.no_grad():
                if len(name_parts) == 1:
                    model_output = single_name_model(
                        name_tensor.unsqueeze(0).to(device)
                    )
                    probabilities = torch.exp(model_output)
                    predicted_class_index = int(
                        torch.argmax(probabilities, dim=1).item()
                    )
                    predicted_name_type = single_name_categories[predicted_class_index]
                else:
                    model_output = positional_name_model(
                        name_tensor.unsqueeze(0).to(device)
                    )
                    probabilities = torch.exp(model_output)
                    predicted_class_index = int(
                        torch.argmax(probabilities, dim=1).item()
                    )
                    predicted_name_type = positional_categories[predicted_class_index]
                return {
                    "name": name_input,
                    "type": predicted_name_type,
                    "prob": probabilities[0, predicted_class_index].item(),
                }

        logger.info("Applying name parsing to all rows")
        result_df["parsed_name"] = result_df[names_col].apply(parse_single_name)
        logger.info("Name parsing completed for %d rows", len(result_df))
        return result_df
