#!/usr/bin/env python
"""Core ML inference pipeline for parsing names."""

import logging
from importlib import resources
from typing import ClassVar, TypedDict

import joblib
import pandas as pd
import torch

from .config import ModelConfig
from .model import LSTM

# Configure logging
logger = logging.getLogger(__name__)


class ParsedNameResult(TypedDict):
    """Structure for parsed name result."""

    name: str
    type: str
    prob: float


class VocabCache(TypedDict):
    """Structure for vocabulary cache."""

    vocab: list[str]
    all_letters: str
    n_letters: int


class Parsernaam:
    """Parse names."""

    _models_cache: ClassVar[dict[str, dict[str, torch.nn.Module]]] = {}
    _vocab_cache: ClassVar[VocabCache | None] = None

    @classmethod
    def _parse_with_models(
        cls, df: pd.DataFrame, model_fn: str, model_fn_pos: str, vocab_fn: str
    ) -> pd.DataFrame:
        """Parse names using ML models.

        Args:
            df: DataFrame with 'name' column containing names to parse
            model_fn: Path to single name model file
            model_fn_pos: Path to positional name model file
            vocab_fn: Path to vocabulary file

        Returns:
            DataFrame with added 'parsed_name' column

        Raises:
            ValueError: If required 'name' column is missing
        """
        logger.info(f"Starting name parsing for DataFrame with {len(df)} rows")

        if not isinstance(df, pd.DataFrame):
            logger.error("Input is not a pandas DataFrame")
            raise ValueError("Input must be a pandas DataFrame")

        if "name" not in df.columns:
            logger.error("DataFrame missing required 'name' column")
            raise ValueError("DataFrame must contain 'name' column")

        if df.empty:
            logger.info("Empty DataFrame provided, returning empty result")
            result_df = df.copy()
            result_df["parsed_name"] = []
            return result_df
        model_path = resources.files("parsernaam") / model_fn
        model_pos_path = resources.files("parsernaam") / model_fn_pos
        vocab_path = resources.files("parsernaam") / vocab_fn

        # Load vocabulary with caching
        if cls._vocab_cache is None:
            logger.info(f"Loading vocabulary from {vocab_path}")
            vectorizer = joblib.load(vocab_path)
            vocabulary_list = list(vectorizer.get_feature_names_out())
            cls._vocab_cache = {
                "vocab": vocabulary_list,
                "all_letters": "".join(vocabulary_list),
                "n_letters": len(vocabulary_list),
            }
            logger.info(f"Loaded vocabulary with {len(vocabulary_list)} characters")

        all_letters_string = cls._vocab_cache["all_letters"]
        vocabulary_size = cls._vocab_cache["n_letters"]
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

        # Initialize models with caching
        cache_key = f"{model_path}_{model_pos_path}_{device}"
        if cache_key not in cls._models_cache:
            logger.info(f"Loading models on device: {device}")

            logger.debug(f"Loading single name model from {model_path}")
            single_name_model = LSTM(
                embedding_vocab_size,
                hidden_layer_size,
                num_single_categories,
                num_layers=ModelConfig.NUM_LAYERS,
            )
            single_name_model.load_state_dict(
                torch.load(str(model_path), map_location=device, weights_only=True)
            )
            single_name_model.to(device)
            single_name_model.eval()

            logger.debug(f"Loading positional name model from {model_pos_path}")
            positional_name_model = LSTM(
                embedding_vocab_size,
                hidden_layer_size,
                len(positional_categories),
                num_layers=ModelConfig.NUM_LAYERS,
            )
            positional_name_model.load_state_dict(
                torch.load(str(model_pos_path), map_location=device, weights_only=True)
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
            tensor = torch.ones(sequence_length) * out_of_bounds_token
            try:
                for position_index, character in enumerate(name_string):
                    if position_index >= sequence_length:
                        break
                    tensor[position_index] = get_character_index(character)
            except (IndexError, ValueError, TypeError):
                pass
            return tensor

        def parse_single_name(name_input: str | None) -> ParsedNameResult:
            if not isinstance(name_input, str) or not name_input.strip():
                return {
                    "name": str(name_input) if name_input is not None else "",
                    "type": "unknown",
                    "prob": 0.0,
                }

            name_parts = name_input.split()
            name_tensor = convert_name_to_tensor(name_input)

            with torch.no_grad():
                # Use single name model for one word, positional model for multiple words
                if len(name_parts) == 1:
                    model_output = single_name_model(
                        name_tensor.unsqueeze(0).to(device)
                    )
                    probabilities = torch.exp(model_output)
                    predicted_class_index = torch.argmax(probabilities)
                    predicted_name_type = single_name_categories[
                        int(predicted_class_index.item())
                    ]
                else:
                    model_output = positional_name_model(
                        name_tensor.unsqueeze(0).to(device)
                    )
                    probabilities = torch.exp(model_output)
                    predicted_class_index = torch.argmax(probabilities)
                    predicted_name_type = positional_categories[
                        int(predicted_class_index.item())
                    ]
                return {
                    "name": name_input,
                    "type": predicted_name_type,
                    "prob": probabilities[0][predicted_class_index].item(),
                }

        logger.info("Applying name parsing to all rows")
        df["parsed_name"] = df["name"].apply(parse_single_name)
        logger.info(f"Name parsing completed for {len(df)} rows")
        return df
