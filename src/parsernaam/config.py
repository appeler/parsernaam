"""Configuration constants for parsernaam.

This module contains all the hardcoded constants used throughout
the parsernaam package, including model parameters, file paths,
and classification categories.
"""

from typing import Final


class ModelConfig:
    """Model configuration constants.

    Contains all the hyperparameters and settings used by the LSTM models
    for name parsing, including architecture parameters and file locations.

    Attributes:
        CATEGORIES_SINGLE: Classification labels for single names
        CATEGORIES_POSITIONAL: Classification labels for multi-word names
        MODEL_FILES: Paths to model and vocabulary files
    """

    # Classification categories for single names (first name only or last name only)
    CATEGORIES_SINGLE: Final[list[str]] = ["last", "first"]

    # Classification categories for multi-word names (position-based)
    CATEGORIES_POSITIONAL: Final[list[str]] = ["last_first", "first_last"]

    # File paths for trained models and vocabulary
    MODEL_FILES: Final[dict[str, str]] = {
        "single": "models/parsernaam.safetensors",  # Single name classifier
        "positional": "models/parsernaam_pos.safetensors",  # Positional classifier
        "vocab": "models/vocabulary.parquet",  # Character vocabulary
    }
