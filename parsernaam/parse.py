#!/usr/bin/env python
"""Public API and CLI entry point for parsing names."""

import sys

import pandas as pd

from .config import ModelConfig
from .naam import Parsernaam
from .utils import get_args


class ParseNames(Parsernaam):
    """Main API class for parsing names using machine learning models.

    This class provides the primary interface for name parsing functionality,
    extending the base Parsernaam class with predefined model file paths.
    Uses LSTM neural networks to classify names as first/last or determine
    positional ordering in multi-word names.

    Example:
        >>> import pandas as pd
        >>> from parsernaam.parse import ParseNames
        >>> df = pd.DataFrame({'name': ['John Smith', 'Kim Yeon']})
        >>> results = ParseNames.parse(df)
        >>> parsed = results['parsed_name'][0]
        >>> parsed['name'], parsed['type']
        ('John Smith', 'first_last')
        >>> parsed['prob'] > 0.5
        True
    """

    MODEL_FN = ModelConfig.MODEL_FILES["single"]
    MODEL_POS_FN = ModelConfig.MODEL_FILES["positional"]
    VOCAB_FN = ModelConfig.MODEL_FILES["vocab"]

    @classmethod
    def parse(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Parse names.

        Args:
            df: DataFrame with names

        Returns:
            DataFrame with parsed names
        """
        return cls._parse_with_models(df, cls.MODEL_FN, cls.MODEL_POS_FN, cls.VOCAB_FN)


parse_names = ParseNames.parse


def main() -> int | None:
    """Main method to parse names.

    Returns:
        Exit code (None for success)
    """
    description = "Parse names"
    epilog = "Example: parsernaam -o output.csv input.csv"
    default_out = "output.csv"
    args = get_args(sys.argv[1:], description, epilog, default_out)

    df = pd.read_csv(args.input, encoding="utf-8")
    df = parse_names(df)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    sys.exit(main())
