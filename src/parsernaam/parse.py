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
    def parse(cls, df: pd.DataFrame, names_col: str = "name") -> pd.DataFrame:
        """Parse names.

        Args:
            df: DataFrame with names.
            names_col: Column containing the name strings.

        Returns:
            DataFrame with parsed names
        """
        return cls._parse_with_models(
            df, cls.MODEL_FN, cls.MODEL_POS_FN, cls.VOCAB_FN, names_col
        )


parse_names = ParseNames.parse


def main(argv: list[str] | None = None) -> int:
    """Parse a Parquet file and write the typed result.

    Args:
        argv: Command-line arguments. Uses ``sys.argv`` when omitted.

    Returns:
        Zero on success.
    """
    description = "Parse names"
    epilog = "Example: parse_names input.parquet -o output.parquet -n name"
    default_out = "output.parquet"
    args = get_args(
        sys.argv[1:] if argv is None else argv, description, epilog, default_out
    )

    df = pd.read_parquet(args.input)
    result = parse_names(df, names_col=args.names_col)
    result.to_parquet(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
