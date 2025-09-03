#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from typing import Optional
import pandas as pd

from .naam import Parsernaam
from .utils import get_args
from .config import ModelConfig


class ParseNames(Parsernaam):
    """
    Main API class for parsing names using machine learning models.
    
    This class provides the primary interface for name parsing functionality,
    extending the base Parsernaam class with predefined model file paths.
    Uses LSTM neural networks to classify names as first/last or determine
    positional ordering in multi-word names.
    
    Example:
        >>> import pandas as pd
        >>> from parsernaam.parse import ParseNames
        >>> df = pd.DataFrame({'name': ['John Smith', 'Kim Yeon']})
        >>> results = ParseNames.parse(df)
        >>> print(results['parsed_name'][0])
        {'name': 'John Smith', 'type': 'first_last', 'prob': 0.998}
    """

    MODEL_FN = ModelConfig.MODEL_FILES['single']
    MODEL_POS_FN = ModelConfig.MODEL_FILES['positional'] 
    VOCAB_FN = ModelConfig.MODEL_FILES['vocab']

    @classmethod
    def parse(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse names

        Args:
            df: DataFrame with names

        Returns:
            DataFrame with parsed names
        """
        return super().parse(df, cls.MODEL_FN , cls.MODEL_POS_FN, cls.VOCAB_FN)
    

parse_names = ParseNames.parse


def main() -> Optional[int]:
    """
    Main method to parse names
    
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
