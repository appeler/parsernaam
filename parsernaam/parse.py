#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import pandas as pd

from .naam import Parsernaam
from .utils import get_args

class ParseNames(Parsernaam):
    """
    Parse names
    """

    MODEL_FN = "models/parsernaam.pt"
    VOCAB_FN = "models/parsernaam.joblib"

    @classmethod
    def parse(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse names

        Args:
            df: DataFrame with names

        Returns:
            DataFrame with parsed names
        """        
        return super().parse(df, cls.MODEL_FN, cls.VOCAB_FN)
    

parse_names = ParseNames.parse

def main() -> None:
    """
    Main method to parse names
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
