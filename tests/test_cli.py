"""Command-line interface contracts."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from parsernaam.parse import main


def test_cli_uses_selected_column_and_typed_parquet(
    tmp_path: Path,
) -> None:
    """The CLI honors --names-col and writes a Parquet result."""
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    source = pd.DataFrame({"full_name": ["Ada"], "group": [3]})
    source.to_parquet(input_path)

    def fake_parse(frame: pd.DataFrame, names_col: str) -> pd.DataFrame:
        assert names_col == "full_name"
        result = frame.copy()
        result["parsed_name"] = [
            {"name": frame.loc[0, names_col], "type": "first", "prob": 0.75}
        ]
        return result

    with patch("parsernaam.parse.parse_names", side_effect=fake_parse):
        status = main(
            [str(input_path), "--output", str(output_path), "--names-col", "full_name"]
        )

    result = pd.read_parquet(output_path)
    assert status == 0
    assert result.loc[0, "full_name"] == "Ada"
    assert result.loc[0, "parsed_name"] == {
        "name": "Ada",
        "prob": 0.75,
        "type": "first",
    }
