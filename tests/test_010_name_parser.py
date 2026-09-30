"""Quality checks on real local or published artifacts."""

import pandas as pd
import pytest

from parsernaam import parse_names


@pytest.mark.live
def test_obvious_labeled_names() -> None:
    examples = {
        "John": "first",
        "Michael": "first",
        "Elizabeth": "first",
        "Smith": "last",
        "Johnson": "last",
        "Williams": "last",
        "John Smith": "first_last",
        "Michael Johnson": "first_last",
        "Elizabeth Williams": "first_last",
        "Smith John": "last_first",
        "Johnson Michael": "last_first",
        "Williams Elizabeth": "last_first",
    }
    predictions = parse_names(pd.DataFrame({"name": list(examples)}))["parsed_name"]
    correct = sum(row["type"] == examples[row["name"]] for row in predictions)
    assert correct / len(examples) >= 0.9


@pytest.mark.live
def test_live_case_invariance() -> None:
    names = ["John Smith", "O'Connor", "Jose Gabriel Smith", "Smith Jose Gabriel"]
    variants = [
        variant for name in names for variant in [name, name.upper(), name.lower()]
    ]
    rows = parse_names(pd.DataFrame({"name": variants}))["parsed_name"].tolist()
    for start in range(0, len(rows), 3):
        group = rows[start : start + 3]
        assert len({row["type"] for row in group}) == 1
        assert (
            max(row["prob"] for row in group) - min(row["prob"] for row in group) < 1e-6
        )
