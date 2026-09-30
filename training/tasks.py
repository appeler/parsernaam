"""Source-field labels shared by training and evaluation, never by baselines."""

import numpy as np
import pandas as pd

from .prepare import single_rows


def tasks(records: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Build single-word and balanced ordering tasks from complete source rows."""
    singles = single_rows(records)
    distinct = singles.drop_duplicates(["name", "label"]).copy()
    counts = singles.groupby(["name", "label"]).size()
    distinct["source_count"] = [
        counts[(n, label)]
        for n, label in zip(distinct["name"], distinct["label"], strict=True)
    ]
    ordering = records[records["first"].ne(records["last"])].copy()
    labels = np.arange(len(ordering)) % 2
    np.random.default_rng(seed).shuffle(labels)
    ordering["label"] = np.where(labels == 1, "first_last", "last_first")
    ordering["name"] = np.where(
        labels == 1,
        ordering["first"] + " " + ordering["last"],
        ordering["last"] + " " + ordering["first"],
    )
    ordering["source_name"] = np.where(
        labels == 1,
        ordering["source_first"] + " " + ordering["source_last"],
        ordering["source_last"] + " " + ordering["source_first"],
    )
    return {
        "single_record_weighted": singles,
        "single_distinct_name": distinct,
        "ordering": ordering,
    }
