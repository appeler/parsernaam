"""Stream verified voter files into surname-disjoint, record-weighted samples.

Run ``uv run --group train python -m training.prepare --help`` for options.
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from parsernaam.text import normalize

from .common import sha256, surname_split, write_json

logger = logging.getLogger(__name__)
RAW = Path.home() / "Documents/GitHub/ethnicolr/scripts/data-acquisition/raw"


def read_source(path: Path, source: str, limit: int | None = None):
    """Yield schema-checked canonical and original fields with source row IDs."""
    columns = (
        ["name_first", "name_last"] if source == "fl" else ["first_name", "last_name"]
    )
    header = pd.read_csv(path, nrows=0).columns
    if not set(columns) <= set(header):
        raise ValueError(f"{path}: required columns {columns}; got {list(header)}")
    offset = 0
    for chunk in pd.read_csv(
        path,
        usecols=columns,
        dtype=str,
        keep_default_na=False,
        chunksize=250_000,
        nrows=limit,
    ):
        chunk = chunk.rename(
            columns=dict(zip(columns, ["source_first", "source_last"], strict=True))
        )
        chunk["source_row"] = np.arange(offset, offset + len(chunk))
        offset += len(chunk)
        for raw, canonical in [("source_first", "first"), ("source_last", "last")]:
            lookup = {value: normalize(value) for value in chunk[raw].unique()}
            chunk[canonical] = chunk[raw].map(lookup)
        yield chunk


def single_rows(records: pd.DataFrame) -> pd.DataFrame:
    """Extract only one-word source fields, retaining their original rows."""
    frames = []
    for field, label in [("first", "first"), ("last", "last")]:
        selected = records[records[field].str.split().str.len().eq(1)].copy()
        selected["name"] = selected[field]
        selected["source_name"] = selected["source_" + field]
        selected["label"] = label
        frames.append(selected)
    return pd.concat(frames, ignore_index=True)


def prepare(args: argparse.Namespace) -> None:
    """Prepare all training and evaluation inputs without model access."""
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    capacities = {
        "train": args.train_records,
        "val": args.eval_records,
        "calibration": args.eval_records,
        "test": args.eval_records,
    }
    samples = {key: pd.DataFrame() for key in capacities}
    frequencies = {"first": Counter(), "last": Counter()}
    counts = Counter()
    source_firsts, source_lasts = set(), set()
    for chunk in read_source(args.fl, "fl", args.limit):
        counts["fl_total"] += len(chunk)
        valid = chunk[chunk["first"].ne("") & chunk["last"].ne("")].copy()
        counts["fl_invalid"] += len(chunk) - len(valid)
        valid["split"] = valid["last"].map(
            {s: surname_split(s) for s in valid["last"].unique()}
        )
        source_firsts.update(valid["first"])
        source_lasts.update(valid["last"])
        valid["_priority"] = rng.random(len(valid))
        training = valid[valid["split"].eq("train")]
        for field in frequencies:
            frequencies[field].update(
                training.loc[training[field].str.split().str.len().eq(1), field]
            )
        for split, capacity in capacities.items():
            selected = valid[valid["split"].eq(split)]
            counts["fl_" + split] += len(selected)
            samples[split] = pd.concat(
                [samples[split], selected], ignore_index=True
            ).nsmallest(capacity, "_priority")
        logger.info("FL scanned %s records", counts["fl_total"])
    for split, sample in samples.items():
        sample = sample.sort_values("source_row").drop(columns=["_priority", "split"])
        sample.to_parquet(args.output / f"fl_{split}.parquet", index=False)
    words = sorted(frequencies["first"].keys() | frequencies["last"].keys())
    pd.DataFrame(
        {
            "name": words,
            "first_count": [frequencies["first"][w] for w in words],
            "last_count": [frequencies["last"][w] for w in words],
        }
    ).to_parquet(args.output / "frequency.parquet", index=False)
    representatives = {}
    nc_frequencies = Counter()
    nc_sample = pd.DataFrame()
    for chunk in read_source(args.nc, "nc", args.limit):
        counts["nc_total"] += len(chunk)
        valid = chunk[chunk["first"].ne("") & chunk["last"].ne("")].copy()
        counts["nc_invalid"] += len(chunk) - len(valid)
        valid["_priority"] = rng.random(len(valid))
        nc_sample = pd.concat([nc_sample, valid], ignore_index=True).nsmallest(
            args.eval_records, "_priority"
        )
        singles = single_rows(valid)
        nc_frequencies.update(zip(singles["name"], singles["label"], strict=True))
        singles = singles.drop_duplicates(["name", "label"])
        for row in singles.drop(columns=["_priority"]).to_dict("records"):
            representatives.setdefault((row["name"], row["label"]), row)
        logger.info("NC scanned %s records", counts["nc_total"])
    nc_sample.sort_values("source_row").drop(columns="_priority").to_parquet(
        args.output / "nc_records.parquet", index=False
    )
    # One row per canonical name. Conflicting roles are represented by the
    # empirical field share as weights in evaluation, rather than an arbitrary label.
    distinct = pd.DataFrame(representatives.values()).sort_values(["name", "label"])
    distinct["source_count"] = [
        nc_frequencies[(n, label)]
        for n, label in zip(distinct["name"], distinct["label"], strict=True)
    ]
    chosen_names = rng.choice(
        distinct["name"].unique(),
        min(args.eval_records * 2, distinct["name"].nunique()),
        replace=False,
    )
    distinct[distinct["name"].isin(chosen_names)].to_parquet(
        args.output / "nc_distinct.parquet", index=False
    )
    pd.DataFrame({"name": sorted(source_firsts), "field": "first"}).to_parquet(
        args.output / "fl_source_firsts.parquet", index=False
    )
    pd.DataFrame({"name": sorted(source_lasts), "field": "last"}).to_parquet(
        args.output / "fl_source_lasts.parquet", index=False
    )
    write_json(
        args.output / "preparation.json",
        {
            "seed": args.seed,
            "limit": args.limit,
            "counts": dict(counts),
            "capacities": capacities,
            "split": "sha256(normalized surname) mod 100: 80/10/5/5",
            "sources": {
                s: {"path": str(p), "sha256": sha256(p)}
                for s, p in [("fl", args.fl), ("nc", args.nc)]
            },
            "files": {p.name: sha256(p) for p in sorted(args.output.glob("*.parquet"))},
        },
    )


def main() -> None:
    """Prepare data from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fl", type=Path, default=RAW / "fl_reg_name_race_2022.csv.gz")
    parser.add_argument("--nc", type=Path, default=RAW / "nc_voter_name_race.csv.gz")
    parser.add_argument("--output", type=Path, default=Path("training/data"))
    parser.add_argument("--train-records", type=int, default=400_000)
    parser.add_argument("--eval-records", type=int, default=20_000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if (
        args.train_records < 1
        or args.eval_records < 1
        or (args.limit is not None and args.limit < 1)
    ):
        parser.error("sample sizes must be positive")
    logging.basicConfig(level=logging.INFO)
    prepare(args)


if __name__ == "__main__":
    main()
