"""Compare frozen models and independent baselines on identical held-out rows."""

import argparse
import html
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file

from parsernaam.config import ModelConfig
from parsernaam.model import LSTM
from parsernaam.text import encode, normalize

from .common import sha256, verify_preparation, write_json
from .legacy import LegacyLSTM, encode_legacy
from .tasks import tasks

logger = logging.getLogger(__name__)


class ModelRun:
    """Read frozen local weights without modifying the package's manifest."""

    def __init__(self, directory: Path, original: bool = False):
        """Load both artifact-verified models and their encoding metadata."""
        self.directory, self.original = directory, original
        self.manifest = json.loads((directory / "model_manifest.json").read_text())
        for filename, metadata in self.manifest["artifacts"].items():
            artifact = directory / filename
            if (
                artifact.stat().st_size != metadata["size"]
                or sha256(artifact) != metadata["sha256"]
            ):
                raise ValueError(f"artifact integrity mismatch: {artifact}")
        self.vocabulary = "".join(
            pq.read_table(directory / "vocabulary.parquet")["token"].to_pylist()
        )
        self.models = {}
        self.prediction_cache = {}
        self.filenames = [
            next(
                filename
                for filename, metadata in self.manifest["artifacts"].items()
                if metadata.get("labels") == categories
            )
            for categories in [
                ModelConfig.CATEGORIES_SINGLE,
                ModelConfig.CATEGORIES_POSITIONAL,
            ]
        ]
        for filename in self.filenames:
            metadata = self.manifest["artifacts"][filename]
            if sha256(directory / filename) != metadata["sha256"]:
                raise ValueError(f"weight hash mismatch: {directory / filename}")
            model_class = LegacyLSTM if metadata["encoding"] == "legacy" else LSTM
            model = model_class(
                len(self.vocabulary) + 2,
                metadata["hidden_size"],
                2,
                metadata["num_layers"],
            )
            model.load_state_dict(
                torch.load(directory / filename, weights_only=True, map_location="cpu")
                if metadata["encoding"] == "legacy"
                else load_file(directory / filename)
            )
            self.models[filename] = model.eval()

    def predict(
        self, names: list[str], batch_size: int = 512
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the original package transform or the shared fixed transform."""
        canonical = [
            " ".join(name.split()) if self.original else normalize(name)
            for name in names
        ]
        missing = list(
            dict.fromkeys(
                name for name in canonical if name not in self.prediction_cache
            )
        )
        predictions = np.full(len(missing), "unknown", dtype=object)
        scores = np.zeros(len(missing))
        for filename, single in zip(self.filenames, [True, False], strict=True):
            metadata = self.manifest["artifacts"][filename]
            positions = [
                i
                for i, n in enumerate(missing)
                if n and (len(n.split()) == 1) == single
            ]
            for start in range(0, len(positions), batch_size):
                selected = positions[start : start + batch_size]
                encoder = encode_legacy if metadata["encoding"] == "legacy" else encode
                inputs = encoder(
                    [missing[i] for i in selected],
                    30 if self.original else metadata["seq_len"],
                    self.vocabulary,
                )
                with torch.inference_mode():
                    probabilities = self.models[filename](inputs).exp()
                confidence, category = probabilities.max(1)
                predictions[selected] = np.asarray(metadata["labels"])[category.numpy()]
                scores[selected] = confidence.numpy()
        self.prediction_cache.update(
            zip(missing, zip(predictions, scores, strict=True), strict=True)
        )
        values = [self.prediction_cache[name] for name in canonical]
        return np.array([label for label, _ in values], dtype=object), np.array(
            [score for _, score in values]
        )


class Frequency:
    """Independent FL source-field frequency baseline using word endpoints."""

    def __init__(self, table: pd.DataFrame):
        """Use only frozen training counts with no fitted neural components."""
        self.probabilities = dict(
            zip(
                table["name"],
                table["first_count"] / (table["first_count"] + table["last_count"]),
                strict=True,
            )
        )
        self.prior = float(
            table["first_count"].sum()
            / (table["first_count"].sum() + table["last_count"].sum())
        )

    def predict(
        self, names: list[str], ordering: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compare first-name field probabilities, using the training prior for OOV."""
        probabilities = []
        for name in names:
            words = name.split()
            left = self.probabilities.get(words[0], self.prior)
            if ordering:
                right = self.probabilities.get(words[-1], self.prior)
                forward, reverse = left * (1 - right), right * (1 - left)
                probability = (
                    forward / (forward + reverse) if forward + reverse else 0.5
                )
            else:
                probability = left
            probabilities.append(probability)
        values = np.asarray(probabilities)
        labels = ("last_first", "first_last") if ordering else ("last", "first")
        return np.where(values > 0.5, labels[1], labels[0]), np.maximum(
            values, 1 - values
        )


def metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    weights: np.ndarray,
    clusters: np.ndarray | None = None,
) -> dict:
    """Report weighted accuracy, reliability, ECE, Brier score and log loss."""
    correct = (labels == predictions).astype(float)
    bands = []
    ece = 0.0
    for band in range(10):
        mask = (scores >= band / 10) & (
            (scores < (band + 1) / 10) if band < 9 else (scores <= 1)
        )
        mass = float(weights[mask].sum())
        if mass:
            confidence = float(np.average(scores[mask], weights=weights[mask]))
            accuracy = float(np.average(correct[mask], weights=weights[mask]))
            ece += mass / weights.sum() * abs(confidence - accuracy)
            bands.append(
                {
                    "lower": band / 10,
                    "upper": (band + 1) / 10,
                    "rows": int(mask.sum()),
                    "weight": mass,
                    "score": confidence,
                    "accuracy": accuracy,
                }
            )
    true_probability = np.where(correct == 1, scores, 1 - scores)
    true_probability = np.where(predictions == "unknown", 0.5, true_probability)
    if clusters is None:
        clusters = np.arange(len(labels))
    groups, _ = pd.factorize(clusters)
    cluster_mass = np.bincount(groups, weights=weights)
    cluster_correct = np.bincount(groups, weights=weights * correct)
    rng = np.random.default_rng(20260929)
    bootstrap = []
    for _ in range(200):
        selected = rng.integers(0, len(cluster_mass), len(cluster_mass))
        bootstrap.append(cluster_correct[selected].sum() / cluster_mass[selected].sum())
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "accuracy_ci_lower": float(lower),
        "accuracy_ci_upper": float(upper),
        "bootstrap_clusters": len(cluster_mass),
        "rows": len(labels),
        "effective_names_or_records": float(weights.sum()),
        "accuracy": float(np.average(correct, weights=weights)),
        "ece": ece,
        "brier": float(np.average((1 - true_probability) ** 2, weights=weights)),
        "log_loss": float(
            np.average(-np.log(np.clip(true_probability, 1e-7, 1)), weights=weights)
        ),
        "coverage": float(np.average(predictions != "unknown", weights=weights)),
        "calibration": bands,
    }


def evaluate(args: argparse.Namespace) -> None:
    """Write the complete comparison, paired source samples, and release gates."""
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    preparation = verify_preparation(args.data)
    runs = {
        "published": ModelRun(args.published, True),
        "fixed_inference": ModelRun(args.published),
        "retrained": ModelRun(args.models),
    }
    if runs["retrained"].manifest["training"]["preparation_sha256"] != sha256(
        args.data / "preparation.json"
    ):
        raise ValueError("evaluation data do not match the model's frozen preparation")
    split_surnames = {
        split: set(
            pd.read_parquet(args.data / f"fl_{split}.parquet", columns=["last"])["last"]
        )
        for split in ["train", "val", "calibration", "test"]
    }
    for split, surnames in split_surnames.items():
        for other, other_surnames in split_surnames.items():
            if split != other and not surnames.isdisjoint(other_surnames):
                raise ValueError("surname leakage between FL splits")
    baseline = Frequency(pd.read_parquet(args.data / "frequency.parquet"))
    train = tasks(
        pd.read_parquet(args.data / "fl_train.parquet"),
        runs["retrained"].manifest["training"]["parameters"]["seed"],
    )
    majorities = {}
    for task in ["single_record_weighted", "ordering"]:
        counts = train[task]["label"].value_counts(normalize=True)
        majorities["ordering" if task == "ordering" else "single"] = (
            counts.index[0],
            float(counts.iloc[0]),
        )
    lookup_seen = set(baseline.probabilities)
    model_seen = {key: set(frame["name"]) for key, frame in train.items()}
    source_last_names = set(
        pd.read_parquet(args.data / "fl_source_lasts.parquet")["name"]
    )
    results, samples = [], []
    case_violations = 0
    for source, filename in [("FL", "fl_test.parquet"), ("NC", "nc_records.parquet")]:
        source_records = pd.read_parquet(args.data / filename)
        task_frames = tasks(source_records, args.seed + 2)
        if source == "NC":
            task_frames["single_distinct_name"] = pd.read_parquet(
                args.data / "nc_distinct.parquet"
            )
        for task, frame in task_frames.items():
            frame = frame.copy()
            ordering = task == "ordering"
            if task == "single_distinct_name":
                frame["weight"] = frame["source_count"] / frame.groupby("name")[
                    "source_count"
                ].transform("sum")
            else:
                frame["weight"] = 1.0
            frame["word_count"] = (
                np.where(frame["name"].str.split().str.len().eq(2), "2", "3+")
                if ordering
                else "1"
            )
            frame["lookup_supported"] = (
                frame["name"].isin(lookup_seen)
                if not ordering
                else frame["name"].map(
                    lambda n: (
                        n.split()[0] in lookup_seen and n.split()[-1] in lookup_seen
                    )
                )
            )
            frame["model_seen"] = frame["name"].isin(
                model_seen["ordering" if ordering else "single_record_weighted"]
            )
            frame["surname_seen_in_fl_source"] = frame["last"].isin(source_last_names)
            for case in ["as_is", "lowercased"]:
                names = (
                    frame["source_name"].tolist()
                    if case == "as_is"
                    else frame["source_name"].str.lower().tolist()
                )
                outputs = {
                    name: run.predict(names, args.batch_size)
                    for name, run in runs.items()
                }
                canonical = frame["name"].tolist()
                outputs["frequency"] = baseline.predict(canonical, ordering)
                label, score = majorities["ordering" if ordering else "single"]
                outputs["majority"] = (
                    np.full(len(frame), label),
                    np.full(len(frame), score),
                )
                for name, (predictions, scores) in outputs.items():
                    frame[name + "_label"] = predictions
                    frame[name + "_score"] = scores
                    masks = {
                        "all": np.ones(len(frame), dtype=bool),
                        "lookup_supported": frame["lookup_supported"].to_numpy(),
                        "lookup_unsupported": ~frame["lookup_supported"].to_numpy(),
                        "model_seen": frame["model_seen"].to_numpy(),
                        "model_unseen": ~frame["model_seen"].to_numpy(),
                    }
                    if ordering:
                        masks.update(
                            {
                                word_count: frame["word_count"]
                                .eq(word_count)
                                .to_numpy()
                                for word_count in ["2", "3+"]
                            }
                        )
                    for subgroup, mask in masks.items():
                        if not mask.any():
                            continue
                        results.append(
                            {
                                "source": source,
                                "task": task,
                                "case": case,
                                "subgroup": subgroup,
                                "run": name,
                                **metrics(
                                    frame["label"].to_numpy()[mask],
                                    predictions[mask],
                                    scores[mask],
                                    frame["weight"].to_numpy()[mask],
                                    frame[
                                        "name"
                                        if task == "single_distinct_name"
                                        else "last"
                                    ].to_numpy()[mask],
                                ),
                            }
                        )
                if case == "as_is":
                    for name in ["fixed_inference", "retrained"]:
                        title = runs[name].predict(canonical, args.batch_size)
                        lower = runs[name].predict(
                            [n.lower() for n in canonical], args.batch_size
                        )
                        case_violations += int(
                            np.count_nonzero(outputs[name][0] != title[0])
                        ) + int(np.count_nonzero(title[0] != lower[0]))
                        if not np.allclose(
                            outputs[name][1], title[1], atol=1e-6
                        ) or not np.allclose(title[1], lower[1], atol=1e-6):
                            raise ValueError(f"{name}: case-dependent scores")
                    sample = frame.sample(
                        min(20, len(frame)), random_state=args.seed
                    ).copy()
                    sample["source"], sample["task"], sample["case"] = (
                        source,
                        task,
                        case,
                    )
                    samples.append(sample)
            logger.info("completed %s %s", source, task)
    table = pd.DataFrame(
        [{k: v for k, v in row.items() if k != "calibration"} for row in results]
    )
    primary = table[table["case"].eq("as_is") & table["subgroup"].eq("all")]
    nc = primary[primary["source"].eq("NC")].pivot(
        index="task", columns="run", values="accuracy"
    )
    gates = {
        "case_invariance": case_violations == 0,
        "case_violations": case_violations,
        "retrained_beats_both_baselines": bool(
            (
                (nc["retrained"] > nc["majority"]) & (nc["retrained"] > nc["frequency"])
            ).all()
        ),
        "fixed_beats_published": bool((nc["fixed_inference"] > nc["published"]).all()),
    }
    report = {
        "gates": gates,
        "results": results,
        "provenance": {
            "seed": args.seed,
            "preparation": {
                **preparation,
                "sources": {
                    source: {**metadata, "path": Path(metadata["path"]).name}
                    for source, metadata in preparation["sources"].items()
                },
            },
            "code_sha256": {
                str(p): sha256(p)
                for p in sorted(
                    [
                        *Path("training").glob("*.py"),
                        *Path("src/parsernaam").glob("*.py"),
                    ]
                )
            },
            "models": {
                name: {k: v for k, v in run.manifest.items() if k != "evaluation"}
                for name, run in runs.items()
            },
        },
        "limitations": [
            "Published/fixed FL comparisons are contaminated: original published models used FL and leaked splits.",
            "NC registrations are from a different state; common name strings can overlap FL and Census. This is geographic transfer, not an unseen-name test.",
            "Surnames are disjoint across FL splits; first names and cross-role words can overlap.",
            "Lookup uses counts from all FL train-assigned source records; LSTMs use the seeded record sample. Model-seen means an exact canonical task input appeared in that sample.",
            "Single distinct-name view weights each canonical name once, distributing weight across observed source roles by their field frequency.",
            "Ordering excludes identical canonical first/last fields; ordering labels for other compound fields can still be textually ambiguous.",
            "No score calibration was fitted; reliability and ECE describe raw model scores.",
        ],
    }
    write_json(args.output / "evaluation.json", report)
    table.to_csv(args.output / "metrics.csv", index=False)
    paired = pd.concat(samples, ignore_index=True)
    paired.to_parquet(args.output / "samples.parquet", index=False)
    columns = ["majority", "frequency", "published", "fixed_inference", "retrained"]
    accuracy = primary.pivot(
        index=["source", "task"], columns="run", values="accuracy"
    )[columns]
    reliability = primary.pivot(index=["source", "task"], columns="run", values="ece")[
        columns
    ]

    def markdown_table(frame: pd.DataFrame, percent: bool) -> str:
        display = frame.reset_index().copy()
        for column in columns:
            display[column] = display[column].map(
                lambda value: f"{value * 100:.1f}%" if percent else f"{value:.3f}"
            )
        heading = (
            "| "
            + " | ".join(display.columns)
            + " |\n| "
            + " | ".join(["---"] * len(display.columns))
            + " |\n"
        )
        return (
            heading
            + "\n".join(
                "| " + " | ".join(map(str, row)) + " |"
                for row in display.itertuples(index=False, name=None)
            )
            + "\n"
        )

    markdown = "Accuracy on as-is input (NC is all caps):\n\n" + markdown_table(
        accuracy, True
    )
    markdown += (
        "\nExpected calibration error of raw scores (lower is better):\n\n"
        + markdown_table(reliability, False)
    )
    markdown += (
        "\nRelease gates: "
        + ", ".join(f"{key}={value}" for key, value in gates.items())
        + ".\n"
    )
    (args.output / "summary.md").write_text(markdown)
    readme = args.readme.read_text()
    begin, end = "<!-- evaluation:start -->", "<!-- evaluation:end -->"
    if readme.count(begin) != 1 or readme.count(end) != 1:
        raise ValueError("README requires exactly one evaluation marker pair")
    before, remainder = readme.split(begin)
    _, after = remainder.split(end)
    args.readme.write_text(before + begin + "\n" + markdown + end + after)
    calibration_rows = [
        {k: row[k] for k in ["source", "task", "case", "subgroup", "run"]} | band
        for row in results
        for band in row["calibration"]
    ]
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(args.output / "calibration.csv", index=False)
    body = (
        "<h1>parsernaam held-out evaluation</h1><p>"
        + html.escape(json.dumps(gates))
        + "</p><ul>"
        + "".join("<li>" + html.escape(x) + "</li>" for x in report["limitations"])
        + "</ul>"
    )
    body += "<h2>Accuracy on as-is input</h2>" + accuracy.to_html(
        float_format=lambda value: f"{value:.2%}"
    )
    body += "<h2>Expected calibration error</h2>" + reliability.to_html(
        float_format=lambda value: f"{value:.3f}"
    )
    body += (
        "<details><summary>All comparisons and support breakdowns</summary>"
        + table.to_html(index=False, float_format=lambda v: f"{v:.4f}")
    )
    body += (
        "</details><details><summary>Reliability by score band</summary>"
        + calibration.to_html(index=False, float_format=lambda v: f"{v:.4f}")
    )
    body += (
        "</details><h2>20 source-linked samples per task and state</h2>"
        + paired.to_html(index=False, float_format=lambda v: f"{v:.4f}")
    )
    (args.output / "evaluation.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>parsernaam evaluation</title><style>body{font:14px system-ui;margin:32px}table{border-collapse:collapse;margin-bottom:32px}td,th{padding:6px;border:1px solid #ddd}th{background:#edf2f7}tr:nth-child(even){background:#fafafa}</style>'
        + body
        + "</html>"
    )
    manifest_path = args.models / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["evaluation"] = {
        "report_sha256": sha256(args.output / "evaluation.json"),
        "gates": gates,
        "nc_accuracy": nc.to_dict(),
    }
    write_json(manifest_path, manifest)
    logger.info("release gates: %s", gates)


def main() -> None:
    """Run the frozen held-out comparison from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("training/data"))
    parser.add_argument("--models", type=Path, default=Path("training/artifacts"))
    parser.add_argument(
        "--published", type=Path, default=Path("training/runs/published")
    )
    parser.add_argument("--output", type=Path, default=Path("training/reports"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    logging.basicConfig(level=logging.INFO)
    evaluate(args)


if __name__ == "__main__":
    main()
