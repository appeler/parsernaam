"""Verify data separation and independent evaluation mathematics."""

import argparse
import json
import shutil
from itertools import product

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import load_file

from training.common import surname_split
from training.evaluate import Frequency, metrics
from training.prepare import prepare, read_source
from training.tasks import tasks


def test_surname_splits_and_source_provenance(tmp_path):
    surnames = {}
    for letters in product("abcdefghijklmnopqrstuvwxyz", repeat=2):
        name = "Surname" + "".join(letters)
        surnames.setdefault(surname_split(name.title()), name)
        if len(surnames) == 4:
            break
    fl = tmp_path / "fl.csv.gz"
    nc = tmp_path / "nc.csv.gz"
    records = pd.DataFrame(
        {
            "name_first": ["José", "Jose Gabriel"] * 4,
            "name_last": [name for name in surnames.values() for _ in range(2)],
        }
    )
    records.to_csv(fl, index=False)
    pd.DataFrame(
        {
            "first_name": ["JOSE", "JOSE GABRIEL", "SMITH"],
            "last_name": ["SMITH", "SMITH", "JOSE"],
        }
    ).to_csv(nc, index=False)
    output = tmp_path / "prepared"
    prepare(
        argparse.Namespace(
            fl=fl,
            nc=nc,
            output=output,
            train_records=10,
            eval_records=10,
            seed=17,
            limit=None,
        )
    )
    splits = {s: pd.read_parquet(output / f"fl_{s}.parquet") for s in surnames}
    for split, frame in splits.items():
        assert all(surname_split(name) == split for name in frame["last"])
        assert set(frame["source_row"]) <= set(range(8))
        assert set(frame["first"]) == {"Jose", "Jose Gabriel"}
    for left, right in product(splits, repeat=2):
        if left != right:
            assert set(splits[left]["last"]).isdisjoint(splits[right]["last"])
    lookup = pd.read_parquet(output / "frequency.parquet").set_index("name")
    assert lookup["first_count"].sum() == 1
    assert lookup["last_count"].sum() == 2
    assert "Jose Gabriel" not in lookup.index
    distinct = pd.read_parquet(output / "nc_distinct.parquet")
    assert set(distinct["name"]) == {"Jose", "Smith"}
    assert distinct.groupby("name")["source_count"].sum().to_dict() == {
        "Jose": 2,
        "Smith": 3,
    }
    rerun = tmp_path / "second"
    prepare(
        argparse.Namespace(
            fl=fl,
            nc=nc,
            output=rerun,
            train_records=10,
            eval_records=10,
            seed=17,
            limit=None,
        )
    )
    for path in output.glob("*.parquet"):
        assert path.read_bytes() == (rerun / path.name).read_bytes()
    from training.common import sha256, verify_preparation
    from training.evaluate import evaluate
    from training.train import train

    models = tmp_path / "models"
    train(
        argparse.Namespace(
            data=output,
            output=models,
            device="cpu",
            seed=17,
            seq_len=20,
            hidden_size=8,
            layers=1,
            batch_size=4,
            epochs=1,
            patience=1,
            learning_rate=0.001,
        )
    )
    published = tmp_path / "published"
    published.mkdir()
    for filename in ["vocabulary.parquet", "model_manifest.json"]:
        shutil.copyfile(models / filename, published / filename)
    manifest = json.loads((published / "model_manifest.json").read_text())
    for filename, length in [
        ("parsernaam.safetensors", 30),
        ("parsernaam_pos.safetensors", 47),
    ]:
        old_filename = filename.replace(".safetensors", ".pt")
        torch.save(load_file(models / filename), published / old_filename)
        metadata = manifest["artifacts"].pop(filename)
        metadata.update(
            encoding="legacy",
            format="pytorch_state_dict",
            seq_len=length,
            sha256=sha256(published / old_filename),
            size=(published / old_filename).stat().st_size,
        )
        manifest["artifacts"][old_filename] = metadata
    (published / "model_manifest.json").write_text(json.dumps(manifest))
    readme = tmp_path / "README.md"
    readme.write_text("<!-- evaluation:start -->\nPending\n<!-- evaluation:end -->")
    reports = tmp_path / "reports"
    args = argparse.Namespace(
        data=output,
        models=models,
        published=published,
        output=reports,
        readme=readme,
        seed=99,
        batch_size=4,
    )
    evaluate(args)
    report = json.loads((reports / "evaluation.json").read_text())
    assert report["gates"]["case_invariance"]
    samples = pd.read_parquet(reports / "samples.parquet")
    assert len(samples) > 0
    trained_tasks = tasks(pd.read_parquet(output / "fl_train.parquet"), 17)
    for row in samples.to_dict("records"):
        training_task = (
            "ordering" if row["task"] == "ordering" else "single_record_weighted"
        )
        assert row["model_seen"] == (
            row["name"] in set(trained_tasks[training_task]["name"])
        )
    assert "Accuracy on as-is" in readme.read_text()
    original_hash = sha256(reports / "evaluation.json")
    evaluate(args)
    assert sha256(reports / "evaluation.json") == original_hash
    metadata = output / "preparation.json"
    original_metadata = metadata.read_text()
    metadata.write_text(original_metadata + " ")
    with pytest.raises(ValueError, match="do not match"):
        evaluate(args)
    metadata.write_text(original_metadata)
    (output / "fl_train.parquet").write_bytes(b"changed")
    with pytest.raises(ValueError, match="prepared input changed"):
        verify_preparation(output, ["fl_train.parquet"])


def test_schema_fails_before_data_processing(tmp_path):
    path = tmp_path / "broken.csv"
    pd.DataFrame({"wrong": ["John"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="required columns"):
        list(read_source(path, "fl"))


def test_task_fields_and_compound_names():
    source = pd.DataFrame(
        {
            "first": ["Jose Gabriel", "John", "John"],
            "last": ["Smith", "Jones", "John"],
            "source_first": ["JOSE GABRIEL", "JOHN", "JOHN"],
            "source_last": ["SMITH", "JONES", "JOHN"],
            "source_row": [0, 1, 2],
        }
    )
    result = tasks(source, 42)
    assert all(
        len(name.split()) == 1 for name in result["single_record_weighted"]["name"]
    )
    assert len(result["ordering"]) == 2
    assert set(result["ordering"]["label"]) == {"first_last", "last_first"}
    for row in result["ordering"].to_dict("records"):
        expected = (
            row["first"] + " " + row["last"]
            if row["label"] == "first_last"
            else row["last"] + " " + row["first"]
        )
        assert row["name"] == expected


def test_frequency_baseline_and_unseen_prior():
    frequency = Frequency(
        pd.DataFrame(
            {"name": ["John", "Smith"], "first_count": [90, 10], "last_count": [10, 90]}
        )
    )
    labels, scores = frequency.predict(["John", "Smith", "Unseen"], False)
    assert labels.tolist() == ["first", "last", "last"]
    assert scores.tolist() == pytest.approx([0.9, 0.9, 0.5])
    labels, scores = frequency.predict(
        ["John Smith", "Smith John", "Unseen Unknown"], True
    )
    assert labels.tolist() == ["first_last", "last_first", "last_first"]
    assert scores.tolist() == pytest.approx([81 / 82, 81 / 82, 0.5])


def test_calibration_weights_and_boundary_scores():
    result = metrics(
        np.array(["first", "first", "last"]),
        np.array(["first", "last", "last"]),
        np.array([0.8, 0.8, 1.0]),
        np.array([1.0, 3.0, 2.0]),
    )
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["ece"] == pytest.approx((4 / 6) * (0.8 - 0.25))
    assert result["calibration"][0]["accuracy"] == pytest.approx(0.25)
    assert result["calibration"][-1]["rows"] == 1
    assert result["log_loss"] == pytest.approx((-np.log(0.8) - 3 * np.log(0.2)) / 6)


def test_clustered_accuracy_preserves_correlated_source_fields():
    result = metrics(
        np.array(["first", "last"]),
        np.array(["first", "first"]),
        np.array([0.9, 0.9]),
        np.ones(2),
        np.array(["Smith", "Smith"]),
    )
    assert result["bootstrap_clusters"] == 1
    assert result["accuracy_ci_lower"] == pytest.approx(0.5)
    assert result["accuracy_ci_upper"] == pytest.approx(0.5)
