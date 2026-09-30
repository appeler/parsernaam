"""Train packed LSTMs on frozen FL splits with validation-only selection."""

import argparse
import logging
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, TensorDataset

from parsernaam.config import ModelConfig
from parsernaam.model import LSTM
from parsernaam.text import VOCABULARY, encode

from .common import sha256, verify_preparation, write_json
from .tasks import tasks

logger = logging.getLogger(__name__)


def loader(
    frame: pd.DataFrame, labels: list[str], args: argparse.Namespace, shuffle: bool
) -> DataLoader:
    """Encode once and return a deterministically shuffled record loader."""
    tensors = encode(frame["name"].tolist(), args.seq_len)
    targets = torch.tensor(
        frame["label"].map(dict(zip(labels, range(len(labels)), strict=True))).tolist()
    )
    return DataLoader(
        TensorDataset(tensors, targets),
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(args.seed),
    )


def train(args: argparse.Namespace) -> None:
    """Train both tasks and checkpoint every epoch, without inspecting test data."""
    verify_preparation(args.data, ["fl_train.parquet", "fl_val.parquet"])
    if args.device == "auto":
        args.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "checkpoints").mkdir(exist_ok=True)
    snapshot = args.output / "source"
    for source in [
        *Path("training").glob("*.py"),
        *Path("src/parsernaam").glob("*.py"),
    ]:
        destination = snapshot / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    train_tasks = tasks(pd.read_parquet(args.data / "fl_train.parquet"), args.seed)
    val_tasks = tasks(pd.read_parquet(args.data / "fl_val.parquet"), args.seed + 1)
    if any(
        frame.empty
        for frame in [
            train_tasks["single_record_weighted"],
            train_tasks["ordering"],
            val_tasks["single_record_weighted"],
            val_tasks["ordering"],
        ]
    ):
        raise ValueError("empty training or validation task")
    pd.DataFrame({"token": list(VOCABULARY)}).to_parquet(
        args.output / "vocabulary.parquet", index=False
    )
    manifest = {
        "schema_version": 2,
        "revision": "local-unpublished",
        "normalization": "casefold/uppercase/NFKD/drop accents/Latin apostrophe hyphen whitespace/title v1",
        "artifacts": {},
        "training": {
            "parameters": {
                k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
            },
            "torch": torch.__version__,
            "python": platform.python_version(),
            "git_commit": subprocess.check_output(  # noqa: S603
                [shutil.which("git") or "/usr/bin/git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "preparation_sha256": sha256(args.data / "preparation.json"),
            "code_sha256": {
                str(p): sha256(p)
                for p in sorted(
                    [
                        *Path("training").glob("*.py"),
                        *Path("src/parsernaam").glob("*.py"),
                    ]
                )
            },
        },
        "evaluation": "pending; no final test accessed during model selection",
    }
    for task, filename, labels in [
        (
            "single_record_weighted",
            "parsernaam.safetensors",
            ModelConfig.CATEGORIES_SINGLE,
        ),
        ("ordering", "parsernaam_pos.safetensors", ModelConfig.CATEGORIES_POSITIONAL),
    ]:
        torch.manual_seed(args.seed)
        training = loader(train_tasks[task], labels, args, True)
        validation = loader(val_tasks[task], labels, args, False)
        model = LSTM(len(VOCABULARY) + 2, args.hidden_size, 2, args.layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        criterion = torch.nn.NLLLoss()
        best, stale, history = float("inf"), 0, []
        for epoch in range(1, args.epochs + 1):
            started = time.monotonic()
            model.train()
            total, count = 0.0, 0
            for inputs, targets in training:
                optimizer.zero_grad()
                output = model(inputs.to(device))
                loss = criterion(output, targets.to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total += float(loss.detach().cpu()) * len(inputs)
                count += len(inputs)
            model.eval()
            val_loss, correct, n = 0.0, 0, 0
            with torch.inference_mode():
                for inputs, targets in validation:
                    output = model(inputs.to(device))
                    val_loss += float(
                        criterion(output, targets.to(device)).cpu()
                    ) * len(inputs)
                    correct += int(output.argmax(1).cpu().eq(targets).sum())
                    n += len(inputs)
            row = {
                "epoch": epoch,
                "train_loss": total / count,
                "val_loss": val_loss / n,
                "val_accuracy": correct / n,
                "seconds": time.monotonic() - started,
            }
            history.append(row)
            logger.info("%s %s", task, row)
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model": state,
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "rng_state": torch.get_rng_state(),
                    "parameters": manifest["training"]["parameters"],
                },
                args.output / "checkpoints" / f"{task}-epoch-{epoch:02d}.pt",
            )
            if row["val_loss"] < best:
                best, stale = row["val_loss"], 0
                save_file(state, args.output / filename)
            else:
                stale += 1
            write_json(args.output / f"{task}-history.json", history)
            if stale >= args.patience:
                break
        manifest["artifacts"][filename] = {
            "format": "safetensors",
            "seq_len": args.seq_len,
            "encoding": "padding0_unknown1",
            "hidden_size": args.hidden_size,
            "num_layers": args.layers,
            "labels": labels,
            "sha256": sha256(args.output / filename),
            "size": (args.output / filename).stat().st_size,
            "best_val_loss": best,
            "history": history,
        }
        write_json(args.output / "model_manifest.json", manifest)
    manifest["artifacts"]["vocabulary.parquet"] = {
        "format": "parquet",
        "sha256": sha256(args.output / "vocabulary.parquet"),
        "size": (args.output / "vocabulary.parquet").stat().st_size,
    }
    write_json(args.output / "model_manifest.json", manifest)


def main() -> None:
    """Run reproducible training from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("training/data"))
    parser.add_argument("--output", type=Path, default=Path("training/artifacts"))
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--seq-len", type=int, default=47)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if (
        min(
            args.seq_len,
            args.hidden_size,
            args.layers,
            args.batch_size,
            args.epochs,
            args.patience,
        )
        < 1
        or args.learning_rate <= 0
    ):
        parser.error("training sizes and learning rate must be positive")
    logging.basicConfig(level=logging.INFO)
    train(args)


if __name__ == "__main__":
    main()
