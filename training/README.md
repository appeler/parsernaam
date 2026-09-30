# Reproducible name-order training

The new models learn the existing four labels from Florida registration fields.
North Carolina registrations provide a geographic transfer test. The source
files remain outside this repository; prepared records, checkpoints, weights,
and source-linked report samples stay local.

## Run

```bash
uv sync --group train --group dev --group test
uv run --group train python -m training.prepare --limit 1000 --train-records 1000 --eval-records 100 --output training/runs/dry-data
uv run --group train python -m training.train --data training/runs/dry-data --output training/runs/dry-artifacts --epochs 1 --hidden-size 32
uv run --group train python -m training.prepare
uv run --group train python -m training.train
```

The default input directory is
`~/Documents/GitHub/ethnicolr/scripts/data-acquisition/raw`. Use `--fl` and `--nc`
to override it. Preparation checks column names before reading records. It
scans the entire source, retains original fields and zero-based source row IDs,
and rejects records where either field becomes empty after normalization.
Normalization casefolds and uppercases Unicode before decomposition so that sharp-s and
dotless-i have the same representation in administrative case variants. It
then removes accents, keeps ASCII letters, apostrophes, hyphens and whitespace,
collapses whitespace, and title-cases words. The package and training import
the same implementation.

A SHA-256 hash of the normalized surname assigns records to train (80%),
validation (10%), calibration (5%), or test (5%). First names and words used
in both source roles can overlap partitions. This design tests transfer to
held-out surnames, rather than guaranteeing unseen first-name strings.
The split rule is applied before sampling or counting. Seeded uniform record
sampling saves up to 400,000 train records and 20,000 records in each other
partition. Counts for the frequency baseline use every valid FL record assigned
to training. This gives the lookup access to more records than the LSTM; the
report records that difference. No Census list is added to the new models.

The single-word training task includes only one-word source fields, counting
repeated records. Multi-word first fields such as `Jose Gabriel` enter the
ordering task alongside their surname. Ordering is balanced between first/last
and last/first; equal normalized fields are excluded because their order cannot
be identified. Sequence length is 47, with padding 0, unknown 1, and vocabulary
characters starting at 2. Packed LSTMs use the state at the last real character.

Training uses MPS when available, then CUDA or CPU. Defaults are two layers,
128 hidden units, batch size 1,024, Adam at 0.001, gradient clipping at 5,
a maximum of 15 epochs, and patience of three epochs. The checkpoint and loss
history are saved every epoch. The best validation-loss weights are selected
without examining final test results. The calibration partition is reserved
for future calibration work; this run reports reliability of raw scores.
The lockfile, seed, software versions, code hashes, source hashes, prepared-file
hashes, parameters and artifact hashes document the run. Exact GPU bitwise
reproduction is not promised across different PyTorch versions or hardware.

## Evaluate

The evaluator requires a local copy of the pinned published weights and manifest:

```bash
uv run python - <<'PY'
import json
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

output = Path("training/runs/published")
output.mkdir(parents=True, exist_ok=True)
manifest = json.loads(Path("training/published_manifest.json").read_text())
for filename in manifest["artifacts"]:
    source = hf_hub_download(manifest["repository"], filename, revision=manifest["revision"])
    shutil.copyfile(source, output / filename)
shutil.copyfile("training/published_manifest.json", output / "model_manifest.json")
PY
uv run --group train python -m training.evaluate
PARSERNAAM_MODEL_DIR=training/artifacts uv run pytest -m live tests/test_010_name_parser.py tests/test_020_edge_cases.py --no-cov
make ci
```

Three runs share exactly the same rows: the original published transform
(no case normalization, length 30 for both models), published weights with
fixed inference (per-model lengths 30 and 47), and retrained packed models.
The majority baseline is fit on training labels. The frequency baseline uses
the probability that a word appears in the first-name field; for ordering it
compares the first and last words. Unseen words use the training prior, and
score ties choose last or last/first. It contains no neural-model logic.

The report covers FL test and NC, record-weighted and distinct-name single-word
views, and ordering by two versus three or more words. The NC distinct-name
sample is drawn from names in the full NC file. Each canonical name has total
weight one, distributed between conflicting source roles by its observed field
counts. FL distinct-name frequencies come from the held-out record sample.
Results include as-is and lowercase input, exact input support in the LSTM
sample, and endpoint-word support in the lookup. As-is NC is all caps.

Published and fixed-weight FL results are contaminated by the original FL
training data and notebook split leakage. NC is a different source state, but
common name strings overlap FL and the original Census surname list. A NC
record can therefore be new without its name string being new. The report
separates model-seen and model-unseen strings; it does not describe all NC
names as unseen.

`training/reports/evaluation.html` contains side-by-side comparisons, every
score-band reliability table, expected calibration error (ECE), Brier score,
log loss, coverage, 95% accuracy intervals from 200 seeded cluster bootstrap
replicates, and 20 seeded sample rows per task and state next to their
original source fields. Score bands have width 0.1 and include score 1.0 in the
last band. ECE weights the absolute score/accuracy gap by each band's evaluation
weight. Bootstrap clusters are canonical names in the distinct-name view and normalized
surnames in record-weighted tasks. No calibration is fit on test data. The aggregate JSON and CSV files
preserve the calculations and provenance. `summary.md` supplies the generated
README table. Source rows and the HTML report remain ignored by Git.

Preparation, training and evaluation accept `--help`. New model artifacts are saved as SafeTensors in `training/artifacts`. The
release hosts them on Hugging Face at an immutable revision. Historical PyTorch
weights and their adapter are used only by the evaluation harness. The original
package comparison is fixed by `training/published_manifest.json`, so a package
release cannot silently change that baseline.
