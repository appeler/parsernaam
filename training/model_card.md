---
license: mit
library_name: parsernaam
language:
  - en
tags:
  - name-parser
  - lstm
  - safetensors
  - tabular-classification
model-index:
  - name: parsernaam 0.4
    results: []
---

# Parsernaam

Two character-level LSTMs classify a single-token name as `first` or `last`,
and a multi-token name as `first_last` or `last_first`. Parsernaam 0.4 uses
the SafeTensors weights in this repository. Older package versions pin the
historical PyTorch weights at their original immutable revision.

## Use

```bash
pip install 'parsernaam>=0.4.0'
```

```python
import pandas as pd
from parsernaam import parse_names

result = parse_names(pd.DataFrame({"name": ["JOHN SMITH", "Smith John"]}))
print(result["parsed_name"].tolist())
```

The package pins a Hugging Face commit and checks artifact SHA-256 hashes.
Each row produces one dictionary with `name`, `type`, and `prob`; the original
input is preserved. Scores are raw model outputs, not calibrated guarantees.
Invalid, blank, and non-Latin-only inputs return `unknown` with score zero.

## Training and preprocessing

These weights use Florida voter registration first/last fields from early
2022. They use no Indian data or Census list. Historical weights used Florida
registrations and a Census surname list; their notebook validation/test
splits overlap training and cannot support held-out quality claims.

Normalized surname hashes assign FL records to train/validation/calibration/test
at 80/10/5/5. Surnames are disjoint between partitions; first names and words
appearing in both field roles can overlap. Training samples 400,000 records
uniformly from the train partition. Repeated records contribute repeated
single-token examples. Compound first-name fields enter the ordering task.
Ordering is balanced between the two directions; identical normalized fields
are excluded. The frequency baseline uses all 12,222,850 valid FL records
assigned to training, more than the LSTMs receive.

Training and inference share Unicode casefold/uppercase/NFKD decomposition,
accent removal, ASCII letter/apostrophe/hyphen/whitespace filtering,
whitespace collapse, and title-casing. Padding is 0, unknown is 1, and
vocabulary indices start at 2. Sequence length is 47, with longer input
truncated. Two layers of 128 hidden units use packed sequences and the state
at the final real character. Adam, validation-loss early stopping, seed
20260929, and MPS were used; the best epochs were 1 for single names and 2
for ordering. No post-hoc calibration was fitted.

The [training pipeline](https://github.com/appeler/parsernaam/tree/8b35dae96de9401972c516c2fde78a73f4c539d5/training)
contains preparation, training, historical comparison, and evaluation scripts.
Source hashes, parameters, histories, code hashes, and artifact hashes are in
the manifest and aggregate evaluation JSON. Released weights are saved directly
as SafeTensors from a committed training implementation.
Raw registration records and source-linked sample rows are not distributed here.

## Evaluation

North Carolina registrations provide a geographic transfer test. Records are
from a different state; common name strings can overlap FL and the original
Census list. This is not a test where every name string is unseen.

Accuracy on the same held-out NC rows (as-is input is all caps):

| Task | Majority | FL frequency lookup | Original package | Fixed old inference | Released weights |
| --- | --- | --- | --- | --- | --- |
| Record-weighted single names | 50.0% | 84.9% | 54.2% | 80.9% | 90.9% |
| Distinct single names | 41.4% | 66.9% | 51.2% | 87.3% | 79.6% |
| Name ordering | 50.0% | 87.1% | 53.2% | 98.4% | 96.2% |

Released-weight ECE is 0.008 for record-weighted single names, 0.084 for
distinct names, and 0.003 for ordering. ECE uses ten score bands, weighting
each band's absolute score/accuracy gap by its evaluation weight. Aggregate
reports include band accuracy, Brier score, log loss, support breakdowns,
word count, case, and bootstrap intervals.

Case variants produce the same predictions and scores. The released models
beat both independent baselines on all three primary NC tasks. Fixed old
inference remains more accurate on distinct names and ordering. On
lookup-supported distinct NC names, frequency lookup scores 92.0% versus
81.3% for the released model. This release favors common-name accuracy and
reproducible training; it is not a uniform accuracy improvement.

See [evaluation/summary.md](evaluation/summary.md),
[evaluation/calibration.csv](evaluation/calibration.csv),
[evaluation/metrics.csv](evaluation/metrics.csv), and
[evaluation/evaluation.json](evaluation/evaluation.json).

## Scope and limitations

The four labels cannot represent every naming convention. Source field errors,
population imbalance, truncation, compound names, and names that serve both
roles can affect predictions. Native scripts are unsupported; normalization
can remove unsupported characters from mixed-script inputs. Indian name
quality has not been evaluated. These outputs describe name-field patterns;
they do not infer ethnicity, citizenship, religion, gender, eligibility, or
identity, and should not be the sole input to consequential decisions.

## License and authors

MIT. Rajashekar Chintalapati and Gaurav Sood.
