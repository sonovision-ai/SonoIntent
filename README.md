# SonoIntent

SonoIntent is a benchmark for clinical objective-driven ultrasound
segmentation. It contains **130,928 image–question samples** from **83,324
ultrasound images** across **30 datasets**, covering **23 clinical scenarios**
and **12 anatomical regions**. Every sample pairs an ultrasound image with a
clinical question and a structured reference reasoning that resolves the
question to the target to be segmented; the target mask is the prediction
target. Models are evaluated in one in-distribution (ID) setting and three
out-of-distribution (OOD) settings.

![Figure 1. Overview of the SonoIntent benchmark](assets/figure1.png)

**[Figure 1. Overview of the SonoIntent benchmark.](assets/figure1.pdf)**

## Task

Ultrasound requests in clinical practice often state a **clinical objective**
— measure ejection fraction, assess a nodule, plan a procedure — rather than
naming the structure to segment. A model must then resolve the objective to
the correct anatomical target before segmenting it. Most existing benchmarks
assume the target is named explicitly; SonoIntent makes objective-to-target
resolution part of the task. Each sample consists of

1. a clinical question, e.g. *"Please delineate the region required for
   calculating the ejection fraction."*;
2. a structured reference reasoning that combines clinician-defined
   objective-to-target mappings and boundary rules with case-specific image
   evidence;
3. a binary segmentation mask of the intended target.

A model that segments the right structure for the wrong objective receives no
credit under the target-aware metrics.

## Composition

| | count |
|---|---:|
| datasets | 30 |
| unique 2D images | 83,324 |
| image–question samples | 130,928 |
| clinical scenarios | 23 |
| anatomical regions | 12 |
| train / val / test | 102,248 / 5,810 / 22,870 |
| target-absent samples (independent test set) | 656 |

The sources span superficial and vascular (11,503 test samples), cardiac
(4,076), musculoskeletal (3,917), abdominal and pelvic (2,336), and obstetric
(1,038) imaging. The questions cover abnormality assessment, quantitative
measurement, morphological assessment, functional assessment, procedure
planning, and anatomical reference identification; Supplementary Table 1 of
the paper lists all 23 scenarios.

## Evaluation settings

The 22,870-sample test set comprises one ID setting and three OOD settings.
Shares refer to the test set.

| setting | samples | share | composition | `ood_type` |
|---|---:|---:|---|---|
| ID | 5,385 | 23.55% | held-in datasets, unseen images | `ID` |
| Dataset-OOD | 3,829 | 16.74% | 5 held-out sources (MUP+MicroSegNet, BUS-UC, DDTI, Fast-U-Net, MMOTU-3D) | `OOD_dataset` |
| Anatomy-OOD | 3,917 | 17.13% | a held-out anatomy (muscle) | `OOD_organ` |
| Objective-OOD | 9,739 | 42.58% | clinical objectives unseen during training | `OOD_intent` |

Target-absent samples form a separate 656-sample test set of breast and liver
questions with empty masks; their `ood_type` is `target_absent`.
Machine-readable subset definitions are in
[`configs/eval_subsets.yaml`](configs/eval_subsets.yaml).

## Metrics

The paper reports **target identification accuracy (TargetAcc)**, **Dice**,
and **target-aware Dice (TA-Dice)**, which credits mask overlap only when the
generated response states the correct target. Across the 26 model
configurations evaluated, the maximum Dice is 89.70% in ID and 18.84% under
Objective-OOD; Sa2VA-14B reaches 99.99% TargetAcc with 10.95% Dice under
Objective-OOD. On the 656 target-absent cases, 25 of the 26 configurations
achieve empty-mask accuracy of at most 3.51%.

For a setting with `n_evaluable` scorable samples:

```text
TargetAcc = n_correct / n_evaluable
TA-Dice   = sum(dice * 1[correct]) / n_evaluable
Dice      = mean of per-sample Dice
gIoU      = mean of per-sample pixel IoU
cIoU      = sum(pixel intersection) / sum(pixel union)
```

Each response is assigned one of four labels. `correct` and `incorrect` are
scorable and stay in the denominator. `A_unstated` means the response does not
name a known target; it stays in the denominator with zero contribution.
`C_unverifiable` means the response carries no usable generated text and is
excluded from both numerator and denominator. Target identification is
deterministic: the target is the object of the last `delineate ...` clause
(with the `, excluding ...` tail removed), matched against anchored,
most-specific-first aliases. No model or judge is used.

This repository implements the per-sample mask metrics (`Dice`, `IoU`,
`precision`, `recall`), the aggregate metrics above, and target scoring from
generated text. See [`sonointent/eval/metrics.py`](sonointent/eval/metrics.py),
[`sonointent/eval/evaluate.py`](sonointent/eval/evaluate.py),
[`sonointent/eval/target_accuracy.py`](sonointent/eval/target_accuracy.py) and
[`sonointent/eval/target_metrics.py`](sonointent/eval/target_metrics.py).

## Repository layout

| path | contents |
|---|---|
| `sonointent/build/` | reference reasoning renderer, target profiles and image-evidence operators, official split construction |
| `sonointent/data/` | loaders for the annotation package and its manifest |
| `sonointent/eval/` | mask metrics, evaluator CLI, official evaluation harness and report, target scoring |
| `examples/lisa_plus/` | reference training adapter for the LISA++ / ReasonSeg stack |
| `scripts/` | split construction and target-scoring entry points |
| `configs/` | evaluation subset definitions |
| `docs/` | data governance and license notes |
| `tests/` | unit tests for metrics, loaders, split logic and target matching |

## Data access

SonoIntent is built from 30 existing ultrasound datasets. This repository
contains no images, masks or annotations.

The SonoIntent annotation package contains the `public_id`-keyed questions
and reference reasoning, split and setting labels, image and mask SHA-256
values, and a per-source license register. It is being prepared for public
release and is not yet available. It contains no pixel data. Loaders for the
package are provided in `sonointent.data.annotations` and
`sonointent.data.manifest`.

The source datasets remain under their own licenses and are available
through their official channels:

| source | official entry point |
|---|---|
| EchoNet-Dynamic | <https://echonet.github.io/dynamic/> (application required; redistribution prohibited) |
| HC18 (fetal head) | <https://hc18.grand-challenge.org/> · <https://zenodo.org/records/1327317> |
| PSFHS (intrapartum) | <https://doi.org/10.1038/s41597-024-03266-4> |
| CUBS (carotid) | <https://doi.org/10.1016/j.ultrasmedbio.2021.03.022> |
| STMUS (muscle) | <https://doi.org/10.1016/j.compbiomed.2021.104623> |
| BUSI (breast) | <https://doi.org/10.1016/j.dib.2019.104863> |
| DDTI (thyroid) | <https://doi.org/10.1117/12.2073532> |

Several sources are non-commercial, share-alike or registration-only, and
some have no verifiable data-level license; check the source terms before use
or redistribution. Access control on a hosting platform does not by itself
permit redistribution. Release tiers are described in
[`docs/DATA_GOVERNANCE.md`](docs/DATA_GOVERNANCE.md) and component licenses in
[`docs/LICENSE_MATRIX.md`](docs/LICENSE_MATRIX.md).

## Installation

```bash
pip install -e ".[test]"        # library, evaluator CLI and pytest
pip install -e ".[build]"       # adds SciPy for the image-evidence operators
pytest tests/
```

Python 3.10 or later is required, with NumPy and Pillow. The suite contains
44 tests; two count checks skip unless `SONOINTENT_ANNOTATIONS` points to an
annotation package, so 42 pass by default.

## Building the benchmark

### Split

The official split is rebuilt from the assembled V1 record file, whose rows
carry `dataset`, `sample_id`, `rendered_image` and `rendered_mask`:

```bash
sonointent-build-split \
  --input train_full_v1_rendered.jsonl \
  --out-dir /path/to/final/split
```

This writes `train.jsonl`, `val.jsonl` and `test.jsonl`. Five sources are held
out whole as Dataset-OOD; `TG3K`, `Segthy` and `CUBS` contribute
Objective-OOD samples; `STMUS NDA` is the Anatomy-OOD source. The remaining
datasets are split 90/5/5 by the highest available group level — patient,
video, sweep, case or volume — with seed 42, so no subject crosses train, val
and test. The command reports component counts, checks that the splits are
disjoint and that no OOD sample entered train or val, and verifies that every
referenced image and mask exists.

### Reference reasoning

`sonointent/build/v1_engine.py` implements the reasoning renderer. For a given
target it walks a target-specific evidence profile, computes image evidence
with the operators in `sonointent/build/v19_metrics.py`, falls back to frozen
per-image evidence where a measurement is unavailable, and emits at most two
evidence clauses followed by the fixed boundary definition. The target
profiles are in `sonointent/build/v19_profiles.py` and the clinical task
specifications in `sonointent/build/v20_specs.py`.

The renderer reads the frozen V15 sample manifest and V17 evidence table from
the source tree named by `SONOINTENT_SOURCE_ROOT`. These frozen intermediate
artifacts and the driver that assembled the released records are not part of
this repository; see [Scope](#scope).

## Training and inference

The released records use four fields: `rendered_image`, `rendered_mask`,
`problem` (the clinical question) and `thinking` (the reference reasoning).
The training target is the reasoning followed by the segmentation token:

```text
thinking + " Corresponding segmentation: [SEG]"
```

[`examples/lisa_plus/ours_reason_seg_dataset.py`](examples/lisa_plus/ours_reason_seg_dataset.py)
is the adapter used for the LISA++ / ReasonSeg stack; it reads a split JSONL
and resolves images and masks relative to the `final/` directory that contains
`split/`. The other model families in the paper (Sa2VA, PixelLM, UGround,
PSALM, VRS, GLaMM, HyperSeg, InstructSeg, OMG-LLaVA, MedFuse, LaSaña and
Seg-Zero) were trained within their own upstream codebases using the same
record fields; their training code is not duplicated here.

Inference follows one protocol for every family: the model receives only the
rendered image and the question, generates greedily with
`max_new_tokens = 384`, and the emitted mask is thresholded at `pred > 0`.
`sonointent/eval/harness.py` is the shared driver. Implement a callable that
maps one split row to per-sample scores and generated text, then call:

```python
from sonointent.eval.harness import load_rows, run_test

rows = load_rows("final/split/test.jsonl")
run_test(out_dir, rows, infer_one, "Model-7B", "ckpt_best")
```

`infer_one(row)` receives the split row with `_image` and `_mask` absolute
paths and returns `dice, iou, precision, recall, inter, union, seg_emitted,
valid_mask` plus `generated_text`. The harness writes `predictions.jsonl`,
`metrics.json` and `paper_results.md`, including bootstrap confidence
intervals and dataset- and target-level tables. A sample that raises is not a
zero: it is retried once, and if it still fails the run is marked INCOMPLETE,
no metrics are written, and the process exits with code 2.

## Evaluation

The evaluator accepts masks as 8-bit PNGs (0/255), 0/1 images, or boolean
arrays.

### Paired predictions

Pairs mode is self-contained and does not require the annotation package.
Each JSONL line names a prediction, its reference mask, and the split:

```jsonl
{"public_id": "si_…", "split": "test", "ood_type": "ID", "pred_mask": "preds/si_….png", "gt_mask": "gts/si_….png"}
```

```bash
sonointent-eval --pairs pairs.jsonl --out metrics.json
```

### Benchmark package

Package mode reads setting labels from the annotation package and expects
reference masks at `gts/<public_id>.png`:

```bash
sonointent-eval --annotations SonoIntent-Annotations \
  --predictions preds.jsonl --gt-root gts/
```

`preds.jsonl` contains one object per line:
`{"public_id": "si_…", "pred_mask": "preds/si_….png"}`.

The full expected `public_id` set of the requested split defines the
benchmark. Missing, duplicate, unknown or unreadable predictions make the run
incomplete: the evaluator exits with code 2 and writes no metrics, so a
partial submission cannot produce full-benchmark numbers. Restrict the
benchmark to one subset with `--split target_absent` or
`--split test --ood-type OOD_intent`.

### Target-aware scoring

TargetAcc and TA-Dice are computed from a split file and the
`predictions.jsonl` written by the harness. Each split row supplies the
canonical target through its mask filename; each prediction supplies
`generated_text` and `dice`:

```bash
python scripts/score_targets.py \
  --split final/split/test.jsonl \
  --predictions predictions.jsonl \
  --model Model-7B \
  --out target_metrics.tsv
```

The output has one row per setting (Overall, ID, Dataset-OOD, Objective-OOD,
Anatomy-OOD) with the label counts, TargetAcc, TA-Dice and mean Dice. The
script refuses to run, with exit code 2, if any split row has no prediction.

### Python API

```python
import numpy as np
from sonointent.eval.metrics import score

pred = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)   # model output
gt   = np.array([[0, 255], [255, 0]], dtype=np.uint8)          # reference mask
dice, iou, precision, recall, inter, union = score(pred, gt)   # 1.0, 1.0, …
```

Binarization follows the official post-processing: float arrays in [0, 1] are
thresholded at `> 0`, stored 0/255 masks at `> 127`, and boolean arrays are
used as-is. Empty masks need no special case: an empty prediction against a
non-empty target scores 0, and two empty masks also score 0. Target-absent
samples are therefore assessed by empty-mask accuracy rather than Dice.

## Scope

This repository provides the reference reasoning renderer and its evidence
operators, the official split construction, the annotation and manifest
loaders, the mask and target-aware metrics, the evaluation CLI and shared
harness, a reference training adapter, and unit tests.

The following are not included:

- the Full V1 assembly driver that downloaded and preprocessed the 30 source
  datasets, rasterized masks and wrote the rendered images and records;
- the frozen V15 sample manifest and V17 per-image evidence table consumed by
  `v1_engine.py`, and the provenance data behind them;
- per-model training code, which lives in the upstream model repositories,
  and all model checkpoints;
- the SonoIntent annotation package, which is being prepared for release;
- the target-absent supplementary records beyond their split-level definition.

Consequently, the benchmark cannot be rebuilt end to end from this repository
alone. What can be reproduced directly is the split given an assembled record
file, target-aware and mask-level scoring given predictions, and the
evaluation protocol given a model.

## Citation

```bibtex
@misc{sonointent,
  title        = {The SonoIntent benchmark for clinical objective-driven
                  ultrasound segmentation},
  author       = {Qingbo Kang and Yajun Mu and Wen Wen and Yong Chen and Yiyue Li
                  and Jun Gao and Huahui Yi and Ting Wen and Yuxi Cheng and
                  Zekun Jiang and Kun Wang and Yushuang He and Wenwu Ling and
                  Kang Li},
  howpublished = {\url{https://github.com/sonovision-ai/SonoIntent}},
  note         = {Citation details will be updated upon publication.}
}
```

Until the paper appears, please also cite this repository and the source
datasets listed in the annotation package's source register.

## License

The code in this repository is released under the Apache License 2.0; see
[`LICENSE`](LICENSE). SonoIntent-authored questions and reference reasoning
are intended for release under CC BY 4.0 with the annotation package. The 30
source datasets, their masks and derived frames remain under each source's
own terms; this repository does not relicense them. See
[`docs/LICENSE_MATRIX.md`](docs/LICENSE_MATRIX.md) for details; attribution
obligations for the sources are listed in the annotation package's
`NOTICE.md`.

## Contact

**Qingbo Kang**  
Email: qingbokang@scu.edu.cn
