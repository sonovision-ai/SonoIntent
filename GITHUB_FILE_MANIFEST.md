# GITHUB_FILE_MANIFEST.md

Per-file justification for every file intended for the public GitHub
repository. Anything not listed here is intentionally kept out (internal
audit reports, staging workspaces, private keys, mappings, data, logs).

| path | why it is public |
|---|---|
| `README.md` | main entry point: benchmark description, composition, settings, metrics, access, install, usage, reproducibility scope, citation, license status, contact. All numbers quoted from the paper; no invented links. |
| `GITHUB_FILE_MANIFEST.md` | this file: documents why each file is published |
| `pyproject.toml` | package metadata, dependencies (numpy, Pillow), the `sonointent-eval` entry point, test extra |
| `.gitignore` | strict exclusions: bytecode, caches, data files, images outside `assets/`, secrets, logs |
| `sonointent/__init__.py` | package marker with `__version__` |
| `sonointent/data/__init__.py` | subpackage marker |
| `sonointent/data/annotations.py` | runnable loader for the public annotation package (text-only, no pixels); used by tests and by users |
| `sonointent/data/manifest.py` | reader for the 11-field portable public index (`image_index.csv`) with schema validation |
| `sonointent/eval/__init__.py` | subpackage marker |
| `sonointent/eval/metrics.py` | official Dice/IoU/precision/recall with the paper's binarization and empty-mask semantics |
| `sonointent/eval/settings.py` | the four evaluation settings (ID / Dataset-OOD / Anatomy-OOD / Objective-OOD) |
| `sonointent/eval/evaluate.py` | runnable per-setting aggregation CLI (pairs mode and package mode). Official aggregate definitions: gIoU = mean per-sample pixel IoU, cIoU = total intersection / total union. Package mode validates the FULL expected public_id set (missing / duplicate / unknown / unreadable → incomplete run, exit 2, no metrics written) |
| `configs/eval_subsets.yaml` | public subset definitions mapping paper setting names to `ood_type` field values |
| `tests/test_metrics.py` | unit tests for the official metrics (dtype handling, empty-mask semantics) |
| `tests/test_manifest.py` | schema tests for the 11-field index (accept real schema, reject legacy 9-field, reject empty) |
| `tests/test_annotations.py` | loader tests on a synthetic public package |
| `tests/test_evaluate.py` | evaluator tests (perfect match, empty prediction, missing-prediction incomplete handling, target-absent bucketing, gIoU/cIoU) |
| `tests/test_counts.py` | paper-number invariants against a real annotation package when `SONOINTENT_ANNOTATIONS` is set (skips otherwise) |
| `docs/DATA_GOVERNANCE.md` | release tiers, privacy policy, release gates |
| `docs/LICENSE_MATRIX.md` | which components carry which license status; nothing relicensed |
| `assets/overview_fig1_page.png` | Figure 1 of the paper (page rendered from the manuscript PDF) used in the README overview; no dataset pixels beyond the paper's own figure |

## Intentionally not published

- `LICENSE` — the Apache-2.0 proposal stays internal until the project lead
  approves; shipping a pending license as if it were final would misstate the
  repository's legal status.
- `CITATION.cff` — still contains TODO placeholders (publication year, DOI);
  kept internal until the paper is published.
- Any image/mask/annotation data, internal identifiers, absolute paths,
  private keys or identifier mappings, experiment results, model weights and
  training logs.
