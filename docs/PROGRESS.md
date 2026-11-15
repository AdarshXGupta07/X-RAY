# Progress

What is done, what is not, and what to study to understand the finished parts.
Roadmap: [PLAN.md](PLAN.md). Last updated: 2026-11-15.

## Status by phase

| Phase | State | Evidence |
|---|---|---|
| P0 Setup | **Done** | Python 3.13 environment; PyTorch 2.14.1 (CUDA 12.6) sees the RTX 2050; package installs in editable mode; repo on GitHub |
| P0 Gate A, literature | **First pass only** | [reading_log.md](reading_log.md): 13 related works logged from abstracts; none read in full |
| P0 Gate B, datasets | **First pass only** | [dataset_audit.md](dataset_audit.md): 2 sites confirmed, 4 candidates to inspect; nothing downloaded |
| P1 Metrics | **Done** | `src/tbshift/metrics.py`, 34 passing tests |
| P1 Data | Not started | Needs datasets |
| P2 Model + score cache | Not started | |
| P3 Calibration + conformal | Specified, not written | 5 + 15 tests waiting (expected-fail) |
| P4 Leave-one-site-out | Not started | |
| P5 Shift detection | Specified, not written | 7 tests waiting |
| P6 Recalibration | Specified, not written | 5 tests waiting |
| P7 Degradations | Specified, not written | 27 tests waiting |
| P8 Demo | Not started | Folders only |
| P9 Paper | Not started | |

Test suite today: **34 passed, 59 expected-fail, 0 errors.**
**There are no experimental results yet.** Nothing in this repo supports any claim about TB detection.

## How to run what exists

```powershell
.\.venv\Scripts\Activate.ps1
pytest              # whole suite
pytest tests/test_metrics.py -v   # just the finished module
```

## What "expected-fail" means here

Each unwritten module has a test file describing exactly how it must behave. Those tests are marked `xfail(strict=True, raises=NotImplementedError)`:

- today they fail in the expected way, so the suite stays green;
- when a module is implemented, remove the `pytestmark` line at the top of its test file and the tests become real;
- if a function is implemented but the marker is left in, pytest reports an error, so it cannot be forgotten.

## Study guide for the finished work

You will be asked about this code in the viva. For each item: read the function, read its tests, then answer the question without looking.

| What was built | Where | Study this | Be able to answer |
|---|---|---|---|
| Package and environment | `pyproject.toml`, `README.md` | Virtual environments; `pyproject.toml`; editable installs; the "src layout" | Why does `import tbshift` work from any folder? Why install PyTorch before the package? |
| Ignore rules | `.gitignore` | What belongs in Git and what does not | Why are datasets and model files never committed, even public ones? |
| Confusion counts, sensitivity, specificity | `metrics.py` | Confusion matrix; the four error types; prevalence | Which cell is the dangerous one here? Why can sensitivity be undefined? |
| AUC by ranks | `metrics.py: auc_roc` | ROC curve; Mann–Whitney U; average ranks for ties | State what AUC means as a probability. Why does a monotonic change to the scores leave it unchanged? |
| ECE | `metrics.py: expected_calibration_error` | Calibration; reliability diagrams; binning | Work the two-bin example in the test by hand. What is the difference between positive-class ECE and the top-label ECE in Guo et al.? |
| Bootstrap interval | `metrics.py: bootstrap_ci` | Sampling with replacement; percentile intervals; why stratify | Why resample each class separately when there are only 58 TB cases? |
| Violation rate | `metrics.py: violation_rate` | Marginal vs conditional guarantees | Why can a valid conformal threshold still show a violation rate near 50%? |
| Tests as specification | `tests/` | pytest basics; `parametrize`; `xfail`; fixtures | Pick one test in `test_conformal.py` and explain what bug it would catch. |

Wider topics for this stage are in [STUDY_TRACK.md](STUDY_TRACK.md), weeks 1 and 4.

## Open decisions (yours)

1. Which datasets to download (see the audit). Needed before any further data or model work.
2. Whether the calibration and conformal modules are written for you or by you. They are the heart of the project and the part an examiner will probe hardest.
3. Read reading-log entry C1. It decides how the paper is framed.
