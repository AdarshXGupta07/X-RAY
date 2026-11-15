# Progress

What is done, what is not, and what to study to understand the finished parts.
Roadmap: [PLAN.md](PLAN.md). Last updated: 2026-11-15 (P3 added).

## Status by phase

| Phase | State | Evidence |
|---|---|---|
| P0 Setup | **Done** | Python 3.13 environment; PyTorch 2.14.1 (CUDA 12.6) sees the RTX 2050; package installs in editable mode; repo on GitHub |
| P0 Gate A, literature | **First pass only** | [reading_log.md](reading_log.md): 13 related works logged from abstracts; none read in full |
| P0 Gate B, datasets | **First pass only** | [dataset_audit.md](dataset_audit.md): 2 sites confirmed, 4 candidates to inspect; nothing downloaded |
| P1 Metrics | **Done** | `src/tbshift/metrics.py`, 34 passing tests |
| P1 Data | Not started | Needs datasets |
| P2 Model + score cache | Not started | |
| P3 Calibration + conformal | **Done** | `calibration.py`, `conformal.py`, 41 passing tests; validity simulation run (below) |
| P4 Leave-one-site-out | Not started | |
| P5 Shift detection | Specified, not written | 7 tests waiting |
| P6 Recalibration | Specified, not written | 5 tests waiting |
| P7 Degradations | Specified, not written | 27 tests waiting |
| P8 Demo | Not started | Folders only |
| P9 Paper | Not started | |

Test suite today: **77 passed, 39 expected-fail, 0 errors.**
**The only results so far are synthetic** (validity simulation, no real images). Nothing in this repo supports any claim about TB detection.

## Validity simulation (synthetic, no shift)

`python -m tbshift.experiments.validity_simulation`: target 90% sensitivity, delta 0.10, 5,000 calibration sets per row, seed 0. True sensitivity is known exactly because scores are standard normal.

| TB positives | Marginal: mean sens. | Marginal: violation rate | Certified: mean sens. | Certified: violation rate |
|---|---|---|---|---|
| 5 | 1.000 (trivial) | 0.000 | 1.000 (trivial) | 0.000 |
| 9 | 0.900 | 0.388 | 1.000 (trivial) | 0.000 |
| 20 | 0.906 | 0.389 | 1.000 (trivial) | 0.000 |
| 22 | 0.913 | 0.340 | 0.957 | 0.093 |
| 50 | 0.902 | 0.432 | 0.961 | 0.030 |
| 100 | 0.900 | 0.463 | 0.940 | 0.061 |
| 200 | 0.900 | 0.462 | 0.925 | 0.095 |

What it shows:

1. The marginal threshold keeps its promise **on average** (mean ≈ 0.90) but a single calibration set misses the target **34–46% of the time**, with no shift at all.
2. The certified threshold keeps the violation rate **at or below 10%** at every size, as designed.
3. The price of certifying is caution: with few positives it flags more images than needed (mean sensitivity 0.96 at 50 positives), which will cost specificity.
4. The label floors are real: below 9 positives no threshold exists; below 22 nothing can be certified.

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
| Temperature scaling | `calibration.py` | Logits and sigmoid; negative log-likelihood; why optimise over log T; `logaddexp` for stability | Why can temperature scaling never change AUC? What does the fit return when the logits carry no signal, and why? |
| Marginal conformal threshold | `conformal.py: conformal_threshold`, `marginal_rank` | Exchangeability; ranks of n + 1 exchangeable values; order statistics | Derive k = floor(alpha(n + 1)) from the rank argument. Why 9 positives at 90%? |
| Certified threshold | `conformal.py: certified_threshold`, `certified_rank` | Binomial distribution; PAC guarantees; Clopper–Pearson idea | Why is "fewer than k calibration scores below q" a Binomial(n, alpha) event? Why 22 positives at alpha = delta = 0.10? |
| Validity simulation | `experiments/validity_simulation.py` | Monte Carlo simulation; why a synthetic control comes before real data | Explain the 34–46% marginal violation rate to a non-expert. |
| Tests as specification | `tests/` | pytest basics; `parametrize`; `xfail`; fixtures | Pick one test in `test_conformal.py` and explain what bug it would catch. |

Wider topics for this stage are in [STUDY_TRACK.md](STUDY_TRACK.md), weeks 1 and 4.

## Open decisions (yours)

1. Which datasets to download (see the audit). Needed before any further data or model work.
2. Study `conformal.py` line by line before anything else. It was written for you, and it is the part an examiner will probe hardest.
3. Read reading-log entry C1. It decides how the paper is framed.
