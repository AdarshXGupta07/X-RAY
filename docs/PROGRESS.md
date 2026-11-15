# Progress

What is done, what is not, and what to study to understand the finished parts.
Roadmap: [PLAN.md](PLAN.md). Last updated: 2026-11-16 (Shenzhen data and P1 pipeline added).

## Status by phase

| Phase | State | Evidence |
|---|---|---|
| P0 Setup | **Done** | Python 3.13 environment; PyTorch 2.14.1 (CUDA 12.6) sees the RTX 2050; package installs in editable mode; repo on GitHub |
| P0 Gate A, literature | **First pass only** | [reading_log.md](reading_log.md): 13 related works logged from abstracts; none read in full |
| P0 Gate B, datasets | **In progress** | [dataset_audit.md](dataset_audit.md): Montgomery and Shenzhen downloaded and verified (2 sites); 4 candidates still to inspect |
| P1 Metrics | **Done** | `src/tbshift/metrics.py`, 34 passing tests |
| P1 Data | **Done for 2 sites** | Manifests, stratified patient-level splits, 512 px cache and `CXRDataset` (37 tests). Both sites split and cached. EDA notebook not written |
| P2 Model + score cache | Not started | |
| P3 Calibration + conformal | **Done** | `calibration.py`, `conformal.py`, 41 passing tests; validity simulation run (below) |
| P4 Leave-one-site-out | Not started | |
| P5 Shift detection | **Done on synthetic data** | `shift.py`, 14 passing tests; real-data run waits for P2 |
| P6 Recalibration | **Done on synthetic data** | `recalibration.py`, 13 passing tests; synthetic comparison of all methods (below) |
| P7 Degradations | Specified, not written | 27 tests waiting |
| P8 Demo | Not started | Folders only |
| P9 Paper | Not started | |

Test suite today: **146 passed, 27 expected-fail, 0 errors.**
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

## Synthetic site-shift experiment (all methods, no real images)

`python -m tbshift.experiments.synthetic_shift --out-dir results`: 1,000 repeats, seed 0, about 50 seconds. Synthetic TB logits: non-TB N(0, 1), TB N(2, 1), 50% TB at the source; 100 source TB cases for calibration; unlabelled target batch of 200.

**Violation rate** (share of repeats with true sensitivity below 90%; lower is better):

| Target site | Source certified | Label-free | Detect + defer | 100 target labels, certified |
|---|---|---|---|---|
| No change | 0.066 | 0.139 | 0.062 | 0.068 |
| All scores drop 0.5 (harmful) | **0.827** | 0.145 | 0.059 | 0.060 |
| All scores rise 0.5 (harmless) | 0.000 | 0.132 | 0.000 | 0.046 |
| TB rises to 80% of cases | 0.056 | **0.975** | 0.000 | 0.082 |
| TB scores more variable (subtle) | **0.980** | **0.874** | **0.723** | 0.055 |

**How often the label-free detector fires** (unlabelled batch size across the top):

| Target site | 10 | 50 | 200 |
|---|---|---|---|
| No change (false alarms) | 0.052 | 0.040 | 0.044 |
| Harmful shift | 0.166 | 0.470 | 0.919 |
| Harmless shift | 0.136 | 0.462 | 0.916 |
| TB rises to 80% | 0.258 | 0.814 | 0.999 |
| TB scores more variable | 0.075 | 0.132 | 0.295 |

What it shows:

1. **Silent failure is real (RQ1).** A shift the source never saw breaks the certified promise in 83–98% of repeats.
2. **The detector cannot tell harm from harmless change (RQ2).** It fires just as often on the harmless shift as the harmful one, fires almost always when only prevalence changed (no harm), and mostly misses the subtle harmful shift. Detect-and-defer is safe only by deferring 91–100% of cases.
3. **Label-free repair works only when its assumption holds (RQ3).** It fixes a pure score shift (0.827 → 0.145) but breaks a site where only TB prevalence changed (0.056 → 0.975). Even with no shift it violates 14% of the time, above the 10% budget, because it has no guarantee.
4. **Labels are the only method that fixed every scenario**, and it takes about 50 TB positives: with 50 labels (about 25 positives) the certified threshold still has to give up 16–17% of the time.

Full table, including every label budget and specificity: `results/synthetic_methods.csv` (regenerate with the command above).
**Caveat:** this is a toy world of normal distributions chosen to illustrate each failure mode. It shows the methods behave as their assumptions predict; it says nothing about how often each failure happens at real hospitals.

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
| KS statistic and test | `shift.py: ks_statistic`, `detect_score_shift` | Empirical CDFs; the KS statistic; p-values; significance level; false alarms vs power | Why must the largest CDF gap occur at an observed value? Why is the false-alarm rate about 5%? |
| Domain classifier | `shift.py: domain_classifier_auc` | Logistic regression; standardisation; stratified k-fold; out-of-fold predictions | Why would scoring on the training images overstate the shift? |
| Quantile matching | `recalibration.py: quantile_match`, `label_free_threshold` | CDFs and quantiles; mid-ranks; what "matching distributions" assumes | Why does matching fail when TB becomes more common, even though nothing is wrong with the scanner? |
| Few-label threshold | `recalibration.py: few_label_threshold` | Conformal thresholds from target positives; binomial number of positives in a random sample | With 50 labels at 50% TB, why can the certified threshold still be trivial? |
| Manifest builder | `data/manifest.py` | Why a manifest is the single source of truth; parsing file names with regular expressions; reading image headers without loading pixels | Why does the builder only look inside `CXR_png` folders? What would go wrong otherwise? |
| Splits | `data/splits.py` | Stratified sampling; patient-level leakage; largest-remainder rounding; why a fixed seed | Why must one patient never appear in both train and test? Why does Montgomery's calibration split get only 9 TB cases? |
| Resize cache | `data/preprocess.py` | Crop vs pad vs stretch; resampling filters; why cache | Why can padding leak the class at Montgomery and Shenzhen? |
| PyTorch dataset | `data/dataset.py` | `Dataset` and `DataLoader`; ImageNet normalisation; augmentation; why no horizontal flip; pickling for Windows workers | Why copy one gray channel into three? Why does a lambda break `num_workers > 0` on Windows? |
| Synthetic shift experiment | `experiments/synthetic_shift.py` | Simulation design; controls; why exact sensitivity is known here | Explain each of the five scenarios and what it was built to reveal. |
| Tests as specification | `tests/` | pytest basics; `parametrize`; `xfail`; fixtures | Pick one test in `test_conformal.py` and explain what bug it would catch. |

Wider topics for this stage are in [STUDY_TRACK.md](STUDY_TRACK.md), weeks 1 and 4.

## Open decisions (yours)

1. Which datasets to download (see the audit). Needed before any further data or model work: the real-data versions of P4–P6 depend on it.
2. Study `conformal.py` line by line before anything else. It was written for you, and it is the part an examiner will probe hardest.
3. Read reading-log entry C1. It decides how the paper is framed.
