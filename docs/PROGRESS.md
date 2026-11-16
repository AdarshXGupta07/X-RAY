# Progress

What is done, what is not, and what to study to understand the finished parts.
Roadmap: [PLAN.md](PLAN.md). Last updated: 2026-11-16 (P9 paper: first full draft).

## Status by phase

| Phase | State | Evidence |
|---|---|---|
| P0 Setup | **Done** | Python 3.13 environment; PyTorch 2.14.1 (CUDA 12.6) sees the RTX 2050; package installs in editable mode; repo on GitHub |
| P0 Gate A, literature | **First pass only** | [reading_log.md](reading_log.md): 13 related works logged from abstracts; none read in full |
| P0 Gate B, datasets | **In progress** | [dataset_audit.md](dataset_audit.md): Montgomery and Shenzhen downloaded and verified (2 sites); 4 candidates still to inspect |
| P1 Metrics | **Done** | `src/tbshift/metrics.py`, 34 passing tests |
| P1 Data | **Done for 2 sites** | Manifests, stratified patient-level splits, 512 px cache and `CXRDataset` (37 tests). Both sites split and cached. EDA notebook not written |
| P2 Model + score cache | **Done (first pass)** | DenseNet-121, frozen baseline + fine-tuning per fold, with and without corner mask; score cache verified against manifests; shortcut checks |
| P3 Calibration + conformal | **Done** | `calibration.py`, `conformal.py`, 41 passing tests; validity simulation run (below) |
| P4 Leave-one-site-out | **Done for 2 sites** | `experiments/loso.py`, 5 tests; 1,000 calibration draws per model; results below |
| P5 Shift detection | **Done for 2 sites** | `shift.py` + `experiments/real_shift.py`; KS power and false alarms by batch size; embedding domain classifier |
| P6 Recalibration | **Done for 2 sites** | Label-free, detect-and-defer and few-label repair scored on an untouched half of the unseen site (below) |
| P7 Degradations | **Done** | `degradations.py` (30 tests) + `experiments/p7_degradations.py`; 4 degradations × 5 severities on the held-out Shenzhen pool (below) |
| P8 Demo | Done (Docker untested) | `site_check.py`, `backend/app/`, `frontend/`; see README "Demo" |
| P9 Paper | First full draft | `paper/paper.md`; figures and tables from `python -m tbshift.experiments.paper_assets` |

Test suite today: **233 passed, 0 expected-fail, 0 errors.** Every planned research module and the demo are implemented.
**First real-data results exist (P2, below)**, from two public datasets and one seed. They are a research prototype's first pass, not clinical evidence.

## P9: paper (first full draft)

`paper/paper.md`. One command regenerates every figure (`paper/figures/`, SVG and PDF) and every table inside the paper (blocks between `<!-- table:… -->` markers): `python -m tbshift.experiments.paper_assets` (seconds, from `results/`), or with `--rerun` to recompute the results from the cached model scores first.

| Item | Status |
|---|---|
| Limitations, Method, Results, Discussion, Conclusion | Drafted; every number checked against the CSVs on 2026-11-16 |
| Figures | F1 pipeline (Mermaid), F2 sensitivity by seed, F3 detector power, F4 label budget, F5 degradations, S1 validity |
| Tables | T1 datasets, T2 main per-seed results, T3 model quality and corner mask, T4 degradations |
| Related work | Drafted from the reading log; **every [VERIFY] reference has only been seen as an abstract or README**. Read C1, C2, C3, C5 in full before submission |
| Abstract | Drafted last, from the checked results |
| Your decisions | Venue and length limit, co-authorship with your supervisor, whether institutional review is needed, an AI-use statement if your institution or venue requires one |

## P8: the site-check demo

**What it does.** One page answers "can this hospital trust the threshold certified at Shenzhen?" Verdicts, in order:

| Verdict | When |
|---|---|
| Not enough data (`defer`) | fewer than 20 images |
| Re-certified (`recertified`) | at least 22 labelled TB cases: certify a new threshold from them (labels beat the detector) |
| Re-certify before use (`recalibrate`) | the labelled TB cases already show too many misses (one-sided binomial test), or the KS test sees a score shift |
| Promise plausible (`plausible`) | none of the above; "plausible, not proven" |

The label-free threshold move is deliberately not offered as a fix: it broke the promise in 27–80% of draws in the multi-seed study.

**Checked by hand (2026-11-16):**

- Real model (`p2_densenet/train_shenzhen__none`), CPU. 30 full-size Montgomery originals (15 TB) uploaded: 14 s; server logits match the P2 cache within 0.11 (CPU float32 vs GPU half precision). Verdict "plausible": KS p = 0.052 at 30 images, just above the 5% line, and none of the 15 labelled TB cases missed. Consistent with P5/P6 (power grows quickly between 20 and 50 images).
- Simulated scenarios over 40 draws each: *similar* → plausible 40/40; *scores drift upward* → recalibrate 40/40 (true sensitivity 1.00); *TB cases look milder* (−1.6 sd) → recalibrate 40/40, true sensitivity of the source threshold 0.65 on average (max 0.83); with 30 TB labels → recertified 40/40, true sensitivity 0.98, but almost no healthy people cleared (the specificity cost of few labels).
- Headless browser: no console errors, no failed requests, no horizontal scroll at 390 px.

**Not checked:** `docker compose up` (Docker Desktop was not running; starting it would compete for the RAM that crashed the seed runs).

## Five training seeds: the headline result (read this first)

Seeds 0–4 retrain every P2 model with different training randomness (initial head weights, batch order, augmentation); data splits are identical. P4 and P5/P6 were rerun on each seed's score cache. Command: `python -m tbshift.experiments.multi_seed` (about 4 minutes on CPU). Outputs: `results/multi_seed/`.

**Trained on Shenzhen, no mask; certified 90% threshold; 1,000 calibration draws per seed:**

| Seed | Shenzhen control: violation | Montgomery AUC | Montgomery: violation | Montgomery: sensitivity | Montgomery: specificity |
|---|---|---|---|---|---|
| 0 | 0.066 | 0.871 | 0.000 | 0.997 | 0.053 |
| 1 | 0.057 | 0.842 | 0.001 | 0.984 | 0.125 |
| **2** | **0.054** | **0.847** | **1.000** | **0.771** | 0.717 |
| 3 | 0.064 | 0.839 | 0.112 | 0.961 | 0.336 |
| 4 | 0.069 | 0.843 | 0.043 | 0.980 | 0.145 |

1. **Silent failure of the safety promise is real, and it depends on the training run.** Seed 2's threshold missed 90% sensitivity at Montgomery in every draw (77% sensitivity). At the source it passed the control like every other seed (5.4%), and its unseen-site AUC (0.847) looks just like the others'. Nothing visible at the source separates it from the safe seeds.
2. **The other four seeds fail the other way:** sensitivity is kept but specificity falls to 0.05–0.34.
3. **The label-free detector does not identify the failing seed.** With 50 unlabelled images it fires in 99–100% of draws for every seed. With 20 images it fires 98% for seed 2 versus 52–75% for the others: a hint, not a test.
4. **Few-label re-certification (69 Montgomery labels, 29 TB) held for all 10 models** (5 seeds × 2 masks): violation 0.114–0.133 on the evaluation half, specificity 0.09–0.36. Label-free repair violated in 27–80% of draws depending on seed.
5. **The corner mask does not help transfer:** paired within seed, it lowered Montgomery AUC in 5 of 5 Shenzhen seeds (mean −0.037) and in 4 of 5 Montgomery seeds (mean −0.042).

**Across-seed means (sd)** for P2: trained on Shenzhen, AUC 0.918 (0.010) at Shenzhen test, 0.849 (0.013) at Montgomery; trained on Montgomery, 0.852 (0.036) and 0.779 (0.043). Frozen baselines do not vary with seed (no training randomness).

This supersedes the single-seed conclusions below that "the certified sensitivity promise never failed". It did not fail for seed 0; it failed for seed 2.

## P7: controlled shifts by degrading images (Shenzhen model, held-out Shenzhen pool)

`python -m tbshift.experiments.p7_degradations score` (GPU, about 1 minute) then `... analyze --repeats 1000` (CPU, about 45 seconds). Thresholds are calibrated on a clean half; every method is scored on a degraded quarter it never saw. Re-scoring the clean images reproduced the P2 cache to within 0.004 logits.

**Violation rate** (sensitivity below 90%) and **detector firing** (KS, about 50 unlabelled degraded images):

| Condition | Certified | Label-free | Few-label certified (≈25 TB labels) | Detector fires | Certified specificity |
|---|---|---|---|---|---|
| Clean (control) | 0.10 | 0.10 | 0.10 | 0.01 | 0.48 |
| Blur, severity 3 | 0.02 | 0.09 | 0.11 | 0.41 | 0.19 |
| Blur, severity 5 | 0.00 | **0.65** | 0.11 | 1.00 | **0.00** |
| Noise, severity 3 | 0.00 | 0.18 | 0.12 | 1.00 | **0.01** |
| JPEG, severity 5 | 0.03 | 0.29 | 0.14 | 0.95 | 0.09 |
| Contrast, severity 4 | 0.09 | 0.15 | 0.12 | **0.11** | 0.41 |
| Contrast, severity 5 | 0.06 | 0.20 | 0.11 | 0.59 | 0.22 |

Full table (all 21 conditions, 7 methods, specificity): `results/p7_degradations/train_shenzhen__none.csv`.

What it shows:

1. **The certified sensitivity promise survived all 20 degradations** (violation ≤ 0.10). What broke was specificity: 0.48 → 0.00 under heavy blur or noise.
2. **Why:** the certified threshold sits deep in the TB low tail (logit −4.87, the 5th percentile of clean TB scores). Degradations squeeze all scores toward the middle (TB spread 4.3 → 1.4 under heavy blur), so even the weakest TB cases rise above it, and so do the normal cases. The model becomes uncertain; a cautious threshold turns that into "flag everyone".
3. **Label-free repair can break sensitivity even with the same TB prevalence** (65% violation under heavy blur): quantile matching assumes both classes move together, and blur moves them in opposite directions.
4. **Few-label re-certification with about 25 TB labels kept violation at 0.10–0.14 in every condition** and kept specificity at 0.14–0.54 where the source threshold fell to 0.00.
5. **Detection depends on the kind of damage:** blur and noise are detected almost always at high severity, but reduced contrast at severity 4 only 11% of the time, while specificity is already falling.
6. Caveat: these are simulated, "compressive" degradations. A real acquisition change that lowers TB scores without making the model uncertain (the synthetic "harmful shift") would still break the promise.

## P5 and P6 on real data: can the shift be seen, and can it be repaired? (two sites)

`python -m tbshift.experiments.real_shift --repeats 1000`: about 35 seconds on CPU. Each repeat splits the unseen site, stratified by class, into an adaptation half (the only place methods may look) and an evaluation half (where every method is scored).

**Detection, trained on Shenzhen → Montgomery, no mask** (KS test on scores):

| Unlabelled batch | 10 | 20 | 50 | 69 |
|---|---|---|---|---|
| Fires on Montgomery batch (power) | 0.10 | 0.58 | 0.99 | 1.00 |
| Fires on Shenzhen batch (false alarm) | 0.04 | 0.03 | 0.02 | 0.02 |

Embedding domain classifier: source vs Montgomery AUC 1.000; source vs source 0.52.

**Repair, trained on Shenzhen → Montgomery, no mask** (scored on the untouched evaluation half; violation = sensitivity below 90%):

| Method | Violation | Mean sensitivity | Specificity | Flags everyone |
|---|---|---|---|---|
| Source certified (no repair) | 1% | 1.00 | 0.05 | 0% |
| Label-free (quantile matching) | **27%** | 0.95 | 0.42 | 0% |
| Detect and defer | 0% | 1.00 | 0.00 | 100% |
| Few-label marginal, 69 labels | 52% | 0.90 | 0.55 | 0% |
| Few-label certified, 50 labels (≈21 TB) | 7% | 0.98 | 0.16 | 59% |
| **Few-label certified, 69 labels (29 TB)** | **11%** | 0.97 | **0.36** | 0% |

With the corner mask: label-free violates in 71% of repeats; few-label certified at 69 labels 13%, specificity 0.15.

**Trained on Montgomery → Shenzhen:** every source-based method flags everyone. Few-label certified on Shenzhen labels: 100 labels (≈51 TB) give 9% violation but specificity only 0.09, because this model is weak (AUC 0.71).

What it shows:

1. **RQ2: the shift is easy to see and the detector cannot tell what it means.** 50 unlabelled images are enough, but the shift here hurt specificity, not sensitivity, so a detector-triggered deferral throws away all automation for a problem that was not a safety problem.
2. **RQ3: label-free repair trades safety for usefulness.** It recovers specificity (0.05 → 0.42) but breaks the sensitivity promise in 27–71% of repeats. Montgomery has fewer TB cases (42%) than Shenzhen (51%), which is the prevalence trap the synthetic study predicted.
3. **RQ3: about 29 TB-positive labels from the new site re-certify the threshold** with specificity 0.36 instead of 0.05. With about 21 TB cases (50 labels) the certified threshold usually cannot be computed at all, as the label floor of 22 predicts.
4. Measured violation rates include sampling noise from the 69-image evaluation half, so values a little above 10% (11–17%) for certified thresholds are expected.

## P4: does the 90% promise survive the other hospital? (two sites)

`python -m tbshift.experiments.loso --repeats 1000`: about 20 seconds on CPU. Each draw takes a fresh, class-stratified half of the source site's held-out images for calibration and uses the other half as the in-distribution control. Violation = measured sensitivity below 90%.

**Trained on Shenzhen, no corner mask** (50 calibration TB cases per draw):

| Threshold | Shenzhen (control): sens. / violation / spec. | Montgomery (unseen): sens. / violation / spec. |
|---|---|---|
| Fixed, P(TB) ≥ 0.5 | 0.81 / 98% / 0.96 | 0.76 / **100%** / 0.83 |
| Marginal conformal | 0.90 / 36% / 0.79 | 0.95 / 8% / 0.36 |
| Certified (δ = 0.10) | 0.96 / **6.6%** / 0.47 | 1.00 / 0% / **0.05** |

With the corner mask the pattern is the same (certified: control violation 6.3%, specificity 0.65 → 0.08 at Montgomery).

**Trained on Montgomery:** the calibration half holds 8 TB cases, below the marginal floor (9) and the certified floor (22). Both conformal thresholds flag everyone in 100% of draws.

**Temperature scaling** (median T 1.2–2.2) barely changed calibration error at the source and did not fix it at the unseen site (ECE at Montgomery 0.08–0.29 depending on mask).

What it shows:

1. **The control passes.** At the home site the certified threshold violates in 6–7% of draws, inside its 10% budget; the marginal threshold violates in 36–38%, as the theory and simulation predicted.
2. **In the one direction that can be certified, the sensitivity promise did not fail; usefulness did.** Montgomery's scores sit higher, so TB cases are still caught, but the certified threshold clears only 5–8% of healthy people (vs 47–65% at home). Nothing in the source data warns of this.
3. **A default threshold fails silently on sensitivity:** P(TB) ≥ 0.5 catches 76% of TB at Montgomery, every time.
4. **With only two sites, RQ1 cannot yet show a sensitivity failure of the certified threshold.** Harmful shifts need more sites (candidates in the dataset audit) or the controlled degradations of P7.

## P2: first real-data results (two sites, seed 0)

Commands: `python -m tbshift.experiments.shortcut_checks`, `python -m tbshift.experiments.p2_models`, `python -m tbshift.experiments.p2_report`. Outputs in `results/` and `artifacts/` (git-ignored).

**Shortcut checks** (cross-validated AUC for predicting TB from one region; 0.5 = no signal):

| Region | Montgomery | Shenzhen |
|---|---|---|
| Top-left corner only (marker area) | **0.895** | 0.713 |
| Upper lung fields (control) | 0.727 | 0.802 |
| Image shape (orientation, size) | 0.535 | 0.695 |
| Age and sex | 0.742 | 0.546 |
| Which site (both sites pooled) | 1.000 | |

**Models** (AUC with 95% bootstrap interval; fine-tuned DenseNet-121, no corner mask):

| Trained on → evaluated on | AUC | Certified 90% threshold: sensitivity | Specificity |
|---|---|---|---|
| Shenzhen → Shenzhen test (99) | 0.914 (0.849–0.964) | 0.90 | 0.71 |
| Shenzhen → Montgomery (138) | 0.871 (0.808–0.924) | 1.00 | **0.26** |
| Montgomery → Montgomery test (20) | 0.812 (0.573–0.979) | 1.00 (trivial) | 0.00 |
| Montgomery → Shenzhen (662) | 0.714 (0.674–0.751) | 1.00 (trivial) | 0.00 |

Full table, including the frozen baseline, the corner-mask runs and the marginal threshold: `results/p2_densenet/summary.csv` and `report.csv`.

What it shows:

1. **A marker shortcut exists at Montgomery.** One corner predicts TB better than the lungs do.
2. **At the new site the sensitivity promise held in this draw, but specificity collapsed** (0.71 → 0.26). Montgomery's scores sit higher than Shenzhen's: the "harmless shift" pattern from the synthetic study, which is harmless for TB cases and costly for everyone else.
3. **Montgomery cannot be a calibration source on its own.** 9 calibration TB cases is below the floor of 22, so the certified threshold flags everyone, as predicted.
4. **The corner mask did not help consistently** with only these runs: it raised the Montgomery-trained model's unseen-site AUC (0.714 → 0.754) and lowered the Shenzhen-trained one's (0.871 → 0.798). One seed, wide intervals: not a conclusion yet.

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
| DenseNet-121 wrapper | `models/densenet.py` | Transfer learning; replacing a classification head; feature vectors (embeddings); memory-efficient DenseNet | What does the 1024-number embedding represent, and why is it cached? |
| Training loop | `training/train.py` | Forward, loss, backward, step; BCE with logits; AdamW; mixed precision; early stopping | Why restore the best-validation weights? Why is early stopping on 14 validation images unreliable? |
| Shortcut checks | `experiments/shortcut_checks.py` | Shortcut learning; positive and negative controls; cross-validation | Why is a corner that beats the lung fields a red flag? |
| P2 runner and report | `experiments/p2_models.py`, `p2_report.py` | Leave-one-site-out; frozen vs fine-tuned features; score caching | Why did fine-tuning help when trained on Shenzhen but hurt when trained on Montgomery? |
| Leave-one-site-out study | `experiments/loso.py` | Repeated random calibration splits; in-distribution controls; violation rate vs mean sensitivity | Why does the control matter more than the headline number? Why can Montgomery never be the calibration source here? |
| Real-data detection and repair | `experiments/real_shift.py` | Adaptation vs evaluation split; power and false-alarm rate; why each method may only see the adaptation half | Why must the few-label images be excluded from evaluation? Why did label-free repair break sensitivity here? |
| Degradations | `degradations.py` | Gaussian noise, Gaussian blur, JPEG compression, contrast; what each mimics in a real hospital | Why define blur strength relative to image size? |
| Degradation study | `experiments/p7_degradations.py` | Controlled experiments; score compression; why calibrate on clean and evaluate on degraded | Explain in one minute why heavy blur did not break the certified sensitivity promise but did destroy specificity. |
| Paper assets | `experiments/paper_assets.py` | Reproducible figures; why tables are generated, never typed; reading a violation-rate figure | Where does every number in Table 2 come from, and how would you regenerate it from scratch? |
| Site-check verdict logic | `site_check.py` | Turning statistics into a decision rule; one-sided binomial test; why labels outrank a detector | A site sends 15 labelled TB cases and 4 are missed. Why is that already evidence, before you have 22? |
| Demo API | `backend/app/` | FastAPI lifespan (load the model once), Pydantic schemas, thin routers, dependency on `app.state`, TestClient with a fake scorer | Why load the model at startup, not per request? How do the API tests run without the model or any patient data? |
| Multi-seed aggregation | `experiments/multi_seed.py` | Training randomness; mean and standard deviation across runs; paired comparisons | Why did one seed break the promise when four did not, and why could you not tell from the source data? |
| Synthetic shift experiment | `experiments/synthetic_shift.py` | Simulation design; controls; why exact sensitivity is known here | Explain each of the five scenarios and what it was built to reveal. |
| Tests as specification | `tests/` | pytest basics; `parametrize`; `xfail`; fixtures | Pick one test in `test_conformal.py` and explain what bug it would catch. |

Wider topics for this stage are in [STUDY_TRACK.md](STUDY_TRACK.md), weeks 1 and 4.

## Open decisions (yours)

1. Which datasets to download (see the audit). Needed before any further data or model work: the real-data versions of P4–P6 depend on it.
2. Study `conformal.py` line by line before anything else. It was written for you, and it is the part an examiner will probe hardest.
3. Read reading-log entry C1. It decides how the paper is framed.
