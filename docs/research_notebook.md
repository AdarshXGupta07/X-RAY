# Research Notebook

One entry per working session or experiment. Newest at the bottom.
Template: **Date · Question · What was done · Result · Conclusion · Next.**

---

## 2026-11-15 — Project start: setup, metrics, gates

**Question.** Can the project start on this machine, and are the two gates (novelty, datasets) passable?

**What was done.**

- Environment: Python 3.13 virtual environment, PyTorch with CUDA for the RTX 2050 (4 GB).
- Repository scaffold; `tbshift` installable as a package.
- `metrics.py` implemented and tested: confusion counts, sensitivity, specificity, AUC (rank-based), ECE, stratified bootstrap interval, violation rate.
- Specification tests written (marked expected-to-fail) for calibration, conformal thresholds, shift detection, recalibration and degradations.
- First-pass literature check → `reading_log.md`. First-pass dataset audit → `dataset_audit.md`.

**Result.** No experimental results yet. No data downloaded.

**Design decisions and why.**

1. *Score once, experiment many times.* The CNN runs over each image once per fold and saves scores and embeddings. All repeats and bootstraps then run on small arrays on CPU. Reason: 4 GB GPU, thousands of repeats.
2. *Two thresholds, not one.* Split-conformal promises the sensitivity target on average over calibration sets; it does not control how often a single calibration set falls short. A certified (PAC) threshold does. Both will be reported.
3. *Budgets counted in TB-positive labels.* Sensitivity is calibrated on positives only, so that is the scarce resource.
4. *Positive-class ECE.* ECE is computed on the TB probability, not on the confidence of the predicted class. The two definitions differ; the paper must say which is used.
5. *Convention.* Higher score = more likely TB; flagged when score ≥ threshold; ruled out when score < threshold.

**Conclusion.** Gate A: not passed yet, needs full reads; the likely framing is an empirical and statistical study, not a new method. Gate B: two sites confirmed, four more candidates to inspect.

**Next.** Decide which datasets to download; read the closest preprint (reading log C1); implement the calibration and conformal modules against their tests.

---

## 2026-11-15 — P3: calibration, conformal thresholds, validity simulation

**Question.** Do the two rule-out thresholds keep their promises when there is no site shift?

**What was done.**

- `calibration.py`: temperature scaling fitted by minimising NLL over log T (bounds 0.01–100).
- `conformal.py`: marginal threshold, k = floor(alpha(n + 1)); certified threshold, largest k with P(Binomial(n, alpha) ≤ k − 1) ≤ delta; both return −inf below their label floor.
- `experiments/validity_simulation.py`: standard-normal TB scores, so true sensitivity is exact; 5,000 calibration sets per size; seed 0.

**Result** (alpha = 0.10, delta = 0.10; full table in PROGRESS.md):

- Marginal: mean sensitivity 0.900–0.913; violation rate 0.34–0.46.
- Certified: violation rate 0.000–0.095 (≤ 0.10 at every size); mean sensitivity 0.925–0.967 once non-trivial.
- Label floors confirmed by simulation: 9 positives (marginal), 22 (certified).

**Conclusion.** The control behaves exactly as the theory says. "Source conformal is valid" and "source conformal violates the target ~40% of the time" are both true, so the paper must define violation precisely and report the certified threshold alongside the marginal one. Any real-data violation rate must be compared with these no-shift baselines, not with zero.

**Caveat.** Synthetic, continuous scores. Real scores have ties and only 58 TB cases at Montgomery, so results there will be noisier.

**Next.** Shift detection (P5) and recalibration (P6) can be written and tested on synthetic scores now; real experiments wait for data.
