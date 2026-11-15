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
