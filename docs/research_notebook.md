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

---

## 2026-11-15 — P5 and P6: shift detection, recalibration, synthetic comparison

**Question.** Before any real data: do the label-free detector, label-free repair and few-label repair behave as their assumptions predict, and where does each one break?

**What was done.**

- `shift.py`: KS statistic written by hand (matches SciPy, including ties); KS test using SciPy's p-value; domain-classifier AUC from out-of-fold logistic-regression predictions.
- `recalibration.py`: quantile matching by mid-rank position; the equivalent one-shot threshold move (`label_free_threshold`); few-label marginal and certified thresholds from target positives only.
- `experiments/synthetic_shift.py`: five target scenarios × six method families, 1,000 repeats, seed 0. Exact sensitivity and specificity from the normal distributions.

**Result.** See the tables in PROGRESS.md. In short:

- Unseen score shift: source certified violates 0.827 (score drop) and 0.980 (TB spread).
- KS detector at batch 200 fires 0.919 on the harmful shift and 0.916 on the harmless one; 0.999 when only prevalence changed; 0.295 on the subtle TB-spread shift. False-alarm rate 0.040–0.055.
- Label-free repair: 0.827 → 0.145 on the score drop; 0.056 → 0.975 when TB rose to 80%; 0.139 with no shift.
- Few-label certified at 100 labels (about 50 positives): 0.046–0.082 in every scenario.

**Conclusion.** The most interesting synthetic finding for RQ2 is that a distribution detector measures "different", not "harmful". Whether that holds at real hospitals is now a concrete, testable question. For RQ3, label-free repair needs a prevalence check; the candidate datasets differ a lot in TB share (Mendeley about 83% TB, Shenzhen about 51%, Montgomery about 42%), so the prevalence failure is likely to matter in practice.

**Caveat.** A toy world chosen to show each failure mode. It shows the code does what the theory says; it does not estimate how often anything happens at a real site.

**Next.** Datasets (Gate B) → P1 data loaders → P2 score cache, then rerun P4–P6 on real cached scores. The domain-classifier detector is only tested on random embeddings so far.

---

## 2026-11-15 — First real data: Montgomery County

**Question.** Does the downloaded Montgomery set match NLM's description, and does it hide any obvious shortcut?

**What was done.**

- Downloaded `NLM-MontgomeryCXRSet.zip` from NLM (616,853,875 bytes; SHA-256 `54601e952315d8f67383e9202a6e145997ade429f54f7e0af44b4e158714f424`). CRC check passed; no unsafe paths; extracted into `data/raw/montgomery/`.
- Read the NLM ReadMe: do not share outside the research group; cite Jaeger et al. 2014 and Candemir et al. 2014 (both IEEE TMI 33(2)).
- Built `data/manifests/montgomery.csv` with `python -m tbshift.data.manifest --site montgomery --raw data/raw/montgomery`.

**Result.**

- 138 images: 58 TB, 80 normal. Matches the ReadMe exactly. Every clinical reading agrees with its file-name label.
- Images are 4020×4892 (97) or 4892×4020 (41). All are 8-bit grayscale (values 0–255), although the ReadMe says 12-bit.
- Orientation differs by class: landscape in 29 of 80 normal (36%) vs 12 of 58 TB (21%).
- Demographics differ by class: mean age 34.3 (normal) vs 48.1 (TB); male 32% (normal) vs 64% (TB). One reading lists sex as "O" and is left blank.

**Conclusion.**

1. Orientation is a possible shortcut. Resizing to a square stretches portrait and landscape images differently, and the classes have different mixes. Pad to square (or crop consistently) before resizing, and in P2 test whether a classifier can predict TB from orientation or aspect ratio alone.
2. Age and sex differ strongly by class. The model may partly learn "older man" rather than "TB". Report it; consider an age- or sex-stratified check later.
3. 8-bit vs 12-bit: no effect on our pipeline (ImageNet models expect 8-bit), but mention it when reporting the data.

**Next.** Shenzhen download (needs approval: 3.77 GB, about 7.5 GB with extraction). Then splits and the image loader (rest of P1).

---

## 2026-11-16 — Second site: Shenzhen; P1 pipeline finished

**Question.** Does Shenzhen match NLM's description, and how different are the two sites before any model sees them?

**What was done.**

- Downloaded Shenzhen file by file from NLM's official data host (the zip link returned an error page). 1,324 files, 0 failures, every PNG size equal to NLM's listing.
- Built manifests for both sites; fixed the readings parser for four Shenzhen formats ("femal", "female24yrs", "Female, 28yrs", ages in months and days) and DICOM-style ages at Montgomery.
- Split both sites (seed 0; 60/10/15/15) and cached centre-cropped 512 px copies (Montgomery 14 MB, Shenzhen 84 MB).
- Looked at thumbnails of both sites.

**Result.**

| | Montgomery (USA) | Shenzhen (China) |
|---|---|---|
| Images (TB / normal) | 138 (58 / 80) | 662 (336 / 326) |
| TB share | 42% | 51% |
| Image size | 4020×4892 or 4892×4020 | 948–3001 px per side |
| Landscape share, normal vs TB | 36% vs 21% | **58% vs 31%** |
| Mean age, normal vs TB | 34.3 vs 48.1 | 33.1 vs 37.7 |
| Male, normal vs TB | 32% vs 64% | 68% vs 68% |
| Under 18, normal vs TB | 13 vs 4 | 24 vs 7 |
| Calibration split TB cases (seed 0) | **9** | 50 |

Other Shenzhen facts: 26 of 27 RGB-format files are TB (all are gray in content, so converting to grayscale removes it); all 21 images with a short side under 2,000 px are normal (children's "ap" views); burned-in "L" and "R" markers appear in both classes in a 24-image sample.

**Conclusion.**

1. **Orientation is a real shortcut risk at both sites.** Centre-cropping removes the shape cue, but text markers and field of view remain. P2 must test it: train a classifier on aspect ratio alone, and one on image corners only. If either beats chance clearly, report it and consider masking corners.
2. **Image size is a shortcut at Shenzhen:** small images are all normal children. After resizing, their sharpness differs.
3. **Site design consequence:** Montgomery's calibration split has 9 TB cases, exactly the marginal floor and far below the certified floor of 22. When Montgomery is the only source, the certified threshold cannot come from its own split. Options for P4: calibrate on the pooled source sites, use a larger calibration fraction at small sites, or report Montgomery-as-source with the marginal threshold only and say so.
4. Demographic shift between sites (sex mix, age gap between classes) is part of the "site" in this study. The paper should describe it, not hide it.

**Next.** P2: frozen DenseNet-121 features + logistic regression baseline, the shortcut checks above, then fine-tuning and the score cache.

---

## 2026-11-16 — P2: model, shortcut checks and score cache

**Question.** Does a fixed DenseNet-121 learn TB rather than site or marker cues, and what does a source-calibrated 90%-sensitivity threshold do at the other site?

**What was done.**

- Environment: switched to Anaconda's scientific stack (Smart App Control blocked newer pip wheels); removed the duplicate OpenMP runtime. 157 tests pass.
- Shortcut checks on the cached images, with lung-field patches as positive control.
- Per fold (train on one site) and per corner-mask setting: frozen ImageNet features + logistic regression (C on the validation split), then fine-tuning (2 head epochs, then full network, AdamW 1e-4, early stopping on validation AUC, patience 5, batch 16, 224 px, mixed precision; 641 MB peak GPU memory). Seed 0.
- Score cache for every image of both sites under each model; verified image by image against the manifests.
- `p2_report.py`: AUC with bootstrap intervals; marginal and certified thresholds from the source calibration split's TB cases, applied to the source test split and the unseen site.

**Result.** Tables in PROGRESS.md. Key numbers:

- Montgomery top-left corner alone: AUC 0.895 (0.834–0.944), above its upper lung fields 0.727. Site from a 16×16 thumbnail: AUC 1.000.
- Trained on Shenzhen, no mask: AUC 0.914 on its test split, 0.871 on Montgomery. Certified threshold: sensitivity 0.90 / specificity 0.71 at Shenzhen test; 1.00 / 0.26 at Montgomery.
- Trained on Montgomery: AUC 0.714 on Shenzhen (frozen baseline 0.847: fine-tuning on 83 images made transfer worse). Certified threshold trivial (9 calibration TB cases).
- Corner mask: mixed effect (see PROGRESS.md).

**Conclusion.**

1. Direction Shenzhen → Montgomery is a real, measurable site shift. The cost appeared in specificity, not sensitivity. RQ1 must report both: "the promise survived" can hide a threshold that clears almost nobody.
2. Montgomery → Shenzhen cannot be certified from Montgomery alone. P4 must either pool calibration data, use few-label re-certification, or present this direction with the marginal threshold and say why.
3. The marker shortcut is real at Montgomery. With one seed and two folds the corner mask's effect is not resolved; it needs several seeds.

**Caveats.** One seed. Montgomery's validation split has 14 images, and its validation AUC reached 1.000, so early stopping there is close to arbitrary. Test splits are small (Montgomery 20, Shenzhen 99). Intervals are wide.

**Next.** P4: repeat the calibration/test split many times on the cached scores (cheap), add temperature scaling, and measure violation rate and specificity per direction. Run 3–5 seeds of P2 for the mask comparison when time allows.

---

## 2026-11-16 — P4: leave-one-site-out on the score cache

**Question (RQ1).** When a threshold calibrated at one hospital is used at the other, how often does the 90% sensitivity promise fail, and what else changes?

**What was done.** `experiments/loso.py`. For each P2 model: 1,000 draws; each draw splits the source site's held-out pool (cal + test splits) into a class-stratified calibration half and a control half; fits temperature on the calibration half; sets fixed (logit ≥ 0), marginal and certified thresholds (α = δ = 0.10); measures sensitivity and specificity on the control half and on every image of the unseen site. Seed 0. About 20 seconds on CPU.

**Result.** Tables in PROGRESS.md. Trained on Shenzhen, no mask:

- Control: marginal mean sensitivity 0.903, violation 35.8%; certified 0.962, violation 6.6%.
- Montgomery: certified sensitivity 0.997, violation 0%, specificity 0.053 (control 0.467). Marginal 0.949 / 8.1% / 0.362. Fixed threshold 0.759 / 100% / 0.825.
- Trained on Montgomery: 8 calibration TB cases per draw; both conformal thresholds trivial in every draw.

**Conclusion.**

1. The pipeline behaves correctly on real data: the in-distribution control reproduces the simulation (certified ≤ δ; marginal ≈ 35–45%).
2. For this pair of sites the shift raises scores. The certified promise survives at Montgomery, but specificity collapses from 0.47 to 0.05. The honest RQ1 finding so far is "silent loss of usefulness", not "silent loss of safety".
3. The naive P ≥ 0.5 threshold is the one that silently fails on sensitivity.
4. Two sites cannot answer RQ1 in general. Priorities: more sites (DA/DB, TBX11K, Mendeley after inspection) and P7 degradations, which can create harmful shifts on purpose.

**Caveats.** One model seed per fold; the target set is fixed (138 Montgomery images, 58 TB), so its own sampling noise is not in the violation rate; temperature scaling is fitted per draw on about 100 images.

**Next.** P5/P6 on real scores: does the KS detector fire for Shenzhen → Montgomery, and do label-free and few-label recalibration restore specificity without breaking sensitivity?

---

## 2026-11-16 — P5 and P6 on real data

**Question.** RQ2: can the Shenzhen → Montgomery shift be detected without labels? RQ3: which repair restores a useful threshold without breaking the 90% sensitivity promise, and how many labels does it take?

**What was done.** `experiments/real_shift.py`, 1,000 repeats per P2 model, seed 0. Source: certified threshold from a calibration half of the held-out pool; the other half is an unlabelled reference. Target: stratified split into adaptation and evaluation halves; methods only touch the adaptation half; all are scored on the evaluation half. KS detector at batch sizes 10–69 (Montgomery) or 10–331 (Shenzhen) with false alarms from source batches; domain classifier on embeddings.

**Result** (Shenzhen → Montgomery, no mask; full tables in PROGRESS.md and `results/p5p6_real/`):

- KS power 0.10 / 0.58 / 0.99 / 1.00 at batch 10 / 20 / 50 / 69; false alarms 1.5–3.8%. Domain AUC 1.000 vs 0.52 null.
- Violation / specificity: source certified 1% / 0.05; label-free 27% / 0.42; detect-defer 0% / 0.00; few-label certified 69 labels 11% / 0.36; 50 labels 7% / 0.16 with 59% trivial.
- With the corner mask, label-free violates 71%.
- Montgomery → Shenzhen: source thresholds trivial; few-label certified at 100 labels 9% / 0.09.

**Conclusion.**

1. Detection is the easy part: 50 unlabelled images. Deciding what to do about it is not, because "different" was not "unsafe" here.
2. Label-free repair is not safe for a sensitivity promise when TB prevalence differs between sites.
3. Few-label re-certification is the only repair that kept the promise and restored usefulness, and its cost is set by TB-positive labels: about 29 here, consistent with the derived floor of 22.

**Caveats.** Two sites, one model seed; evaluation halves are small (69 Montgomery images, 29 TB), which inflates measured violation slightly; the domain classifier reaches 1.000 so easily that it says little beyond "these are different hospitals".

**Next.** P7 degradations to create shifts that do hurt sensitivity; more sites from the audit; several P2 seeds for the corner-mask question.

---

## 2026-11-16 — P7: controlled degradations

**Question.** If a shift is made on purpose (noise, blur, JPEG, contrast at 5 severities), does the source certified threshold break, does the detector notice, and do the repairs work?

**What was done.** Implemented `degradations.py` (per-image seeds; blur defined at 224 px and scaled up). `p7_degradations.py score` ran the P2 Shenzhen model (no mask) on the 198-image held-out Shenzhen pool under 21 conditions (clean check: max difference 0.004 logits from the P2 cache). `analyze`: 1,000 repeats per condition; calibrate on a clean stratified half; split the degraded other half into adaptation and evaluation quarters (about 25 TB each).

**Result.** Tables in PROGRESS.md. Certified violation ≤ 0.10 in every condition; certified specificity 0.48 (clean) → 0.00 (blur 5, noise 3–5). Label-free violation up to 0.65 (blur 5). Few-label certified 0.10–0.14 everywhere. KS detection: blur 4–5 and noise 2–5 ≈ 1.00; contrast 4 = 0.11.

Distribution check (all 100 clean TB cases): certified threshold −4.87. TB score 5th percentile / median / spread: clean −4.89 / +3.62 / 4.28; blur 5 −2.55 / −0.73 / 1.36. Normal median −5.78 → −1.46.

**Conclusion.**

1. For this model, image degradation makes predictions less confident rather than confidently wrong. A low, certified rule-out threshold is robust to that on sensitivity and pays in specificity.
2. This sharpens RQ1's answer from P4: the silent failure we can actually demonstrate on public data is loss of usefulness, not loss of safety. The paper should say this plainly and show the synthetic "harmful shift" as the case that would break safety.
3. Label-free quantile matching has a second failure mode besides prevalence: shifts that move the two classes differently.
4. The "detect, then re-certify with about 25 labelled TB cases" combination worked in every condition. That is the practical recommendation the data supports so far.

**Caveats.** One model, one seed; evaluation quarters have about 25 TB cases, so violation estimates carry roughly ±0.03 of their own noise; degradations are simulated.

**Next.** Experiments for the core questions are now complete for two sites. Remaining: more P2 seeds (corner mask question), more sites if possible, then the demo (P8) and the paper (P9). Per the plan, freeze experiment design here; new ideas go to "future work".

---

## 2026-11-16 — Five training seeds (changes the RQ1 conclusion)

**Question.** Are the P2–P6 results properties of the method, or of one training run? Does the corner mask help?

**What was done.** Added `--seed` / `--run-name` to the P2 runner (splits unchanged; only training randomness varies) and made it save after every model. Trained seeds 1–4 (seed 0 existed). `experiments/multi_seed.py` reran P4 and P5/P6 (1,000 draws each) on every seed's cache and reports mean, sd and paired mask differences. Seeds 2–4 had to be rerun one at a time because the machine ran out of memory (bug log).

**Result.** Tables in PROGRESS.md. Key numbers (trained on Shenzhen, no mask):

- Certified violation at Montgomery per seed: 0.000, 0.001, **1.000**, 0.112, 0.043. Shenzhen control: 0.054–0.069 for all seeds.
- Seed 2 at Montgomery: sensitivity 0.771, specificity 0.717, AUC 0.847. Other seeds: sensitivity 0.96–1.00, specificity 0.05–0.34.
- KS power at batch 50: 0.989–1.000 for every seed; at batch 20: seed 2 0.983, others 0.52–0.75.
- Few-label certified, 69 labels: violation 0.114–0.133 for all 10 models.
- Corner mask, paired: Montgomery AUC lower in 5/5 Shenzhen seeds (−0.037) and 4/5 Montgomery seeds (−0.042).

**Conclusion.**

1. RQ1, revised: a source-certified sensitivity promise can fail silently at a new hospital. Here it happened for 1 of 5 equally good training runs. The source-side evidence (control violation, AUC) looked identical for the run that failed and the runs that did not.
2. The failure is not predictable from accuracy: seed 2's unseen-site AUC was average. What differs is where the model places the new site's TB scores relative to a threshold learnt elsewhere.
3. RQ2: the detector sees "different" for every seed and cannot isolate the harmful one at practical batch sizes.
4. RQ3: few-label re-certification is the only method that held for every model. Cost: about 29 TB-positive labels from the new site.
5. Single-run results in a site-shift study can be misleading in either direction. Report across seeds.
6. Drop the corner mask from the main pipeline; report it as a checked-and-rejected mitigation.

**Caveats.** Five seeds is still few; Montgomery is small (58 TB, 29 in each evaluation half), so per-seed violation estimates carry their own noise; one architecture.

**Next.** Experiments are frozen per the plan. Next: demo (P8) and paper (P9). Use the multi-seed tables as the main results.

---

## 2026-11-16 — P8 site-check demo

**Question.** Can the study's findings be turned into a usable decision rule for a new hospital?

**What was done.** `tbshift/site_check.py` holds the rule (no web code): defer below 20 images; re-certify when at least 22 TB cases are labelled; otherwise "re-certify before use" if a one-sided binomial test on the labelled TB cases rejects 90% sensitivity or the KS test sees a score shift; otherwise "plausible". FastAPI backend (model loaded once at startup, CPU), React frontend (Site check, Study results, Limitations), simulated scenarios with known ground truth so no patient images are needed.

**Result.** See PROGRESS.md (P8). Real Montgomery upload of 30 images: no shift detected (p = 0.052), consistent with P5/P6 power at this batch size. Simulated harmful shift: the check says "re-certify" every time, and the ground-truth panel shows the source threshold catching about 65% of TB.

**Conclusion.**

1. The rule never recommends the label-free threshold move, because the multi-seed study showed it is unsafe.
2. "Shift detected" is worded as "promise unverified", not "promise broken": the detector fired for all five seeds, but only one failed.
3. With only about 30 TB labels the re-certified threshold restores sensitivity but clears almost no healthy people. The demo shows this cost instead of hiding it.

**Next.** P9: paper and report.
