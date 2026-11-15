# TB-Recertify — Interview Prep

Fill every `[__]` from your own `results/` files. Never quote a number you cannot regenerate, and never say a result before you have it.

---

## 1. The pitch

**15 seconds**

> I studied what happens to a chest X-ray TB model's safety promise when it moves to a hospital it never saw. I showed how often a "we catch 90% of TB" guarantee silently breaks, whether you can detect that from images alone, and how many labelled cases it takes to restore it.

**60 seconds** (problem → why it is hard → what you did → what you found → honest limit)

> A TB screening model is tuned to catch at least 90% of cases. That number is measured at the hospitals it was trained on. A new hospital has a different X-ray machine, and the promise can break without anyone noticing, because nobody has labels there yet. The dangerous error is a missed TB case.
>
> I took one fixed CNN and used split-conformal calibration to set a rule-out threshold with a target sensitivity. Then I ran leave-one-site-out tests on public datasets and compared five strategies: a fixed threshold, source-calibrated conformal, label-free recalibration, a shift detector that defers, and re-certifying with a small number of local labels.
>
> I found [__]. The guarantee held in [__] of [__] unseen-site runs, the detector needed about [__] images to flag a shift, and restoring the guarantee took about [__] labelled TB cases.
>
> It is a research prototype on public data with no clinical validation, and with [__] sites the conclusions are a case study, not a general law.

---

## 2. Stories to prepare (situation → action → result → lesson)

Write each in five sentences once it has happened. Do not prepare stories for things that did not happen.

1. **A finding that changed the design.** Candidate: discovering that plain conformal does not control the violation rate, and adding a certified threshold.
2. **A bug you chased down.** Take the best one from `bugs_log.md`.
3. **An engineering decision.** Caching scores so thousands of repeats run on a 4 GB laptop GPU.
4. **A time you chose honesty over a nicer result.** Dataset rejected for mixed sources, or a claim you narrowed after the literature check.
5. **Scope control.** Time-boxing the demo; freezing experiments.

---

## 3. Question bank with answer skeletons

Skeletons are prompts for your own words, not scripts.

**Method**

- *What is conformal prediction, simply?* Use held-out data to pick a cut-off so that a statement holds with a chosen probability, assuming new data looks like the held-out data. No assumption about the model being right.
- *What assumption does it need, and where does your project break it?* Exchangeability. A new hospital breaks it. That break is the subject of the project.
- *Why calibrate on TB-positive cases only?* Sensitivity is a statement about positives. It also makes the threshold insensitive to prevalence changes.
- *Marginal vs conditional guarantee?* Marginal holds on average over calibration sets. A single calibration set can land below target. Explain how you measured both.
- *What does temperature scaling do, and what does it not do?* Fixes over-confidence with one parameter. It keeps the ranking of scores, so [your tested answer on AUC and rank-based thresholds].
- *Why DenseNet-121? Why one fixed model?* Common chest X-ray baseline; fixing the model isolates the effect of calibration from the effect of architecture.

**Evaluation**

- *Why leave-one-site-out rather than a random split?* A random split mixes hospitals and hides the exact failure you are studying.
- *Why sensitivity and not accuracy?* Unequal error costs; a missed TB case is worse than an unnecessary follow-up. Accuracy also depends on prevalence.
- *How did you get confidence intervals? Why are they wide?* Bootstrap within class. Small numbers of TB-positive images per site.
- *What is your violation rate, exactly?* Give the precise definition you used and say which guarantee it tests.
- *How do you know it is site shift and not a bug?* The in-distribution control: same pipeline, same-site test data, guarantee holds.

**Data and shortcuts**

- *What is shortcut learning? Did you check for it?* Model learns the hospital, not the disease. Same-site rule, the site-classifier check, and the dataset you rejected.
- *How did you prevent leakage?* Split tests, patient-level separation where IDs exist, thresholds fitted only on calibration splits.

**Shift and recalibration**

- *How can you detect shift without labels?* Compare distributions of scores or embeddings between source and new-site images.
- *Does the detector firing mean the model is unsafe?* Not necessarily. Report what you found about detection vs actual harm.
- *When does label-free recalibration fail?* When TB prevalence differs, because it matches the mixed score distribution.
- *How many labels do you need, and is there a hard minimum?* Your empirical curve plus the floor you derived.

**Engineering**

- *Walk me through the repo.* Research package, tests, experiments driven by configs, cached scores, API, frontend.
- *How is it reproducible?* Seeds, configs and commit hash stored with results; one command regenerates figures.
- *How does the demo work?* Batch in, shift statistic and verdict out; model loaded once; CPU inference.

**Judgement**

- *What is novel here?* Say exactly what Gate A let you claim, name the closest work, and state the difference. If it is mainly a careful empirical study, say that.
- *Biggest limitation?* Few sites, small datasets, public data only, no clinical validation.
- *Would you deploy this?* No. Say what would be needed: prospective multi-site data, clinical oversight, regulatory review.
- *What would you do with three more months?* More sites, more than one model, prevalence-aware label-free method, prospective-style evaluation.

---

## 4. Things not to say

| Avoid | Say instead |
|---|---|
| "It detects TB with X% accuracy." | "On public dataset [__], sensitivity was [__] with interval [__]." |
| "It guarantees 90% sensitivity." | "It guarantees it under exchangeability; I measured what happens when that fails." |
| "This is the first work to…" | "The closest work I found is [__]; mine differs in [__]." |
| "It can be used in hospitals." | "It is a research prototype with no clinical validation." |
| "I used conformal prediction." (and stop) | Explain it in two sentences before they ask. |

---

## 5. Resume lines (fill after results exist)

- Built a reproducible evaluation of site-shift failure for a chest X-ray TB classifier using split-conformal calibration and leave-one-site-out testing across [__] public datasets ([__] images).
- Showed a source-calibrated [__]% sensitivity target was violated in [__]% of unseen-site runs; restored it with about [__] labelled positives.
- Implemented label-free shift detection ([__] images to reach [__]% power) and a FastAPI + React "site check" demo; [__] tests, Dockerised.

---

## 6. Mock schedule

- End of week 4: explain conformal prediction to a non-ML friend in 3 minutes. Record it.
- End of week 7: 20-minute mock with me on Method + Evaluation questions.
- End of week 10: full 45-minute mock (pitch, deep dive, one NumPy coding task, limitations).
