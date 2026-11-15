# Reading Log and Literature Check (Gate A)

**First pass:** 2026-11-15, web search only.
**Honest status:** every entry below was logged from a search summary, an abstract page or a repository README. **None has been read in full yet.** Before citing any of them, open the paper and confirm the title, authors, year and the claim you are citing. "Read?" stays ☐ until you have.

How to log a paper once you read it: problem · method · data · main result · limitation · what it means for this project.

---

## 1. Closest work (read these first)

| # | Work | What it appears to do | What it appears NOT to do | Read? |
|---|---|---|---|---|
| C1 | M. Farrel Aditya, 2026, Research Square preprint, DOI 10.21203/rs.3.rs-11034929/v1. Code: github.com/SeraKah-1/tb-conformal-triage. **Title unclear:** the repo README gives "Selective Conformal AI Architecture for Tuberculosis Triage: Mitigating Sensor Shift and False-Negative Risks via Pre-Analytic Quality Assurance and Clinical Deferral Interlocks"; a search summary gave a different title for the same DOI. Check the DOI page. | TB chest X-ray triage with a DenseNet-121, calibrated P(TB), conformal "Non-TB / Defer" outputs, image-quality checks, an out-of-distribution module, external tests on cohorts from India and Pakistan. A second linked preprint distils a foundation model for in-browser use. | The README reports no few-label recalibration results and no label-free shift-detection results. Conformal variant and target coverage are not stated on the README. | ☐ |
| C2 | Mehrtens et al., 2025, "Pitfalls of Conformal Predictions for Medical Image Classification", arXiv 2506.18162 | Shows conformal coverage holds in-distribution and fails under domain shift (histopathology example); says recalibration on a new calibration set is required; warns that coverage is unreliable for subsets such as single classes. | Not TB. No label-budget study that I could see. | ☐ |
| C3 | Yilmaz & Heckel, 2022, "Test-time Recalibration of Conformal Predictors Under Distribution Shift Based on Unlabeled Examples", arXiv 2210.04166 | Recalibrates conformal predictors using unlabelled target data. This is the general-purpose version of our label-free arm. | Not medical, no sensitivity-specific target. | ☐ |
| C4 | Roschewitz et al., 2023, Nature Communications (s41467-023-42396-y), unsupervised prediction alignment (UPA) | Label-free correction of prediction drift under acquisition shift using only unlabelled images from the new site. Confirmed as the "UPA" in our method list. | Not conformal; no formal sensitivity bound. | ☐ |
| C5 | Biswas et al., 2026, "Few-Shot Cross-Dataset Adaptation for Tuberculosis Detection Using DenseNet", arXiv 2608.21427 | TB, DenseNet-121, source TBX11K → target Mendeley (Pakistan) set, varies the number of labelled target images; headline 98.36% accuracy at 75 labels per class. | Abstract reports accuracy only: no sensitivity target, no calibration, no guarantee. It adapts the model; we keep the model fixed and adapt only the threshold. | ☐ |
| C6 | Retinal OCT cross-site study using Mondrian (class-conditional) conformal prediction, PMC13464582 | Reports that site-specific calibration on roughly 10% of target data is needed. | Different modality. Details unverified. | ☐ |

## 2. Retinal candidates for "the retinal conformal-under-site-shift preprint"

I could not identify a single preprint matching that description. Candidates found; confirm which one you meant:

| # | Work | Note | Read? |
|---|---|---|---|
| R1 | Sulman et al., Nov 2026, "RetinaGuard: Risk-Controlled Agentic Triage for Multi-Disease Retinal Screening", arXiv 2611.10689 | Conformal risk control for accepting automated decisions. A search summary says the guarantee failed on an external dataset; the abstract itself does not mention site shift. | ☐ |
| R2 | "Uncertainty-Aware Last-Layer Adaptation of RETFound for Referable Diabetic Retinopathy Screening Under Dataset Shift", arXiv 2607.02569 | Large sensitivity drop when moving between retinal datasets. | ☐ |
| R3 | "Uncertainty-Aware AI-Assisted Diabetic Retinopathy Grading from Fundus Images with Ordinal Conformal Prediction", Diagnostics 2026, doi 10.3390/diagnostics16162645 | Conformal sets did not restore a formal guarantee after dataset shift. | ☐ |
| R4 | "Fundus Image-based Visual Acuity Assessment with PAC-Guarantees", arXiv 2412.06624 | Useful for the certified (PAC) threshold idea. | ☐ |

## 3. Also relevant

| # | Work | Note | Read? |
|---|---|---|---|
| A1 | "Conformal selective prediction with cost aware deferral for safe clinical triage under distribution shift", Scientific Reports 2026, s41598-026-40637-w | Could not open the article. A secondary page suggests it is about sepsis triage, not imaging. Unverified. | ☐ |
| A2 | "Estimating Test Performance for AI Medical Devices under Distribution Shift with Conformal Prediction", arXiv 2207.05796 | Estimating performance on unlabelled shifted data. | ☐ |
| A3 | "Missingness-Aware Conformal Prediction Under Cross-Hospital Distribution Shift", ICML 2026 | Cross-hospital conformal; probably tabular. Abstract only. | ☐ |

## 4. Foundations (from the study track; not searched today)

Angelopoulos & Bates (conformal introduction) · Guo et al. 2017 (calibration) · Jaeger et al. 2014 (Montgomery and Shenzhen) · Zech et al. 2018 · DeGrave et al. 2021 · Rabanser et al. 2019 · Tibshirani et al. 2019 · Vovk 2012 · Bates et al. 2021 · Geirhos et al. 2020.

---

## 5. Preliminary novelty assessment

This is an assessment from abstracts, not a verified claim.

**Already established by others, so do not claim as new:**

- Conformal guarantees break under site or domain shift in medical imaging (C2, R3).
- Conformal prediction with deferral for TB chest X-rays, tested on external cohorts (C1).
- Label-free recalibration using unlabelled target data (C3, C4).
- Few labelled target images help a TB model cross datasets (C5).

**Not found in this first pass, so possibly your contribution:**

1. Measuring the guarantee as a **violation rate over calibration sets**, separating the marginal promise from a certified one, for TB rule-out specifically.
2. The **label floor**: the minimum number of TB-positive labels below which no threshold can be certified, with budgets counted in positives.
3. A **head-to-head of five strategies with the model held fixed** (fixed threshold, source conformal, label-free, detect-and-defer, few-label) under leave-one-site-out.
4. Whether a label-free detector fires **when the guarantee actually breaks**, as opposed to whenever the images look different.

**Honest framing if this holds up:** a careful empirical and statistical study, not a new algorithm. That is a legitimate paper type; say so plainly.

**Still to do before any novelty sentence goes in the paper:**

- [ ] Read C1 in full. It is the closest and decides how the introduction is written.
- [ ] Read C2, C3, C5.
- [ ] Search Google Scholar and Semantic Scholar for papers citing C1, C2 and C4.
- [ ] Repeat the search the week before submission; this area is moving fast.
- [ ] Show the "how we differ" table to your supervisor.
