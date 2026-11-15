# Dataset Audit (Gate B)

**First pass:** 2026-11-15, web search only. **Nothing has been downloaded.**
A dataset can serve as a "site" only if TB and non-TB images were collected at the same place by the same process. Otherwise a model can learn "which hospital" instead of "TB".

Status key: ✅ confirmed from a primary source today · ⚠️ from a secondary source or memory, verify · ❓ unknown.

## 1. Candidate sites

| Dataset | Origin | Images (TB / non-TB) | Both classes same site? | Licence / terms | Access | Verdict |
|---|---|---|---|---|---|---|
| **Montgomery County** | TB screening programme, Montgomery County, Maryland, USA. 4020×4892 PNG, 12-bit ✅ | 138 (58 / 80) ✅ | Yes ✅ | ✅ NLM ReadMe: **do not share outside your research group**; report errors to NLM; attribute NLM and cite Candemir et al. 2014 (IEEE TMI 33(2):577–90) and Jaeger et al. 2014 (IEEE TMI 33(2):233–45) | ✅ Direct download from NLM: `openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip` (616,853,875 bytes). Same files listed at `data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/` | **Use. Downloaded 2026-11-15**, SHA-256 `54601e95…8f424`; zip integrity OK; manifest counts 58 / 80 match. Small: only 58 TB cases, so wide intervals |
| **Shenzhen** | Shenzhen No. 3 People's Hospital, China; out-patient clinics; Philips DR systems; about 3K×3K PNG ✅ | 662 (336 / 326) ✅ | Yes ✅ (one hospital, routine capture) | ✅ Same terms as Montgomery; also attribute Shenzhen No. 3 People's Hospital, Guangdong Medical College; same two citations | ✅ Official per-file folders at `data.lhncbc.nlm.nih.gov/public/Tuberculosis-Chest-X-ray-Datasets/Shenzhen-Hospital-CXR-Set/` (662 PNG, 3,594 MB; 662 readings). The zip on `openi.nlm.nih.gov` returned an error page on 2026-11-15 | **Use. Downloaded 2026-11-16**: 1,324 files, every PNG size equal to NLM's listing; counts 336 / 326 match |
| **DA** | National Institute of TB and Respiratory Diseases, New Delhi; X-ray machine A, 8-bit ✅ | ❓ | Yes ⚠️ (one institute) | ❓ | ⚠️ SourceForge project `tbxpredict`; not confirmed still available | **Use if obtainable.** |
| **DB** | Same institute, a different machine, 14-bit ✅ | ❓ | Yes ⚠️ | ❓ | Same as DA ⚠️ | **Use if obtainable.** DA vs DB is a rare "same hospital, different scanner" pair, which is exactly the shift in the project title |
| **Mendeley "Dataset of Tuberculosis Chest X-rays Images"** (Kiran & Jabeen), DOI 10.17632/8j2g3csprk.3 | "A local hospital in Pakistan" ⚠️ (hospital not named) | 3,008 (2,494 / 514) ⚠️ | Claimed ⚠️ | CC BY 4.0 for versions 1–2 ⚠️; version 3 not confirmed | Mendeley Data, direct | **Inspect before trusting.** Heavily imbalanced. Another group's DenseNet baseline scored about chance (AUROC ≈ 0.50) on it and they describe it as a JPEG-artifact stress case, which is a warning sign for artifacts or class-specific processing |
| **TBX11K** | Media Computing Lab, Nankai University; several hospitals in China ⚠️ | 11,200 total ⚠️ (healthy / sick-non-TB / TB groups) | ❓ No per-image hospital ID found | CC BY 4.0 per the official project page ⚠️ | Official page mmcheng.net/tb; released at 512×512 ⚠️ | **One composite site at most.** Cannot be split into hospitals. Check whether test-split labels are public |

## 2. Excluded

| Dataset | Reason |
|---|---|
| Belarus TB portal images | TB-positive only. One class cannot define a site |
| Kaggle-style merged "TB Chest X-ray Database" collections | TB and normal images come from different sources. Textbook shortcut-learning trap. Keep as the cautionary example in the report |
| NIH ChestX-ray14 | No TB label; about 45 GB, more than twice the free disk space |

## 3. Leads not yet identified

The closest related preprint (reading log C1) evaluates on cohorts it calls "India Solan (155 images)" and "NITRD New Delhi (278 images)". These may be public sets, possibly related to DA/DB. Find their source in that preprint's data section.

## 4. What this means for the experiment design

- **Confirmed today: two usable sites** (Montgomery, Shenzhen). That gives two leave-one-site-out folds, which is thin.
- **Realistic target: four to six sites** if DA, DB, the Mendeley set and TBX11K pass inspection.
- Montgomery has 58 TB cases. The certified threshold needs about 22 TB-positive calibration cases at 90% sensitivity with 90% confidence, leaving roughly 36 to measure achieved sensitivity. Every Montgomery number needs an interval and many repeats.
- Approximate disk need for Montgomery + Shenzhen + TBX11K (512 px): under 10 GB ⚠️. Free space on D: was about 21 GB before PyTorch was installed. Check before downloading.

## 5. Consequences of the NLM terms for this repo

- Images, extracted folders and manifests (which hold patient age and sex) stay git-ignored. Only code and aggregate results are committed.
- The web demo must not ship or display NLM images publicly. Use synthetic or user-supplied images in any public demo.
- The two IEEE TMI 2014 citations and the NLM attribution go in the paper's data section and the README.

## 6. To do before any training

- [x] Open the official NLM page, record the exact download route and the terms of use text for Montgomery and Shenzhen. (Done 2026-11-15; ReadMe copies in `data/raw/docs/`, git-ignored.)
- [ ] Get exact DA and DB counts and terms from Chauhan et al. 2014 (PLOS ONE, doi 10.1371/journal.pone.0112980) and check the SourceForge files.
- [ ] Read the Mendeley dataset description in full; look at 50 random images per class for borders, text markers, resolution and compression differences between classes.
- [ ] Confirm TBX11K licence, label availability and class sourcing on the official page.
- [ ] For each accepted dataset record: access date, URL, licence, file checksum, counts found on disk.
- [ ] Per-site sanity check after download: can a tiny classifier separate TB from non-TB using only a 32×32 thumbnail or only the image borders? If yes, there is a shortcut.
