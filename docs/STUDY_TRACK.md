# TB-Recertify — Study Track and Coding-Round Prep

Study the topic the same week you build with it. About 5 h/week of study beside 8–10 h of building.

For every topic: (1) read or watch, (2) explain it aloud in two minutes without notes, (3) write a tiny NumPy example, (4) add three lines to `docs/research_notebook.md`.

---

## 1. Weekly study map

| Week | Building | Study alongside | You should be able to answer |
|---|---|---|---|
| 1 | Metrics | NumPy vectorisation and broadcasting · conditional probability, Bayes' rule · confusion matrix, sensitivity, specificity, PPV/NPV · ROC and AUC | Why does PPV change with prevalence but sensitivity does not? What does AUC = 0.9 mean in one sentence? |
| 2 | Data | PyTorch tensors, autograd, `Dataset`/`DataLoader` · train/validation/test and why a separate *calibration* split · data leakage · stratified sampling | What goes wrong if the same patient is in train and test? |
| 3 | Model | Convolution, pooling, receptive field, batch norm · cross-entropy loss · SGD vs Adam, learning rate · overfitting, augmentation, early stopping · transfer learning · DenseNet vs ResNet | Why fine-tune a pretrained model instead of training from scratch on 800 images? |
| 4 | Calibration + conformal | Logits, softmax, NLL · calibration, reliability diagrams, ECE · temperature scaling · quantiles and order statistics · exchangeability · split conformal · marginal vs conditional coverage | A model with 95% accuracy says "0.99" on every prediction. Is it calibrated? |
| 5 | LOSO | Bootstrap · confidence intervals · binomial distribution, Clopper–Pearson · experimental design, controls · cross-validation variants | Why is leave-one-site-out harder than random k-fold? |
| 6 | Shift detection | Covariate, label and concept shift · two-sample tests (KS) · MMD · classifier two-sample test · false alarms vs power · multiple testing | Give one example each of covariate shift and label shift in a hospital. |
| 7 | Recalibration | Quantile matching, CDFs · importance weighting · weighted conformal · risk control (Learn-then-Test idea) | Why does matching score distributions fail if TB prevalence doubles? |
| 8 | Degradations | Image noise models · blur as convolution · JPEG basics · gamma and contrast · robustness benchmarks | Which degradation should hurt a chest X-ray model most, and why? |
| 9 | Demo | Model serving patterns · FastAPI lifespan and dependency injection · Docker layers · reproducibility | Why load the model once at startup instead of per request? |
| 10 | Paper | How to read and write a paper · limitations sections · honest claims | State your contribution in one sentence with no adjectives. |

---

## 2. Reading list

These are real and I am confident of the authors and topics. **Verify each citation yourself before it goes in your paper**, and log every one you read in `docs/reading_log.md` (problem, method, data, result, limitation, link to your project).

Start here (in this order):

1. Angelopoulos & Bates — *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. Your main conformal reference.
2. Guo et al. 2017 — *On Calibration of Modern Neural Networks*. Temperature scaling and ECE.
3. Jaeger et al. 2014 — the Montgomery and Shenzhen dataset paper.
4. Zech et al. 2018 (PLoS Medicine) — a pneumonia X-ray model that failed to generalise across hospitals. Your motivation paper.
5. DeGrave et al. 2021 (Nature Machine Intelligence) — COVID X-ray models learning shortcuts. Why the same-site rule exists.

Then:

6. Rabanser et al. 2019 — *Failing Loudly*. Shift detection methods.
7. Tibshirani et al. 2019 — *Conformal Prediction Under Covariate Shift*.
8. Vovk 2012 — conditional validity of inductive conformal predictors. The theory behind Finding 4 in the plan.
9. Bates et al. 2021 (risk-controlling prediction sets) and Angelopoulos et al. 2021 (*Learn then Test*). The "certified" threshold family.
10. Roschewitz et al. 2023 — unsupervised prediction alignment under acquisition shift. Check this is the "UPA" you meant.
11. Geirhos et al. 2020 — *Shortcut Learning in Deep Neural Networks*.
12. Huang et al. 2017 (DenseNet) and Rajpurkar et al. 2017 (CheXNet).
13. Hendrycks & Dietterich 2019 — corruption robustness benchmarks. Template for Phase 7.
14. WHO guidance on TB screening and its target product profile for triage tests. This is where a "~90% sensitivity" target comes from; quote the exact wording from the document, not from memory.
15. Keshav — *How to Read a Paper* (three-pass method). Read this before the others.

Plus the two 2026 preprints from Gate A.

Courses and tutorials (pick one per row, not all):

| Need | Option |
|---|---|
| Intuition for neural nets | 3Blue1Brown neural network series |
| PyTorch from zero | Official PyTorch "Learn the Basics" tutorials |
| Building nets by hand | Karpathy, *Neural Networks: Zero to Hero* (first 2–3 videos) |
| CNN theory | Stanford CS231n notes (convolution, training sections) |
| Statistics refreshers | StatQuest videos on ROC/AUC, bootstrap, p-values |

---

## 3. Coding-round prep

Three kinds of round are likely. The project already forces you to practise the second and third.

### A. ML from scratch (NumPy only, 20–30 min each, no looking things up)

Do one per week in the order below. Each is a function you write for the project anyway.

1. Confusion matrix, sensitivity, specificity, precision, F1.
2. AUC from scores and labels without scikit-learn.
3. Threshold that achieves at least 90% sensitivity on a validation set.
4. Stratified train/test split with a seed.
5. Sigmoid, softmax (numerically stable), binary and multi-class cross-entropy.
6. Logistic regression with gradient descent.
7. ECE with M bins.
8. Temperature scaling by grid search over T.
9. Empirical quantile and split-conformal threshold.
10. Bootstrap confidence interval for any metric.
11. KS statistic between two samples.
12. 2D convolution with a loop, then output-size formula with stride and padding.
13. K-means.
14. k-fold and leave-one-group-out index generators.
15. Mini-batch iterator with shuffling.

### B. PyTorch fluency (be able to type these from memory)

- A custom `Dataset` and `DataLoader`.
- A full training loop: zero grad, forward, loss, backward, step; `model.train()` vs `model.eval()`; `torch.no_grad()`.
- Freeze a backbone and replace the classifier head.
- Save and load a checkpoint.
- Move things to the GPU and explain a device mismatch error.
- Count parameters; print the shape after each layer.

### C. General DSA (if the company runs a standard coding round)

You have a web-dev background, so you have probably seen some of this. Cover patterns, not a problem count. Two to three problems per pattern:

arrays and hashing · two pointers · sliding window · binary search (including "binary search on the answer", which is the same idea as searching for a threshold) · stack · heap / top-k · linked list · trees (DFS, BFS) · graphs (BFS, DFS, topological sort) · basic dynamic programming · intervals.

Routine: 4 problems a week, 35 minutes each, talk aloud, write the complexity before running the code.

### Debugging questions interviewers like

- Training loss is not going down. List five causes in the order you would check them.
- Validation accuracy is 99% on a medical dataset. What do you suspect first?
- Test results change on every run. Why, and how do you fix it?
- The model is great on hospital A and poor on hospital B. What do you look at?

Your `bugs_log.md` becomes your answer bank for these.
