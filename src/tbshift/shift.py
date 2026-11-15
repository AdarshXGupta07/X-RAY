"""Label-free shift detection (plan phase P5).

NOT IMPLEMENTED YET. The contracts below are pinned down by
``tests/test_shift.py``.

Question: given a batch of UNLABELLED images from a new site, can we tell
that it differs from the source sites? Two views of the same batch:

* scores      - the model's 1-D output; compare distributions with a
                two-sample Kolmogorov-Smirnov test.
* embeddings  - the model's feature vectors; train a small classifier to tell
                source from target. If it cannot (AUC near 0.5) there is no
                detectable shift; if it can (AUC near 1) there is.

A detector firing does NOT prove the sensitivity guarantee broke. Measuring
how often the two coincide is part of research question 2.
"""

from __future__ import annotations

from typing import NamedTuple


class ShiftResult(NamedTuple):
    statistic: float
    p_value: float
    shifted: bool  # True when p_value < significance level


def ks_statistic(a, b) -> float:
    """Largest vertical gap between the empirical CDFs of two 1-D samples (0..1)."""
    raise NotImplementedError("phase P5")


def detect_score_shift(source_scores, target_scores, significance: float = 0.05) -> ShiftResult:
    """Two-sample KS test on model scores."""
    raise NotImplementedError("phase P5")


def domain_classifier_auc(source_embeddings, target_embeddings, seed: int = 0) -> float:
    """Cross-validated AUC of a simple classifier separating source from target."""
    raise NotImplementedError("phase P5")
