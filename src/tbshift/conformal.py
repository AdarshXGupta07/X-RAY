"""Rule-out thresholds with a sensitivity guarantee (plan phase P3).

NOT IMPLEMENTED YET. The contracts below are pinned down by
``tests/test_conformal.py``. This is the core of the project.

Setting: we rule out every image whose score is below a threshold tau.
Sensitivity = P(score >= tau | TB). Only TB-positive calibration scores are
needed, because sensitivity is a statement about TB cases only.

Two different promises
----------------------
Marginal (split conformal):
    P(score_new >= tau | TB) >= 1 - alpha,
    where the probability is over BOTH the calibration set and the new case.
    On average the promise holds; a particular calibration set may fall short.

Certified (PAC):
    P( sensitivity(tau) >= 1 - alpha ) >= 1 - delta,
    where the probability is over the calibration set only. This is the
    promise that controls the project's "violation rate".

Both assume the calibration cases and future cases are exchangeable, which is
exactly what a new hospital can break.
"""

from __future__ import annotations

import math


def conformal_threshold(pos_scores, alpha: float) -> float:
    """Marginal threshold from TB-positive calibration scores.

    Returns ``-inf`` (rule out nobody) when there are too few positives for a
    non-trivial threshold at this alpha.
    """
    raise NotImplementedError("phase P3")


def certified_threshold(pos_scores, alpha: float, delta: float) -> float:
    """PAC threshold: sensitivity >= 1 - alpha with probability >= 1 - delta.

    Returns ``-inf`` when there are too few positives to certify anything.
    """
    raise NotImplementedError("phase P3")


def min_positives_marginal(alpha: float) -> int:
    """Smallest number of TB-positive calibration cases giving a finite marginal threshold."""
    raise NotImplementedError("phase P3")


def min_positives_certified(alpha: float, delta: float) -> int:
    """Smallest number of TB-positive calibration cases giving a finite certified threshold."""
    raise NotImplementedError("phase P3")


TRIVIAL_THRESHOLD = -math.inf
