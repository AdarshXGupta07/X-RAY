"""Rule-out thresholds with a sensitivity guarantee (plan phase P3).

Setting: we rule out every image whose score is below a threshold tau, and
flag every image with score >= tau. Sensitivity = P(score >= tau | TB).
Only TB-positive calibration scores are used, because sensitivity is a
statement about TB cases only. (A side effect: the threshold does not depend
on how common TB is at a site.)

Both thresholds below are one of the sorted calibration scores, s_(1) <= ... <= s_(n).
Choosing tau = s_(k) means "allow the k-1 lowest-scoring calibration TB cases
to fall below the line". The two methods differ only in how k is chosen.

Two different promises
----------------------
Marginal (split conformal), k = floor(alpha * (n + 1)):
    P(score_new >= tau | TB) >= 1 - alpha,
    where the probability is over BOTH the calibration set and the new case.
    Why: if the n calibration scores and the new score are exchangeable, the
    new score is equally likely to land in any of the n + 1 rank positions.
    It falls below s_(k) only in the lowest k positions: probability
    k / (n + 1) <= alpha.
    On average the promise holds; a particular calibration set can fall short.

Certified (PAC), k = the largest k with P(Binomial(n, alpha) <= k - 1) <= delta:
    P( sensitivity(tau) >= 1 - alpha ) >= 1 - delta,
    where the probability is over the calibration set only.
    Why: let q be the true score below which a fraction alpha of TB cases lie.
    Sensitivity of s_(k) is below 1 - alpha exactly when s_(k) > q, i.e. when
    fewer than k calibration scores fall below q. That count is
    Binomial(n, alpha), so the chance of a bad calibration set is
    P(Binomial(n, alpha) <= k - 1). We pick the largest k keeping it <= delta.
    This is the promise that controls the project's "violation rate".

Both assume calibration cases and future cases are exchangeable, which is
exactly what a new hospital can break.

When there are too few TB-positive scores, no k >= 1 satisfies the rule and
the only honest answer is the trivial threshold -inf: rule out nobody.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom

TRIVIAL_THRESHOLD = -math.inf

# Guards against floating-point products like 0.29 * 100 = 28.999999999999996.
_EPS = 1e-9


def _check_level(value: float, name: str) -> None:
    if not 0 < value < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1, got {value}")


def _sorted_scores(pos_scores) -> np.ndarray:
    s = np.asarray(pos_scores, dtype=float)
    if s.ndim != 1:
        raise ValueError("pos_scores must be 1-D")
    if np.isnan(s).any():
        raise ValueError("pos_scores contains NaN")
    return np.sort(s)


def marginal_rank(n: int, alpha: float) -> int:
    """Rank k (1-based) of the order statistic used by the marginal threshold.

    Returns 0 when no non-trivial threshold exists.
    """
    _check_level(alpha, "alpha")
    if n < 0:
        raise ValueError("n must be >= 0")
    return math.floor(alpha * (n + 1) + _EPS)


def certified_rank(n: int, alpha: float, delta: float) -> int:
    """Rank k (1-based) of the order statistic used by the certified threshold.

    Returns 0 when no non-trivial threshold can be certified.
    """
    _check_level(alpha, "alpha")
    _check_level(delta, "delta")
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return 0
    ks = np.arange(1, n + 1)
    # P(bad calibration set) for each candidate k. It grows with k, so the
    # acceptable ks form a prefix 1..k_max and counting them gives k_max.
    p_bad = binom.cdf(ks - 1, n, alpha)
    return int(np.sum(p_bad <= delta))


def conformal_threshold(pos_scores, alpha: float) -> float:
    """Marginal threshold from TB-positive calibration scores.

    Returns ``-inf`` (rule out nobody) when there are too few positives for a
    non-trivial threshold at this alpha.
    """
    s = _sorted_scores(pos_scores)
    k = marginal_rank(s.size, alpha)
    return TRIVIAL_THRESHOLD if k < 1 else float(s[k - 1])


def certified_threshold(pos_scores, alpha: float, delta: float) -> float:
    """PAC threshold: sensitivity >= 1 - alpha with probability >= 1 - delta.

    Returns ``-inf`` when there are too few positives to certify anything.
    """
    s = _sorted_scores(pos_scores)
    k = certified_rank(s.size, alpha, delta)
    return TRIVIAL_THRESHOLD if k < 1 else float(s[k - 1])


def min_positives_marginal(alpha: float) -> int:
    """Smallest number of TB-positive calibration cases giving a finite marginal threshold.

    Closed form: need floor(alpha * (n + 1)) >= 1, i.e. n >= 1/alpha - 1.
    At alpha = 0.10 that is 9.
    """
    _check_level(alpha, "alpha")
    n = max(0, math.ceil(1 / alpha) - 2)  # start just below the closed form
    while marginal_rank(n, alpha) < 1:
        n += 1
    return n


def min_positives_certified(alpha: float, delta: float) -> int:
    """Smallest number of TB-positive calibration cases giving a finite certified threshold.

    Closed form: k = 1 is certifiable when P(no calibration score below q)
    = (1 - alpha)^n <= delta, i.e. n >= log(delta) / log(1 - alpha).
    At alpha = delta = 0.10 that is 22.
    """
    _check_level(alpha, "alpha")
    _check_level(delta, "delta")
    n = max(0, math.floor(math.log(delta) / math.log(1 - alpha)) - 1)
    while certified_rank(n, alpha, delta) < 1:
        n += 1
    return n
