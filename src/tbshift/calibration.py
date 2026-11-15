"""Temperature scaling for a binary TB classifier (plan phase P3).

Binary setting: the model outputs one logit z per image (the log-odds of TB).
Temperature scaling divides it by a single number T > 0 before the sigmoid:

    p = sigmoid(z / T)

T > 1 softens over-confident predictions, T < 1 sharpens them, T = 1 changes
nothing. T is fitted on a held-out calibration split, never on test data.

Because z / T keeps the ORDER of the scores, temperature scaling never changes
AUC or which images fall above a rank-based threshold. It only changes how
trustworthy the probability numbers are (ECE, NLL).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

# Search range for T. Outside this range something is wrong with the logits
# (for example they carry no signal), and the fit stops at the boundary.
T_MIN = 0.01
T_MAX = 100.0


def _as_logits(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    if z.ndim != 1 or z.size == 0:
        raise ValueError("logits must be a non-empty 1-D array")
    if not np.isfinite(z).all():
        raise ValueError("logits contain NaN or infinity")
    return z


def _check_temperature(temperature: float) -> None:
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError(f"temperature must be a positive finite number, got {temperature}")


def apply_temperature(logits, temperature: float) -> np.ndarray:
    """Return TB probabilities ``sigmoid(logits / temperature)``.

    ``expit`` is SciPy's numerically safe sigmoid: it does not overflow for
    very large or very negative logits.
    """
    _check_temperature(temperature)
    return expit(_as_logits(logits) / temperature)


def binary_nll(logits, y_true, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of the labels under ``sigmoid(logits / T)``.

    For x = z / T:
        TB case     (y = 1):  -log(sigmoid(x))     = log(1 + e^x) - x
        non-TB case (y = 0):  -log(1 - sigmoid(x)) = log(1 + e^x)
    so both cases are ``log(1 + e^x) - y * x``. ``np.logaddexp(0, x)`` computes
    log(1 + e^x) without overflow.
    """
    _check_temperature(temperature)
    z = _as_logits(logits)
    y = np.asarray(y_true)
    if y.shape != z.shape:
        raise ValueError(f"length mismatch: {y.shape} labels vs {z.shape} logits")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    x = z / temperature
    return float(np.mean(np.logaddexp(0.0, x) - y * x))


def fit_temperature(logits, y_true) -> float:
    """Return the T > 0 that minimises negative log-likelihood on (logits, y_true).

    We search over log(T) rather than T so the optimiser treats "twice as hot"
    and "twice as cold" symmetrically and can never propose T <= 0.
    The NLL has a single minimum along this direction (it is convex in 1/T),
    so a bounded one-dimensional search is enough.
    """
    z = _as_logits(logits)
    y = np.asarray(y_true)
    if y.shape != z.shape:
        raise ValueError(f"length mismatch: {y.shape} labels vs {z.shape} logits")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must contain only 0 and 1")
    if y.min() == y.max():
        raise ValueError("calibration data needs both TB and non-TB cases")

    result = minimize_scalar(
        lambda log_t: binary_nll(z, y, math.exp(log_t)),
        bounds=(math.log(T_MIN), math.log(T_MAX)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return float(math.exp(result.x))
