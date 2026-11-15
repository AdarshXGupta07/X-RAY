"""Temperature scaling (plan phase P3).

NOT IMPLEMENTED YET. The contracts below are pinned down by
``tests/test_calibration.py``.

Binary setting: the model outputs one logit z per image (the log-odds of TB).
Temperature scaling divides it by a single number T > 0 before the sigmoid:

    p = sigmoid(z / T)

T > 1 softens over-confident predictions, T < 1 sharpens them, T = 1 changes
nothing. T is fitted on a held-out calibration split, never on test data.
"""

from __future__ import annotations

import numpy as np


def apply_temperature(logits, temperature: float) -> np.ndarray:
    """Return TB probabilities ``sigmoid(logits / temperature)``."""
    raise NotImplementedError("phase P3")


def fit_temperature(logits, y_true) -> float:
    """Return the T > 0 that minimises negative log-likelihood on (logits, y_true)."""
    raise NotImplementedError("phase P3")
