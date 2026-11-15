"""Restoring the guarantee at a new site (plan phase P6).

NOT IMPLEMENTED YET. The contracts below are pinned down by
``tests/test_recalibration.py``.

Two routes:

Label-free (quantile matching)
    Map each target score to the source score at the same quantile, so the
    target score distribution lines up with the source one, then reuse the
    source threshold. Needs no labels, but silently assumes the mix of TB and
    non-TB cases is the same at both sites.

Few-label re-certification
    Spend a small number of labelled target cases and recompute the threshold
    from the target TB-positives among them.
"""

from __future__ import annotations

import numpy as np


def quantile_match(target_scores, source_scores) -> np.ndarray:
    """Map target scores onto the source score distribution, preserving their order."""
    raise NotImplementedError("phase P6")


def few_label_threshold(target_scores, target_labels, alpha: float, delta: float | None = None) -> float:
    """Threshold recomputed from labelled target cases.

    ``delta=None`` gives the marginal conformal threshold; a float gives the
    certified one. Returns ``-inf`` when the labelled sample has too few
    TB-positives.
    """
    raise NotImplementedError("phase P6")
