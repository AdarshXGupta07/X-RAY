"""tbshift: research code for the TB-Recertify project.

Convention used everywhere in this package
-------------------------------------------
* ``y_true``  : 1 = TB, 0 = not TB.
* ``y_score`` : higher means "more likely TB".
* A case is FLAGGED (kept for follow-up) when ``score >= threshold``.
* A case is RULED OUT (sent home) when ``score < threshold``.

So a missed TB case (false negative) is a TB image whose score fell below the
threshold. Sensitivity is the fraction of TB images that were flagged.
"""

__version__ = "0.1.0"
