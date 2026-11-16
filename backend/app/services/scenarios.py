"""Simulated hospital batches, so the demo works without any patient images.

Each scenario resamples the source pool's logits within each class, adds a
little noise, and moves one or both classes by a multiple of that class's
standard deviation. Because the batch is simulated, the true labels are known,
so the demo can show what the site check could not see: whether the source
threshold really still catches 90% of TB cases.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tbshift.site_check import UNLABELLED, Reference


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    description: str
    n_tb: int
    n_normal: int
    tb_shift: float        # in units of the TB class's logit sd
    normal_shift: float    # in units of the normal class's logit sd
    labelled_tb: int = 0
    labelled_normal: int = 0


SCENARIOS = {
    s.name: s for s in [
        Scenario("similar", "Similar hospital",
                 "Same kind of patients and scanner as the source. Nothing should fire.",
                 30, 30, 0.0, 0.0),
        Scenario("harmless", "Scores drift upward",
                 "Every image scores higher, as at Montgomery for most training runs. "
                 "TB is still caught, but many more healthy people are flagged.",
                 30, 30, 1.0, 1.0),
        Scenario("harmful", "TB cases look milder",
                 "TB cases score lower than at the source, as for the training run whose "
                 "promise failed. No labels.",
                 30, 30, -1.6, 0.0),
        Scenario("harmful_labelled", "TB cases look milder, 30 labelled",
                 "The same harmful shift, but the site has labelled 30 TB and 30 normal cases.",
                 60, 60, -1.6, 0.0, labelled_tb=30, labelled_normal=30),
    ]
}


def simulate(reference: Reference, scenario: Scenario, seed: int = 0):
    """Return (logits, labels shown to the check, true labels)."""
    rng = np.random.default_rng(seed)
    parts, truth, shown = [], [], []
    for cls, n, shift, n_shown in [(1, scenario.n_tb, scenario.tb_shift, scenario.labelled_tb),
                                   (0, scenario.n_normal, scenario.normal_shift,
                                    scenario.labelled_normal)]:
        pool = reference.logits[reference.labels == cls]
        sd = float(pool.std())
        z = rng.choice(pool, size=n, replace=True) + rng.normal(0.0, 0.15 * sd, size=n) + shift * sd
        parts.append(z)
        truth.append(np.full(n, cls))
        shown.append(np.where(np.arange(n) < n_shown, cls, UNLABELLED))
    order = rng.permutation(scenario.n_tb + scenario.n_normal)
    return (np.concatenate(parts)[order], np.concatenate(shown)[order],
            np.concatenate(truth)[order])


def ground_truth(reference: Reference, logits: np.ndarray, truth: np.ndarray,
                 threshold: float) -> dict:
    """What actually happens at this simulated site (only knowable because it is simulated)."""
    tb, normal = truth == 1, truth == 0
    return {
        "source_threshold_sensitivity": float(np.mean(logits[tb] >= reference.threshold)),
        "source_threshold_specificity": float(np.mean(logits[normal] < reference.threshold)),
        "in_force_sensitivity": float(np.mean(logits[tb] >= threshold)),
        "in_force_specificity": float(np.mean(logits[normal] < threshold)),
        "target_sensitivity": 1 - reference.alpha,
    }
