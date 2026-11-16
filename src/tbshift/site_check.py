"""Site check: should a new hospital trust the source rule-out threshold? (phase P8)

This is the logic behind the demo, kept free of any web or PyTorch code so it
can be unit-tested on plain numbers.

Inputs
------
* A ``Reference``: what is known from the source hospital. The model's logits
  on the source calibration pool, the fitted temperature, and the certified
  90%-sensitivity threshold computed from that pool's TB cases.
* A batch of logits from the new site, and optionally a label (0/1) for some
  of them.

Verdict (in this order)
-----------------------
``defer``        Too few images to say anything (below ``MIN_IMAGES``).
``recertified``  Enough labelled TB cases at the new site (``min_positives_certified``,
                 22 at alpha = delta = 0.10) to certify a new threshold there.
                 Labels beat the detector, so this wins whenever it is possible.
``recalibrate``  The labelled TB cases already show too many misses (binomial
                 test), or the score distribution differs from the source (KS
                 test). The source promise is unverified here; collect TB labels.
``plausible``    No shift detected. The promise is plausible, not proven.

Why not offer the label-free threshold move as a fix? In this study it broke
the promise in 27-80% of calibration draws at the real unseen site, depending
on the training seed. And why does "shift detected" not mean "promise broken"?
The detector fired for all five training seeds at Montgomery, while the
promise failed for only one of them. Only labels separate the two cases.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import binom

from tbshift.calibration import apply_temperature, fit_temperature
from tbshift.conformal import TRIVIAL_THRESHOLD, certified_threshold, min_positives_certified
from tbshift.shift import detect_score_shift

ALPHA = 0.10            # allowed miss rate: the promise is sensitivity >= 90%
DELTA = 0.10            # allowed chance that the certification itself is wrong
SIGNIFICANCE = 0.05     # KS test level
MIN_IMAGES = 20         # below this the detector has too little power to be useful
HISTOGRAM_BINS = 20
UNLABELLED = -1         # label value for "no label given"


@dataclass(frozen=True)
class Reference:
    """What the demo knows about the source hospital."""

    model: str
    source_site: str
    logits: np.ndarray          # source calibration pool (cal + test splits)
    labels: np.ndarray
    temperature: float
    threshold: float            # certified threshold, logit scale
    alpha: float = ALPHA
    delta: float = DELTA

    @property
    def n_tb(self) -> int:
        return int(np.sum(self.labels == 1))

    @property
    def ruled_out_fraction(self) -> float:
        return float(np.mean(self.logits < self.threshold))

    def summary(self) -> dict:
        return {
            "model": self.model,
            "source_site": self.source_site,
            "n_images": int(self.logits.size),
            "n_tb": self.n_tb,
            "temperature": self.temperature,
            "threshold_logit": _finite_or_none(self.threshold),
            "threshold_probability": _probability_or_none(self.threshold, self.temperature),
            "ruled_out_fraction": self.ruled_out_fraction,
            "target_sensitivity": 1 - self.alpha,
            "confidence": 1 - self.delta,
            "tb_labels_needed": min_positives_certified(self.alpha, self.delta),
        }


def build_reference(logits, labels, model: str, source_site: str,
                    alpha: float = ALPHA, delta: float = DELTA) -> Reference:
    """Fit temperature and the certified threshold on a labelled source pool."""
    z = np.asarray(logits, dtype=float)
    y = np.asarray(labels).astype(int)
    if z.shape != y.shape or z.ndim != 1:
        raise ValueError("logits and labels must be 1-D arrays of the same length")
    temperature = fit_temperature(z, y)
    threshold = certified_threshold(z[y == 1], alpha, delta)
    if threshold == TRIVIAL_THRESHOLD:
        raise ValueError("source pool has too few TB cases to certify a threshold")
    return Reference(model, source_site, z, y, temperature, threshold, alpha, delta)


def reference_from_score_cache(scores_npz: Path | str, model: str, source_site: str) -> Reference:
    """Build the reference from a P2 score cache file, using the cal + test splits (as in P4)."""
    with np.load(scores_npz, allow_pickle=False) as f:
        pool = np.isin(f["splits"], ("cal", "test"))
        return build_reference(f["logits"][pool], f["labels"][pool], model, source_site)


@dataclass
class SiteCheck:
    verdict: str
    headline: str
    reasons: list[str]
    n_images: int
    n_labelled: int
    n_labelled_tb: int
    tb_labels_needed: int
    ks_statistic: float | None
    ks_p_value: float | None
    shift_detected: bool | None
    source_threshold_logit: float
    ruled_out_fraction_source: float      # same threshold, on the source pool
    ruled_out_fraction_site: float        # source threshold, on this batch
    labelled_tb_missed: int | None        # labelled TB cases the source threshold would rule out
    recertified_threshold_logit: float | None
    ruled_out_fraction_recertified: float | None
    threshold_in_force_logit: float
    probabilities: list[float]            # temperature-scaled, for display only
    ruled_out: list[bool]                 # under the threshold in force
    histogram: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        for key in ("source_threshold_logit", "recertified_threshold_logit", "threshold_in_force_logit"):
            out[key] = _finite_or_none(out[key])
        return out


def site_check(reference: Reference, logits, labels=None) -> SiteCheck:
    """Run the site check on one batch from a new hospital."""
    z = np.asarray(logits, dtype=float)
    if z.ndim != 1 or z.size == 0:
        raise ValueError("logits must be a non-empty 1-D array")
    if not np.isfinite(z).all():
        raise ValueError("logits must be finite numbers")
    y = np.full(z.size, UNLABELLED) if labels is None else np.asarray(labels).astype(int)
    if y.shape != z.shape:
        raise ValueError("labels must have the same length as logits")
    if not np.isin(y, (0, 1, UNLABELLED)).all():
        raise ValueError(f"labels must be 0, 1 or {UNLABELLED} (no label)")

    labelled = y != UNLABELLED
    tb = y == 1
    n, n_tb = int(z.size), int(tb.sum())
    needed = min_positives_certified(reference.alpha, reference.delta)
    tau_src = reference.threshold
    target_pct = round(100 * (1 - reference.alpha))

    shift = detect_score_shift(reference.logits, z, SIGNIFICANCE) if n >= MIN_IMAGES else None
    missed = int(np.sum(z[tb] < tau_src)) if n_tb else None

    # Even below the re-certification floor, a few labels can already prove a
    # failure: e.g. 5 misses out of 15 TB cases is very unlikely if the true
    # miss rate is at most alpha (one-sided binomial test).
    labels_show_failure = bool(
        missed and binom.sf(missed - 1, n_tb, reference.alpha) < SIGNIFICANCE
    )

    tau_new = None
    if n_tb >= needed:
        tau_new = certified_threshold(z[tb], reference.alpha, reference.delta)

    reasons: list[str] = []
    if shift is not None:
        reasons.append(
            f"Score shift test (KS): statistic {shift.statistic:.2f}, p = {shift.p_value:.3g} "
            f"({'shift detected' if shift.shifted else 'no shift detected'} at the "
            f"{SIGNIFICANCE:.0%} level)."
        )
    reasons.append(
        f"The source threshold rules out {np.mean(z < tau_src):.0%} of this batch, "
        f"versus {reference.ruled_out_fraction:.0%} of the source images."
    )
    if missed is not None:
        reasons.append(f"It would rule out {missed} of the {n_tb} labelled TB cases here.")

    if tau_new is not None:
        verdict = "recertified"
        headline = (f"New threshold certified from {n_tb} labelled TB cases at this site "
                    f"({target_pct}% sensitivity, {1 - reference.delta:.0%} confidence).")
        reasons.append(
            f"Labels beat the detector: with at least {needed} labelled TB cases the promise "
            "can be checked here directly, whatever the shift test says."
        )
        reasons.append(
            f"The new threshold rules out {np.mean(z < tau_new):.0%} of this batch. With few "
            "labelled TB cases it has to sit near the lowest of them; more labels let it clear "
            "more healthy people."
        )
    elif n < MIN_IMAGES:
        verdict = "defer"
        headline = f"Too few images to check this site ({n} < {MIN_IMAGES})."
        reasons.append(
            f"Send at least {MIN_IMAGES} images. In this study, 50 unlabelled images detected the "
            "real site shift for every model."
        )
    elif labels_show_failure:
        verdict = "recalibrate"
        headline = (f"The labelled TB cases already show the {target_pct}% sensitivity promise "
                    "failing at this site.")
        reasons.append(
            f"{missed} misses out of {n_tb} would be very unlikely (p < {SIGNIFICANCE}) if the "
            f"source threshold still caught {target_pct}% of TB here. Label at least {needed} "
            f"TB-positive cases ({needed - n_tb} more) to certify a new threshold."
        )
    elif shift.shifted:
        verdict = "recalibrate"
        headline = ("This site's scores differ from the source. The "
                    f"{target_pct}% sensitivity promise is unverified here.")
        reasons.append(
            "A detected shift does not prove the promise broke: in this study the test fired "
            "for all five training runs at the new hospital, but the promise failed for one. "
            "Only labels tell these apart."
        )
        reasons.append(
            f"Label at least {needed} TB-positive cases from this site to re-certify "
            f"(you have {n_tb}; {needed - n_tb} more needed). Moving the threshold without "
            "labels broke the promise in 27-80% of trials in this study."
        )
    else:
        verdict = "plausible"
        headline = (f"No shift detected in {n} images. The source {target_pct}% sensitivity "
                    "promise is plausible here, not proven.")
        reasons.append(
            "The test can miss shifts, especially small batches or subtle changes (it missed a "
            "strong contrast change in this study). Labelled TB cases are the only direct check."
        )

    tau_in_force = tau_new if tau_new is not None else tau_src
    probs = apply_temperature(z, reference.temperature)
    return SiteCheck(
        verdict=verdict,
        headline=headline,
        reasons=reasons,
        n_images=n,
        n_labelled=int(labelled.sum()),
        n_labelled_tb=n_tb,
        tb_labels_needed=max(0, needed - n_tb),
        ks_statistic=None if shift is None else shift.statistic,
        ks_p_value=None if shift is None else shift.p_value,
        shift_detected=None if shift is None else shift.shifted,
        source_threshold_logit=tau_src,
        ruled_out_fraction_source=reference.ruled_out_fraction,
        ruled_out_fraction_site=float(np.mean(z < tau_src)),
        labelled_tb_missed=missed,
        recertified_threshold_logit=tau_new,
        ruled_out_fraction_recertified=None if tau_new is None else float(np.mean(z < tau_new)),
        threshold_in_force_logit=tau_in_force,
        probabilities=[float(p) for p in probs],
        ruled_out=[bool(v) for v in z < tau_in_force],
        histogram=score_histogram(reference, z, tau_src, tau_new),
    )


def score_histogram(reference: Reference, site_logits, source_threshold: float,
                    new_threshold: float | None = None, bins: int = HISTOGRAM_BINS) -> dict:
    """Share of images per probability bin, for source and site, plus thresholds on that scale."""
    t = reference.temperature
    edges = np.linspace(0.0, 1.0, bins + 1)

    def shares(logits) -> list[float]:
        counts, _ = np.histogram(apply_temperature(np.asarray(logits, dtype=float), t), bins=edges)
        return (counts / max(counts.sum(), 1)).tolist()

    return {
        "edges": edges.tolist(),
        "source": shares(reference.logits),
        "site": shares(site_logits),
        "source_threshold": _probability_or_none(source_threshold, t),
        "new_threshold": None if new_threshold is None else _probability_or_none(new_threshold, t),
    }


def _finite_or_none(v: float | None) -> float | None:
    return None if v is None or not math.isfinite(v) else float(v)


def _probability_or_none(logit_threshold: float, temperature: float) -> float | None:
    if not math.isfinite(logit_threshold):
        return None
    return float(apply_temperature(np.array([logit_threshold]), temperature)[0])
