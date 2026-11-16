"""Request and response shapes (Pydantic). FastAPI validates against these and
publishes them in the OpenAPI docs at /docs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

MAX_SCORES = 5000


class ScoresRequest(BaseModel):
    logits: list[float] = Field(min_length=1, max_length=MAX_SCORES,
                                description="Model logits for one site's batch")
    labels: list[Literal[0, 1] | None] | None = Field(
        default=None, description="Optional 0/1 label per logit; null = unlabelled")


class Histogram(BaseModel):
    edges: list[float]
    source: list[float]
    site: list[float]
    source_threshold: float | None
    new_threshold: float | None


class ImageResult(BaseModel):
    name: str
    probability: float
    ruled_out: bool
    label: Literal[0, 1] | None = None


class SiteCheckResponse(BaseModel):
    verdict: Literal["defer", "plausible", "recalibrate", "recertified"]
    headline: str
    reasons: list[str]
    n_images: int
    n_labelled: int
    n_labelled_tb: int
    tb_labels_needed: int
    ks_statistic: float | None
    ks_p_value: float | None
    shift_detected: bool | None
    source_threshold_logit: float | None
    ruled_out_fraction_source: float
    ruled_out_fraction_site: float
    labelled_tb_missed: int | None
    recertified_threshold_logit: float | None
    ruled_out_fraction_recertified: float | None
    threshold_in_force_logit: float | None
    histogram: Histogram
    images: list[ImageResult]
    ground_truth: dict | None = Field(
        default=None, description="Only for simulated scenarios, where the true labels are known")


class ScenarioInfo(BaseModel):
    name: str
    title: str
    description: str
    n_images: int
    n_labelled: int
