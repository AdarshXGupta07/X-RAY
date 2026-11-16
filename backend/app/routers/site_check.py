"""Site-check endpoints. Thin: parse the request, call the services, shape the answer."""

from __future__ import annotations

import csv
import io
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from backend.app.schemas.site_check import ScenarioInfo, ScoresRequest, SiteCheckResponse
from backend.app.services import scenarios as sc
from backend.app.services.scorer import ImageError, decode_image
from tbshift.site_check import UNLABELLED, site_check

router = APIRouter(prefix="/api", tags=["site check"])

MAX_IMAGES = 300


def _reference(request: Request):
    ref = request.app.state.reference
    if ref is None:
        raise HTTPException(503, "No source reference loaded: the score cache for the configured "
                                 "model is missing (run P2 first).")
    return ref


def _response(result, names, labels, ground_truth=None) -> dict:
    out = result.to_dict()
    out["images"] = [
        {"name": n, "probability": p, "ruled_out": r, "label": None if y == UNLABELLED else int(y)}
        for n, p, r, y in zip(names, out.pop("probabilities"), out.pop("ruled_out"), labels,
                              strict=True)
    ]
    out["ground_truth"] = ground_truth
    return out


@router.post("/site-check/scores", response_model=SiteCheckResponse)
def check_scores(body: ScoresRequest, request: Request):
    """Site check on logits computed elsewhere (no images needed)."""
    ref = _reference(request)
    if body.labels is not None and len(body.labels) != len(body.logits):
        raise HTTPException(422, "labels must have the same length as logits")
    labels = np.array([UNLABELLED if v is None else v for v in body.labels]
                      if body.labels is not None else [UNLABELLED] * len(body.logits))
    try:
        result = site_check(ref, np.array(body.logits), labels)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return _response(result, [f"#{i + 1}" for i in range(len(body.logits))], labels)


def parse_labels_csv(text: str, names: list[str]) -> np.ndarray:
    """CSV with header ``filename,label`` (label 0 or 1). Unlisted images stay unlabelled."""
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))  # Excel adds a BOM
    if reader.fieldnames is None or not {"filename", "label"} <= set(reader.fieldnames):
        raise ValueError("labels CSV needs a header row with columns: filename,label")
    index = {n: i for i, n in enumerate(names)}
    labels = np.full(len(names), UNLABELLED)
    for line, row in enumerate(reader, start=2):
        name, value = (row["filename"] or "").strip(), (row["label"] or "").strip()
        if name not in index:
            raise ValueError(f"labels CSV line {line}: {name!r} is not one of the uploaded images")
        if value not in ("0", "1"):
            raise ValueError(f"labels CSV line {line}: label must be 0 or 1, got {value!r}")
        labels[index[name]] = int(value)
    return labels


@router.post("/site-check/images", response_model=SiteCheckResponse)
async def check_images(request: Request, images: Annotated[list[UploadFile], File()],
                       labels_csv: Annotated[UploadFile | None, File()] = None):
    """Score uploaded X-rays with the model, then run the site check. Nothing is stored."""
    ref = _reference(request)
    scorer = request.app.state.scorer
    if scorer is None:
        raise HTTPException(503, "No model loaded: the checkpoint for the configured model is "
                                 "missing. Use the scores endpoint or a simulated scenario.")
    if len(images) > MAX_IMAGES:
        raise HTTPException(413, f"at most {MAX_IMAGES} images per check")
    names = [f.filename or f"image {i + 1}" for i, f in enumerate(images)]
    if len(set(names)) != len(names):
        raise HTTPException(422, "uploaded file names must be unique")
    try:
        decoded = [decode_image(await f.read(), n) for f, n in zip(images, names, strict=True)]
        labels = (parse_labels_csv((await labels_csv.read()).decode("utf-8-sig"), names)
                  if labels_csv is not None else np.full(len(names), UNLABELLED))
        logits = scorer.score(decoded)
        result = site_check(ref, logits, labels)
    except (ImageError, ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    return _response(result, names, labels)


@router.get("/scenarios", response_model=list[ScenarioInfo])
def list_scenarios():
    return [ScenarioInfo(name=s.name, title=s.title, description=s.description,
                         n_images=s.n_tb + s.n_normal, n_labelled=s.labelled_tb + s.labelled_normal)
            for s in sc.SCENARIOS.values()]


@router.post("/site-check/scenario/{name}", response_model=SiteCheckResponse)
def check_scenario(name: str, request: Request, seed: int = 0):
    """Site check on a simulated hospital batch; also returns what really happens there."""
    ref = _reference(request)
    scenario = sc.SCENARIOS.get(name)
    if scenario is None:
        raise HTTPException(404, f"unknown scenario {name!r}")
    logits, shown, truth = sc.simulate(ref, scenario, seed)
    result = site_check(ref, logits, shown)
    gt = sc.ground_truth(ref, logits, truth, result.threshold_in_force_logit)
    return _response(result, [f"#{i + 1}" for i in range(logits.size)], shown, gt)
