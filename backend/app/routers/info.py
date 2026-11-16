"""Health and study-results endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from backend.app.services import results

router = APIRouter(prefix="/api", tags=["info"])


@router.get("/health")
def health(request: Request):
    state = request.app.state
    return {
        "status": "ok",
        "model_loaded": state.scorer is not None,
        "reference": None if state.reference is None else state.reference.summary(),
        "notes": state.load_notes,
    }


@router.get("/results")
def list_results(request: Request):
    return {"tables": results.available(request.app.state.settings.results_dir)}


@router.get("/results/headline")
def headline(request: Request):
    rows = results.headline(request.app.state.settings.results_dir)
    if rows is None:
        raise HTTPException(404, "multi-seed results not found (run tbshift.experiments.multi_seed)")
    return rows


@router.get("/results/{name}")
def result_table(name: str, request: Request):
    rows = results.table(request.app.state.settings.results_dir, name)
    if rows is None:
        raise HTTPException(404, f"no result table {name!r}")
    return rows
