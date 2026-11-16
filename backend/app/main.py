"""TB-Recertify site-check demo: FastAPI app.

Run (from the project folder):
    uvicorn backend.app.main:app --port 8000

Then open http://localhost:8000 (built frontend) or http://localhost:8000/docs (API).

Research prototype. Not for clinical use. Uploaded images are scored in
memory and never stored.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.config import Settings, load_settings
from backend.app.routers import info, site_check
from tbshift.site_check import Reference, reference_from_score_cache

_UNSET = object()


def _load_reference(settings: Settings, notes: list[str]) -> Reference | None:
    if not settings.source_scores.exists():
        notes.append(f"score cache not found: {settings.source_scores}")
        return None
    return reference_from_score_cache(settings.source_scores, settings.model_name,
                                      settings.source_site)


def _load_scorer(settings: Settings, notes: list[str]):
    if not settings.checkpoint.exists():
        notes.append(f"checkpoint not found: {settings.checkpoint}")
        return None
    from backend.app.services.scorer import DenseNetScorer
    return DenseNetScorer(settings.checkpoint, settings.corner_mask, settings.device)


def create_app(settings: Settings | None = None, reference=_UNSET, scorer=_UNSET) -> FastAPI:
    """Build the app. Tests pass ``reference`` and ``scorer`` to skip loading real files."""
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Load once at startup, shared by every request.
        notes: list[str] = []
        app.state.settings = settings
        app.state.load_notes = notes
        app.state.reference = _load_reference(settings, notes) if reference is _UNSET else reference
        app.state.scorer = _load_scorer(settings, notes) if scorer is _UNSET else scorer
        yield

    app = FastAPI(title="TB-Recertify site check",
                  description="Research prototype. Not for clinical use.", lifespan=lifespan)
    app.include_router(info.router)
    app.include_router(site_check.router)
    if settings.frontend_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
    return app


app = create_app()
