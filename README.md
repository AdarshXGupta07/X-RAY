# TB-Recertify

**When does a TB rule-out guarantee survive a new hospital?**
Silent failure and re-certification of chest X-ray AI under site shift.

> Research prototype. Public datasets only. No clinical validation. Not a medical device and not for diagnostic use.

## The question

A chest X-ray model is tuned at some hospitals to catch at least ~90% of TB cases. At a new hospital with a different X-ray machine that promise can break without anyone noticing. This project measures:

1. How often the sensitivity promise silently fails at an unseen site.
2. Whether the shift can be detected from images alone, with no labels.
3. How few labelled images from the new site restore the promise.

## Status

Work in progress. See [docs/PROGRESS.md](docs/PROGRESS.md) for what is done and [docs/PLAN.md](docs/PLAN.md) for the roadmap. No experimental results exist yet.

## Setup (Windows, Python 3.13)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 1. PyTorch first, from the CUDA index (pick the build for your GPU at pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 2. This package, in editable mode, with dev tools
pip install -e ".[dev]"

# 3. Run the tests
pytest
```

## Layout

| Path | Contents |
|---|---|
| `src/tbshift/` | Research code: metrics, calibration, conformal thresholds, shift detection, recalibration, degradations |
| `tests/` | Unit tests. Tests for unwritten modules are marked `xfail` and act as the specification |
| `configs/` | YAML experiment configs |
| `data/manifests/` | CSV manifests (paths, labels, site). Images are never committed |
| `artifacts/` | Cached model scores and checkpoints (git-ignored) |
| `results/` | Experiment CSVs and figures (git-ignored) |
| `backend/app/` | FastAPI "site check" service |
| `frontend/src/` | React + Vite demo |
| `docs/` | Plan, progress, research notebook, bug log, reading log, dataset audit |

## Data

Datasets are downloaded separately from their original sources and are not redistributed here. See [docs/dataset_audit.md](docs/dataset_audit.md) for sources, licences and the same-site check.
