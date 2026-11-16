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

Experiments are complete for two sites and five training seeds; the demo is built; the paper is next. See [docs/PROGRESS.md](docs/PROGRESS.md) for results and [docs/PLAN.md](docs/PLAN.md) for the roadmap.

## Setup (Windows, Anaconda Python 3.13)

This machine has Windows **Smart App Control** switched on. It intermittently blocks
freshly downloaded compiled Python files (the newest SciPy and scikit-learn wheels
from pip), while Anaconda's own builds load reliably. So the environment uses
Anaconda's scientific stack and installs only PyTorch with pip.

```powershell
# 1. Virtual environment that can see Anaconda's packages
C:\Users\<you>\anaconda3\python.exe -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1

# 2. PyTorch with CUDA (pick the build for your GPU at pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 3. This package without re-installing NumPy/SciPy/scikit-learn/pandas from pip
pip install -e . --no-deps
pip install ruff

# 4. One OpenMP runtime only: Anaconda's NumPy and PyTorch each ship libiomp5md.dll,
#    and loading both crashes CPU code. Set PyTorch's copy aside; PyTorch then
#    finds Anaconda's copy in anaconda3\Library\bin.
Rename-Item .venv\Lib\site-packages\torch\lib\libiomp5md.dll libiomp5md.dll.bak

# 5. Run the tests
pytest
```

On a machine without Smart App Control, the plain route also works:
`pip install torch torchvision --index-url ...` followed by `pip install -e ".[dev]"`.

## Demo: site check

A small web app that answers one question for a new hospital: *can we trust the
rule-out threshold certified elsewhere?* It compares the site's model scores with
the source (KS test), checks any labelled TB cases, and re-certifies the threshold
when at least 22 TB cases are labelled. It also shows the five-seed results and
the limitations. Research prototype; not for clinical use.

```powershell
# once: web packages (pure Python) and the frontend build
pip install fastapi uvicorn python-multipart
cd frontend; npm install; npm run build; cd ..

# run (needs the P2 checkpoint and score cache in artifacts/)
uvicorn backend.app.main:app --port 8000
```

Open http://localhost:8000 (app) or http://localhost:8000/docs (API).
**Simulated hospital** needs no images: it resamples the source model's scores,
shifts them, and shows the true answer next to the check's verdict.
**Upload images** scores your own PNG/JPEG X-rays in memory (nothing is stored);
an optional `filename,label` CSV adds labels. No NLM images are shipped or shown.

Frontend development with hot reload: run the API as above, then `cd frontend; npm run dev`
and open http://localhost:5173 (it forwards `/api` to port 8000).

Docker: `docker compose up --build` builds one image (CPU PyTorch, FastAPI serving
the built React app) and mounts `artifacts/` and `results/` read-only, since the
model derives from NLM data and is not baked into the image.

Smart App Control note: Rollup's native binary is blocked on this machine, so
`frontend/package.json` overrides it with `@rollup/wasm-node` (same bundler,
WebAssembly build).

## Layout

| Path | Contents |
|---|---|
| `src/tbshift/` | Research code: metrics, calibration, conformal thresholds, shift detection, recalibration, degradations |
| `tests/` | Unit tests (research code and demo API) |
| `configs/` | YAML experiment configs |
| `data/manifests/` | CSV manifests (paths, labels, site). Images are never committed |
| `artifacts/` | Cached model scores and checkpoints (git-ignored) |
| `results/` | Experiment CSVs and figures (git-ignored) |
| `backend/app/` | FastAPI "site check" service: routers (thin), schemas (Pydantic), services (model, scenarios, results) |
| `frontend/src/` | React + Vite demo: Site check, Study results, Limitations pages |
| `docs/` | Plan, progress, research notebook, bug log, reading log, dataset audit |

## Data

Datasets are downloaded separately from their original sources and are not redistributed here. See [docs/dataset_audit.md](docs/dataset_audit.md) for sources, licences and the same-site check, and [data/manifests/README.md](data/manifests/README.md) for building the manifests.

The Montgomery County and Shenzhen chest X-ray sets are provided by the National Library of Medicine, National Institutes of Health, Bethesda, MD, USA (Shenzhen in collaboration with Shenzhen No. 3 People's Hospital, Guangdong Medical College, Shenzhen, China). Per NLM's terms they are not shared outside the research group; request them from NLM. Work using them should cite:

- Jaeger S, et al. Automatic tuberculosis screening using chest radiographs. *IEEE Trans Med Imaging*. 2014;33(2):233–45. doi:10.1109/TMI.2013.2284099
- Candemir S, et al. Lung segmentation in chest radiographs using anatomical atlases with nonrigid registration. *IEEE Trans Med Imaging*. 2014;33(2):577–90. doi:10.1109/TMI.2013.2290491
