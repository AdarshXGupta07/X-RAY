# TB-Recertify site-check demo: one container, FastAPI serving the built React app.
# The trained model and score cache are NOT baked in (they derive from NLM data
# that may not be redistributed); mount artifacts/ and results/ at run time.

FROM node:22-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
WORKDIR /app
# CPU-only PyTorch: the demo scores a few dozen images per request.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src/ src/
RUN pip install --no-cache-dir ".[api]"
COPY backend/ backend/
COPY --from=frontend /app/frontend/dist frontend/dist
ENV TBSHIFT_DEVICE=cpu
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
