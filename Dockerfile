# Credit Risk Assessment API
# Same Python version as .python-version, so the container matches local runs.
FROM python:3.13.5-slim

# No .pyc files, logs straight to the console, no pip cache in the image.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first: this layer is reused as long as requirements.txt doesn't change.
# On Linux the `xgboost` package also pulls in NVIDIA's GPU library (~800 MB).
# The API only predicts on CPU, so the image installs `xgboost-cpu` instead:
# same version, same `import xgboost`, same model format. requirements.txt keeps
# plain `xgboost` because xgboost-cpu has no macOS build.
COPY requirements.txt .
RUN sed 's/^xgboost==/xgboost-cpu==/' requirements.txt > /tmp/requirements-docker.txt \
    && grep -q '^xgboost-cpu==' /tmp/requirements-docker.txt \
    && pip install -r /tmp/requirements-docker.txt \
    && rm /tmp/requirements-docker.txt

# Only what the API needs at runtime (see .dockerignore).
COPY main.py model_io.py credit_risk_model.pkl best_threshold.pkl ./
COPY static ./static

# Run as a normal user, not root. /app/data holds the prediction log.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app/data
USER appuser

# Render (and most hosts) pass the port in $PORT; 8000 when run locally.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health', timeout=4)"

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
