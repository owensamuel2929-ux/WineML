# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Shared base: installs dependencies from the lockfile into a virtualenv.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# uv is copied from its official image rather than pip-installed, which keeps
# the build reproducible and avoids a bootstrap dependency on PyPI.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# ---------------------------------------------------------------------------
# Builder: resolve and install dependencies. Kept separate so the (slow,
# rarely changing) dependency layer is cached independently of source changes.
# ---------------------------------------------------------------------------
FROM base AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python .

# ---------------------------------------------------------------------------
# Runtime: no compilers, no build tooling, non-root user.
# ---------------------------------------------------------------------------
FROM base AS runtime

RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --create-home appuser

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser raw_data/ ./raw_data/

# Artifacts are mounted as a volume in compose, but the directories must exist
# and be writable so the training profile can populate them.
RUN mkdir -p /app/models /app/reports && chown -R appuser:appuser /app/models /app/reports

ENV PYTHONPATH=/app/src

USER appuser

EXPOSE 8000

# Compose overrides this for the dashboard service.
CMD ["uvicorn", "wine_quality.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# Training stage: same image, different entrypoint. Selected via the compose
# `train` profile so `docker compose up` stays fast for demos.
# ---------------------------------------------------------------------------
FROM runtime AS training

USER appuser

CMD ["python", "-m", "wine_quality.models.train"]