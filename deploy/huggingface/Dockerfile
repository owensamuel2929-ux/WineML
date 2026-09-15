# Hugging Face Spaces deployment image.
#
# This file becomes the ROOT Dockerfile of the Space repository. Hugging Face
# only auto-detects a Dockerfile named `Dockerfile` at the repository root, and
# there is no config key to relocate it — which is why the Space is a separate
# repository assembled by `scripts/deploy_hf_space.sh` rather than a branch of
# the main one. The main repo keeps its two-service compose setup intact.
#
# Differences from the compose image, all forced by the Spaces platform:
#   * Runs as UID 1000 (`user`), per the Spaces permissions contract.
#   * Exposes exactly one port (7860); nginx multiplexes API and dashboard.
#   * Model artifacts are baked in, because a Space has no compose volume.
#     This adds ~16 MB and makes the image self-contained.

FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Build tooling is needed for wheels without a prebuilt release for this
# platform, then discarded with the stage.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Spaces run as UID 1000. Creating the user before any COPY keeps the later
# --chown flags cheap and avoids a recursive chown that would duplicate layers.
RUN useradd --create-home --uid 1000 --shell /bin/bash user

# Dependencies first: this layer is cached independently of source changes.
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN uv venv /opt/venv \
    && uv pip install --python /opt/venv/bin/python .

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    # Settings resolve their project root from this, so paths must be absolute.
    WINE_PROJECT_ROOT=/app \
    WINE_API_BASE_URL=http://127.0.0.1:8000 \
    WINE_API_PORT=8000 \
    PORT=7860

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 --shell /bin/bash user

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# --chown=user on every COPY, as the Spaces docs require. A later recursive
# chown would duplicate these files into an extra layer.
COPY --chown=user . /app

# nginx config and pid path. The default config file is replaced outright.
COPY --chown=user deploy/huggingface/nginx.conf /etc/nginx/nginx.conf

# nginx writes its pid here; give the unprivileged user ownership so the master
# process can start without elevated permissions.
RUN mkdir -p /var/lib/nginx /var/log/nginx /tmp/nginx \
    && chown -R user:user /var/lib/nginx /var/log/nginx /tmp/nginx /app

USER user

# The single port Spaces publishes. Must match `app_port` in README.md.
EXPOSE 7860

# Fail the Space health check if either service is unhealthy. `/api/health`
# exercises nginx, the proxy hop, and the API's own readiness in one request.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:7860/api/health || exit 1

# The supervisor starts uvicorn, streamlit, and nginx, and tears the container
# down if any of them dies — so a crash surfaces as a failed Space rather than a
# half-working demo.
CMD ["python", "deploy/huggingface/entrypoint.py"]