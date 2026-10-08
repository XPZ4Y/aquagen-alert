# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Stage 1: build the venv with uv (cached dependency layer)
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# 1a. Resolve+install only the lockfile-referenced deps first, so code edits
#     don't invalidate the (slow) dependency layer.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 1b. Now copy the source and install the project itself.
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev


# ---------------------------------------------------------------------------
# Stage 2: slim runtime (core: risk engine + API + alerts, no torch)
# ---------------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    AQUA_HOST=0.0.0.0 \
    AQUA_PORT=8000

# Non-root user
RUN groupadd --system --gid 1001 aqua \
 && useradd  --system --uid 1001 --gid aqua --create-home aqua

WORKDIR /app

# venv from builder
COPY --from=builder --chown=aqua:aqua /opt/venv /opt/venv

# App code + static assets the API serves + bundled offline samples
COPY --chown=aqua:aqua src/     ./src/
COPY --chown=aqua:aqua screens/ ./screens/
COPY --chown=aqua:aqua data/    ./data/
COPY --chown=aqua:aqua docs/    ./docs/

USER aqua
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; \
      sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/catalog', timeout=3).status==200 else 1)"

CMD ["uvicorn", "aqua_alert.api.server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]


# ---------------------------------------------------------------------------
# Stage 3: ml runtime — same as `runtime` but with torch/transformers
#          build with:  docker build --target ml -t aqua-alert:ml .
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS ml-builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra ml
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ml

FROM python:3.11-slim-bookworm AS ml

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/home/aqua/.cache/huggingface \
    TRANSFORMERS_OFFLINE=0

RUN groupadd --system --gid 1001 aqua \
 && useradd  --system --uid 1001 --gid aqua --create-home aqua

WORKDIR /app
COPY --from=ml-builder --chown=aqua:aqua /opt/venv /opt/venv
COPY --chown=aqua:aqua src/     ./src/
COPY --chown=aqua:aqua screens/ ./screens/
COPY --chown=aqua:aqua data/    ./data/
COPY --chown=aqua:aqua docs/    ./docs/

USER aqua
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; \
      sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/catalog', timeout=3).status==200 else 1)"

CMD ["uvicorn", "aqua_alert.api.server:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--proxy-headers", "--forwarded-allow-ips", "*"]
