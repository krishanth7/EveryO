# Multi-stage production container for EveryO CPU serving
# Security hardened: non-root execution, minimal attack surface, pinned base image

FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

# Install build prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY everyo ./everyo

# Build standard wheel
RUN pip install --no-cache-dir --upgrade pip build wheel && \
    python -m build --wheel --no-isolation

# ---------------------------------------------------------
# Production Runtime Image
# ---------------------------------------------------------
FROM python:3.12-slim-bookworm AS runtime

LABEL maintainer="EveryO Project"
LABEL org.opencontainers.image.title="EveryO Model Serving Runtime"
LABEL org.opencontainers.image.description="Hardened non-root container for EveryO neural inference"
LABEL org.opencontainers.image.version="0.5.0"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root dedicated application user and group
ENV APP_USER=everyo \
    APP_UID=10001 \
    APP_GID=10001 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

RUN groupadd --gid ${APP_GID} ${APP_USER} && \
    useradd --uid ${APP_UID} --gid ${APP_GID} --create-home --shell /bin/bash ${APP_USER}

WORKDIR /app

# Copy built distribution wheel from builder
COPY --from=builder /build/dist/*.whl /tmp/

# Install EveryO wheel and remove temporary wheel
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm -rf /tmp/*.whl && \
    mkdir -p /app/models /app/logs && \
    chown -R ${APP_USER}:${APP_USER} /app

# Switch to non-privileged user
USER ${APP_USER}:${APP_USER}

# Expose loopback/container serving port
EXPOSE 8000

# Container liveness probe
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; resp = urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2); exit(0 if resp.status == 200 else 1)"

ENTRYPOINT ["python", "-m", "everyo.cli.main"]
CMD ["info"]
