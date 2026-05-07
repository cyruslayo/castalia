# Production Sandbox Dockerfile
# Hardened Python execution environment for untrusted code
#
# Build: docker build -f sandbox.Dockerfile -t castalia-sandbox:latest .
# Run:   docker run --rm --network none --memory 128m --read-only \
#          -v /path/to/code.py:/app/code.py:ro castalia-sandbox:latest

FROM python:3.11-alpine

# ── Security: Remove unnecessary files ──────────────────────────
# python:3.11-alpine is already minimal — just clean up
RUN rm -rf /var/cache/apk/* \
    /usr/share/doc \
    /usr/share/man \
    /tmp/*

# ── Create unprivileged user ──────────────────────────────────────
RUN addgroup -S sandboxuser -g 1000 \
    && adduser -S sandboxuser -G sandboxuser -u 1000

# ── Remove pip entirely ─────────────────────────────────────────
# This prevents: subprocess.run(["pip", "install", "requests"])
RUN pip uninstall -y pip setuptools wheel \
    && rm -rf /usr/local/lib/python3.11/site-packages/pip* \
    && rm -rf /usr/local/lib/python3.11/site-packages/setuptools* \
    && rm -rf /usr/local/lib/python3.11/site-packages/wheel* \
    && rm -rf /usr/local/bin/pip* \
    && rm -rf /usr/local/lib/python3.11/ensurepip \
    && rm -rf /usr/local/lib/python3.11/distutils

# ── Runtime setup ────────────────────────────────────────────────
WORKDIR /app
USER sandboxuser

# Health check: verify Python works
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Unbuffered output, no bytecode writing, isolated home
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    HOME=/tmp

# Entrypoint: execute code from stdin (no file mount needed)
# Usage: echo "print(2+2)" | docker run -i --rm ... castalia-sandbox:latest
# Or:    docker run --rm ... -v /path/to/code.py:/app/code.py:ro castalia-sandbox:latest
ENTRYPOINT ["python", "-u", "-"]
