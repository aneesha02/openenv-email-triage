FROM python:3.11-slim

LABEL org.opencontainers.image.title="Email Triage OpenEnv"
LABEL org.opencontainers.image.description="OpenEnv-compliant email triage environment"
LABEL org.opencontainers.image.version="1.0.0"

# HF Spaces runs as user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[baseline]" 2>/dev/null || true

# Copy source
COPY src/ ./src/
COPY baseline/ ./baseline/
COPY server.py ./
COPY openenv.yaml ./
COPY ui/ ./ui/

# Install the package
RUN pip install --no-cache-dir -e ".[baseline]"

# Run tests to validate install
COPY tests/ ./tests/
RUN pip install --no-cache-dir pytest pytest-asyncio httpx && \
    python -m pytest tests/ -v --tb=short 2>&1 | tail -30

USER appuser

EXPOSE 7860

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
