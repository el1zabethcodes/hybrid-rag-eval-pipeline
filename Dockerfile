# Multi-stage Dockerfile for FastAPI RAG API and Streamlit UI

# ==============================================================================
# 1. Builder Stage
# ==============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install package dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# ==============================================================================
# 2. Runtime Stage
# ==============================================================================
FROM python:3.11-slim AS runtime

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy source code and config files
COPY src/ /app/src/
COPY ui/ /app/ui/
COPY config.yaml /app/config.yaml
COPY .env.example /app/.env

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501
