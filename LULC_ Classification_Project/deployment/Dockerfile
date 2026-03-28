# ============================================================
# TransUNet-RS — Dockerfile (Multi-stage)
# ============================================================
# Build: docker build -t transunet-rs .
# Run:   docker run -p 8000:8000 transunet-rs
# GPU:   docker run --gpus all -p 8000:8000 transunet-rs
# ============================================================

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints data/raw data/processed results

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default command: start the API server
CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
