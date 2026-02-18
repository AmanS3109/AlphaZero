# ─────────────────────────────────────────────
# AlphaZero Deployment
# ─────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Install CPU-only PyTorch first (smaller image)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY api/ api/
COPY configs/ configs/
COPY checkpoints/ checkpoints/
COPY play.py .
COPY train.py .

# Expose API port
EXPOSE 8000

# Environment
ENV CHECKPOINT_DIR=/app/checkpoints
ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
