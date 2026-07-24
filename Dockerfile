# ==========================================
# STAGE 1: Base Build
# ==========================================
FROM python:3.11-slim AS base

# Set env variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install native system dependencies for OpenCV, Pillow, and healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python packages using pip cache
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app/

# Setup entrypoint permissions
RUN chmod +x /app/docker/entrypoint.sh

ENTRYPOINT ["/app/docker/entrypoint.sh"]

# ==========================================
# STAGE 2: API Backend Target
# ==========================================
FROM base AS backend
EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ==========================================
# STAGE 3: Streamlit UI Target
# ==========================================
FROM base AS frontend
EXPOSE 8501
HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "frontend/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
