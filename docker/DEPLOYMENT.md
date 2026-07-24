# DocVision AI - Docker & Deployment Guide

This guide details the steps required to compile, orchestrate, and deploy the DocVision AI application in production-grade containerised environments.

---

## 1. Prerequisites

### CPU Execution (General)
- **Docker Engine** v20.10.0 or higher.
- **Docker Compose** v2.0.0 or higher.

### GPU Execution (CUDA Acceleration)
To enable GPU hardware acceleration for PyTorch VLM (Florence-2) operations inside Docker:
1. Ensure your host machine has an **NVIDIA GPU** with compatible drivers installed.
2. Install the **NVIDIA Container Toolkit**:
   - [NVIDIA Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
3. Restart the Docker daemon to register the `nvidia` runtime:
   ```bash
   sudo systemctl restart docker
   ```

---

## 2. Environment Configuration

Copy the sample environment file to create your active `.env` parameters file:
```bash
cp .env.example .env
```

Ensure the parameters match your network requirements:
```ini
HOST=0.0.0.0
PORT=8000
STREAMLIT_PORT=8501
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/docvision.log
HF_TOKEN=your_huggingface_write_token_here
```

---

## 3. Build & Orchestration

Use Docker Compose to build and start both the FastAPI backend and Streamlit dashboard:

```bash
# Build and launch services in the foreground
docker-compose up --build

# Run in detached (background) daemon mode
docker-compose up -d --build
```

### Automatic Health Verification
Docker Compose automatically tracks container health checks:
- **`backend`**: Pings `/health` and monitors PyTorch/SQLite status.
- **`frontend`**: Pings Streamlit's `_stcore/health` endpoint and starts **only after** the backend container reports a `healthy` state.

---

## 4. Service Access

Once container initialization succeeds, access the services:
- **FastAPI OpenAPI (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **FastAPI Redoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501)

---

## 5. Troubleshooting & Commands

### Inspecting Container Logs
```bash
# Inspect all logs
docker-compose logs -f

# Inspect backend service only
docker-compose logs -f backend
```

### Stopping Services
```bash
# Stop and retain volumes
docker-compose down

# Stop and clear persistent SQLite and VLM model cache volumes
docker-compose down -v
```

### SQLite Database Verification
The SQLite database file `docvision.db` is saved inside the named volume `docvision_outputs`. It persists transactions across container restarts. To inspect raw logs from the host:
```bash
docker exec -it docvision_backend sqlite3 /app/outputs/docvision.db "SELECT * FROM history;"
```
