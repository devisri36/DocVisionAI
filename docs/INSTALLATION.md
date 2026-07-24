# Installation Guide

This guide details how to install and configure DocVision AI locally or via containerized environments.

---

## 1. Local Development (Virtualenv)

### Step 1: Clone and Enter the Directory
```bash
git clone https://github.com/yourportfolio/DocVision-AI.git
cd DocVision-AI
```

### Step 2: Set Up Python Virtual Environment
DocVision requires **Python 3.10+**.
```bash
# Create venv
python -m venv venv

# Activate on Windows (CMD/Powershell)
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 3: Install Packages
```bash
pip install -r requirements.txt
```

### Step 4: Configure Variables
Copy `.env.example` to `.env` and fill in active variables:
```bash
cp .env.example .env
```

---

## 2. Containerized Deployment (Docker)

To run the application via Docker Compose:

```bash
# Build and run containers
docker-compose up --build

# Run in background daemon mode
docker-compose up -d --build
```

### Port Mappings
- **FastAPI Backend Swagger**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501)
