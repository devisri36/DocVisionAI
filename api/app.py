import io
import base64
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, Query, HTTPException, Depends, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from PIL import Image
import torch

from configs.config_loader import ConfigManager
from utils.logger import setup_logger

# Import Database & Security Services
from backend.database import (
    init_db,
    register_user,
    get_user_by_username,
    fetch_history_entries,
    get_file_record
)
from backend.services.auth_service import (
    hash_password,
    verify_password,
    create_access_token,
    get_current_user,
    security
)
from backend.services.document_service import DocumentService

# Preprocessing endpoints (sandbox)
from backend.preprocessor import DocumentPreprocessor

# Initialize Logger and Config
logger = setup_logger("api")
config = ConfigManager().config

app = FastAPI(
    title=config.project_name,
    description="DocVision AI Backend: Intelligent Identity Verification & Document Understanding System API Console",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Instantiate Document Service Layer
doc_service = DocumentService()

@app.on_event("startup")
async def startup_event():
    """Initializes sqlite database structures on API start."""
    init_db()

# --- PYDANTIC SCHEMAS ---

class UserAuthSchema(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, example="admin")
    password: str = Field(..., min_length=6, example="docvisionpass")

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class FileUploadResponse(BaseModel):
    file_id: str
    filename: str
    filepath: str

class FileIdRequest(BaseModel):
    file_id: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000")

class AskQuestionRequest(BaseModel):
    file_id: str = Field(..., example="550e8400-e29b-41d4-a716-446655440000")
    question: str = Field(..., example="What is the document number?")

# --- ENDPOINTS ---

@app.get("/health", tags=["System"])
async def get_health_status():
    """Outputs system health indices, storage capacity, and hardware profile."""
    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    device = "cuda" if gpu_available else "cpu"
    
    # Check DB accessibility
    try:
        from backend.database import get_db_path
        db_path = get_db_path()
        db_exists = db_path.exists()
    except Exception:
        db_exists = False
        
    return {
        "status": "Healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "hardware": {
            "device": device,
            "gpu_count": torch.cuda.device_count() if gpu_available else 0
        },
        "database": {
            "online": True,
            "file_exists": db_exists
        }
    }

@app.post("/auth/register", status_code=status.HTTP_201_CREATED, tags=["Authentication"])
async def register(auth_data: UserAuthSchema):
    """Registers a new backend user profile."""
    pw_hash = hash_password(auth_data.password)
    success = register_user(auth_data.username, pw_hash)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered."
        )
    return {"message": f"User '{auth_data.username}' registered successfully."}

@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login(auth_data: UserAuthSchema):
    """Authenticates credentials and returns a JWT access token."""
    user = get_user_by_username(auth_data.username)
    if not user or not verify_password(auth_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/upload", response_model=FileUploadResponse, tags=["Document Verification"])
async def upload_document(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Accepts document uploads, saves them securely, and registers the file_id."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload an image."
        )
        
    try:
        content = await file.read()
        res = doc_service.save_upload(file.filename, content, current_user["username"])
        return res
    except Exception as e:
        logger.error(f"Failed to handle document upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload save error: {str(e)}"
        )


@app.post("/extract", tags=["Document Verification"])
async def extract_document_info(
    req: FileIdRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Runs OCR parsing, quality checks, classifications, and field extractions on a file_id."""
    try:
        res = doc_service.run_full_analysis(req.file_id, current_user["username"])
        return res
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to extract document information: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction analysis error: {str(e)}"
        )


@app.post("/verify", tags=["Document Verification"])
async def verify_document_quality(
    req: FileIdRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Performs image quality screens (blur, crop, resolution, contrast) on a file_id."""
    try:
        res = doc_service.verify_document_quality(req.file_id)
        return res
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to verify quality metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality audit error: {str(e)}"
        )


@app.post("/detect-fraud", tags=["Document Verification"])
async def detect_document_fraud(
    req: FileIdRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Performs JPEG Error Level Analysis (ELA) and duplicate matching checks on a file_id."""
    try:
        res = doc_service.detect_fraud_forensics(req.file_id)
        return res
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to perform fraud audit: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fraud forensics error: {str(e)}"
        )


@app.post("/ask", tags=["Document Verification"])
async def ask_document_vlm(
    req: AskQuestionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Queries the Vision-Language Model regarding document visuals."""
    try:
        res = doc_service.ask_vlm_question(req.file_id, req.question)
        return res
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        logger.error(f"Failed VLM QA request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Visual QA error: {str(e)}"
        )


@app.get("/metrics", tags=["System"])
async def get_system_metrics(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Computes high-level usage, pass rates, score distributions, and volumes from SQLite logs."""
    try:
        history = fetch_history_entries(current_user["username"])
        
        total_runs = len(history)
        if total_runs == 0:
            return {
                "total_verifications": 0,
                "verification_pass_rate": 0.0,
                "average_authenticity": 0.0,
                "average_fraud_risk": 0.0,
                "category_distribution": {}
            }
            
        auth_sum = sum(entry["authenticity_score"] for entry in history)
        fraud_sum = sum(entry["fraud_score"] for entry in history)
        
        # Calculate pass rate: percentage of runs with authenticity >= 0.70
        passes = sum(1 for entry in history if entry["authenticity_score"] >= 0.70)
        pass_rate = passes / total_runs
        
        # Category distribution counter
        categories = {}
        for entry in history:
            cat = entry["category"]
            categories[cat] = categories.get(cat, 0) + 1
            
        return {
            "total_verifications": total_runs,
            "verification_pass_rate": round(pass_rate, 2),
            "average_authenticity": round(auth_sum / total_runs, 2),
            "average_fraud_risk": round(fraud_sum / total_runs, 2),
            "category_distribution": categories
        }
    except Exception as e:
        logger.error(f"Failed to fetch system metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metrics service is temporarily unavailable."
        )


@app.get("/history", tags=["System"])
async def get_user_history(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Retrieves SQLite history logs for the authenticated user."""
    try:
        history = fetch_history_entries(current_user["username"])
        return history
    except Exception as e:
        logger.error(f"Failed to fetch user history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="History retrieval error."
        )

# --- BACKWARD COMPATIBLE SANDBOXPreprocess ENDPOINT ---

@app.post("/api/v1/preprocess", tags=["Sandbox Preprocessing"])
async def preprocess_document(file: UploadFile = File(...), augment: bool = Query(False)):
    """Sandbox endpoint for LayoutLM coordinate mapping preprocessing test."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
        
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_width, img_height = image.size
        
        # Simple simulated boxes
        bboxes = [[10.0, 20.0, 100.0, 120.0]]
        words = ["SIMULATED"]
        labels = ["header"]
        
        preprocessor = DocumentPreprocessor(augment=augment)
        processed_data = preprocessor.preprocess(image, bboxes, words, labels)
        
        processed_tensor = processed_data["image"]
        mean = torch.tensor(config.preprocessing.normalization.mean).view(3, 1, 1)
        std = torch.tensor(config.preprocessing.normalization.std).view(3, 1, 1)
        
        denorm_tensor = processed_tensor * std + mean
        denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
        
        img_array = (denorm_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
        processed_pil = Image.fromarray(img_array)
        
        buffered = io.BytesIO()
        processed_pil.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return {
            "original_size": {"width": img_width, "height": img_height},
            "preprocessed_size": {"width": preprocessor.target_width, "height": preprocessor.target_height},
            "preprocessed_image_base64": img_base64,
            "bboxes": processed_data["bboxes"],
            "normalized_bboxes": processed_data["normalized_bboxes"],
            "words": processed_data["words"],
            "labels": processed_data["labels"]
        }

    except Exception as e:
        logger.error(f"Failed to preprocess uploaded document: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {str(e)}")
