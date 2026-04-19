"""
api/main.py
===========
FastAPI REST API for Customer Churn Prediction.

Endpoints:
  GET  /            — Health check + API info
  GET  /health      — Structured health status
  POST /predict     — Single customer churn prediction
  POST /predict/batch — Batch predictions (CSV-like JSON)
  GET  /model/info  — Loaded model metadata

Run with:
  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

# Ensure project root is in path when running from api/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.logger import setup_logger
from src.predict import load_model, predict_batch, predict_single
from src.preprocessing import load_config

logger = setup_logger("api")

# ---------------------------------------------------------------------------
# Configuration & startup
# ---------------------------------------------------------------------------

CONFIG_PATH = "config/config.yaml"
config = load_config(CONFIG_PATH)
api_cfg = config["api"]

# Lazy-load model at startup
_model = None
_model_load_time: Optional[float] = None


def get_model():
    """Return cached model, loading it on first call."""
    global _model, _model_load_time
    if _model is None:
        t0 = time.time()
        _model = load_model(config=config)
        _model_load_time = time.time() - t0
        logger.info("Model loaded in %.2fs", _model_load_time)
    return _model


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app):
    logger.info("API starting up...")
    try:
        get_model()
        logger.info("Startup complete.")
    except FileNotFoundError:
        logger.warning("Model file not found at startup. Train with: python main_pipeline.py")
    yield
    logger.info("API shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=api_cfg["title"],
    description=api_cfg["description"],
    version=api_cfg["version"],
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CustomerFeatures(BaseModel):
    """Input schema matching the Telco Customer Churn dataset columns."""
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
                "Dependents": "No", "tenure": 12, "PhoneService": "Yes",
                "MultipleLines": "No", "InternetService": "Fiber optic",
                "OnlineSecurity": "No", "OnlineBackup": "No",
                "DeviceProtection": "No", "TechSupport": "No",
                "StreamingTV": "Yes", "StreamingMovies": "Yes",
                "Contract": "Month-to-month", "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.0, "TotalCharges": 1020.0,
            }
        }
    )
    gender: str = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Is senior citizen (0/1)")
    Partner: str = Field(..., description="Has a partner")
    Dependents: str = Field(..., description="Has dependents")
    tenure: float = Field(..., ge=0, description="Months with company")
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_label: bool
    churn_label_text: str
    risk_tier: str
    confidence: float
    model_version: str = api_cfg["version"]


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    total: int
    predicted_churn: int
    churn_rate_pct: float
    predictions: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    api_version: str
    uptime_seconds: float


# ---------------------------------------------------------------------------
# Startup event
# ---------------------------------------------------------------------------

_start_time = time.time()



# ---------------------------------------------------------------------------
# Middleware: request timing
# ---------------------------------------------------------------------------

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.time() - t0) * 1000, 2))
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["Root"])
async def root():
    """API welcome message and links."""
    return {
        "message": "Customer Churn Prediction API",
        "version": api_cfg["version"],
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Structured liveness / readiness check."""
    model_loaded = False
    try:
        get_model()
        model_loaded = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=api_cfg["model_path"],
        api_version=api_cfg["version"],
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    Returns:
    - **churn_probability**: Float between 0 and 1
    - **churn_label**: True if predicted to churn
    - **churn_label_text**: "Churn" or "No Churn"
    - **risk_tier**: "High" | "Medium" | "Low"
    - **confidence**: Model confidence in its prediction
    """
    try:
        model = get_model()
        data = customer.model_dump()
        result = predict_single(data, model=model, config=config)
        result["model_version"] = api_cfg["version"]
        return PredictionResponse(**result)

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {exc}. Please train the model first.",
        )
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(exc)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch_endpoint(request: BatchPredictionRequest):
    """
    Predict churn for a batch of customers (up to 1000 records).

    Returns aggregated statistics and per-record predictions.
    """
    if len(request.customers) > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size exceeds limit of 1000 records.",
        )

    try:
        model = get_model()
        records = [c.model_dump() for c in request.customers]
        df = pd.DataFrame(records)
        results_df = predict_batch(df, model=model, config=config)

        n_churn = int(results_df["churn_label"].sum())
        churn_rate = round(n_churn / len(results_df) * 100, 2)

        predictions = []
        for _, row in results_df.iterrows():
            predictions.append({
                "churn_probability": float(row["churn_probability"]),
                "churn_label": bool(row["churn_label"]),
                "risk_tier": str(row["risk_tier"]),
            })

        return BatchPredictionResponse(
            total=len(results_df),
            predicted_churn=n_churn,
            churn_rate_pct=churn_rate,
            predictions=predictions,
        )

    except Exception as exc:
        logger.exception("Batch prediction failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(exc)}",
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Return metadata about the currently loaded model."""
    meta_path = Path("models/training_metadata.json")
    if meta_path.exists():
        with open(meta_path) as fh:
            meta = json.load(fh)
    else:
        meta = {"note": "Training metadata not found. Run training first."}

    return {
        "api_version": api_cfg["version"],
        "model_path": api_cfg["model_path"],
        "churn_threshold": api_cfg["churn_threshold"],
        "training_metadata": meta,
    }


# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url)},
    )


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"},
    )


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        reload=True,
        log_level="info",
    )
