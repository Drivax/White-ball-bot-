"""
FootPredict-Pro — FastAPI REST endpoint.

Provides a production-ready REST API for match predictions.

Run: uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health          - Health check
    POST /predict         - Full match prediction
    GET  /teams           - List known teams
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from src.inference.predict_match import MatchPredictor

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="FootPredict-Pro API",
    description="Football match outcome and goal-scorer prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    home_team: str = Field(..., example="Manchester City")
    away_team: str = Field(..., example="Arsenal")
    home_lineup: Optional[List[str]] = Field(
        None,
        example=["Ederson", "Walker", "Dias", "Akanji", "Gvardiol",
                 "Rodri", "De Bruyne", "Silva", "Doku", "Foden", "Haaland"],
    )
    away_lineup: Optional[List[str]] = Field(None)
    home_positions: Optional[Dict[str, str]] = Field(
        None, example={"Haaland": "ST", "De Bruyne": "CAM"}
    )
    away_positions: Optional[Dict[str, str]] = Field(None)


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str = "1.0.0"


# ---------------------------------------------------------------------------
# Predictor (initialized on startup)
# ---------------------------------------------------------------------------

_predictor: Optional[MatchPredictor] = None


@app.on_event("startup")
async def startup_event() -> None:
    global _predictor
    _predictor = MatchPredictor()
    _predictor.load()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        models_loaded=_predictor is not None and _predictor._loaded,
    )


@app.post("/predict")
async def predict(request: PredictRequest) -> dict:
    """
    Generate a full football match prediction.

    Returns calibrated outcome probabilities, expected goals,
    most likely scorelines, and player goal probabilities.
    """
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    try:
        pred = _predictor.predict(
            home_team=request.home_team,
            away_team=request.away_team,
            home_lineup=request.home_lineup,
            away_lineup=request.away_lineup,
            home_positions=request.home_positions,
            away_positions=request.away_positions,
        )
        return pred.to_dict()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@app.get("/")
async def root() -> dict:
    """API root."""
    return {
        "name": "FootPredict-Pro API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
