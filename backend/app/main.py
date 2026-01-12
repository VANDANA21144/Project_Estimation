# backend/app/main.py
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from .model import ModelService
from .schemas import (
    PredictRequest,
    PredictResponse,
    AnalogousRequest,
    AnalogousResponse
)
from .db import log_prediction, log_analogous
from .auth import get_current_admin
from pathlib import Path
import shutil
import datetime

app = FastAPI(title="Project Estimation API")

# -----------------------
# CORS
# -----------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Model setup
# -----------------------
model_service = ModelService()

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_MODEL_PATH = MODELS_DIR / "rf_teamexp_classifier.pkl"

# -----------------------
# Health check
# -----------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# -----------------------
# Predict Effort (ML)
# -----------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    """
    Predict development effort (in hours) using Random Forest Regressor
    """
    try:
        df = pd.DataFrame([req.features])

        # Align columns with training features
        if hasattr(model_service.model, "feature_names_in_"):
            df = df.reindex(
                columns=list(model_service.model.feature_names_in_),
                fill_value=0
            )

        estimated_effort = model_service.predict_effort(df)[0]

        # log prediction (best effort, non-blocking)
        try:
            log_prediction(
                req.features,
                {"estimated_effort_hours": estimated_effort},
                notes=f"remote={request.client.host}"
            )
        except Exception:
            pass

        return {"estimated_effort_hours": estimated_effort}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------
# Analogous Cost Estimation
# -----------------------
@app.post("/analogous", response_model=AnalogousResponse)
async def analogous(r: AnalogousRequest, request: Request):
    """
    Estimate project cost using analogous estimation
    """
    try:
        res = model_service.analogous_cost(r.size)

        # log cost estimation
        try:
            log_analogous(
                r.size,
                res.get("mean_cost_per_fp"),
                res.get("base_cost"),
                notes=f"remote={request.client.host}"
            )
        except Exception:
            pass

        return res

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------
# Feature Importance (Explainability)
# -----------------------
@app.get("/feature-importance")
async def feature_importance():
    """
    Return feature importance from Random Forest model
    """
    try:
        return model_service.get_feature_importance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------
# Admin: upload new model
# -----------------------
@app.post("/admin/upload-model")
async def upload_model(
    file: UploadFile = File(...),
    admin=Depends(get_current_admin)
):
    """
    Upload a new trained model (.pkl / .joblib) and reload it safely
    """
    try:
        if not file.filename.lower().endswith((".pkl", ".joblib")):
            raise HTTPException(
                status_code=400,
                detail="Only .pkl or .joblib model files are allowed"
            )

        # backup old model
        if CURRENT_MODEL_PATH.exists():
            ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            backup_path = CURRENT_MODEL_PATH.with_name(
                f"rf_teamexp_classifier_backup_{ts}.pkl"
            )
            shutil.copy2(CURRENT_MODEL_PATH, backup_path)

        # save new model
        with CURRENT_MODEL_PATH.open("wb") as f:
            content = await file.read()
            f.write(content)

        # reload model
        try:
            model_service.load_model()
        except Exception as load_err:
            if "backup_path" in locals() and backup_path.exists():
                shutil.copy2(backup_path, CURRENT_MODEL_PATH)
                model_service.load_model()
            raise HTTPException(
                status_code=500,
                detail=f"Model load failed: {load_err}"
            )

        return {"detail": "Model uploaded and reloaded successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
