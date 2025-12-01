# backend/app/main.py (with admin token dependency)
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Depends
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from .model import ModelService
from .schemas import PredictRequest, PredictResponse, AnalogousRequest, AnalogousResponse
from .db import log_prediction, log_analogous
from .auth import get_current_admin
from pathlib import Path
import shutil
import datetime
import os

app = FastAPI(title="Software Estimation API")

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for local dev; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service = ModelService()
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CURRENT_MODEL_PATH = MODELS_DIR / "rf_teamexp_classifier.pkl"

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, request: Request):
    try:
        df = pd.DataFrame([req.features])
        try:
            preds = model_service.predict_teamexp(df)
        except Exception as e_inner:
            if hasattr(model_service.model, "feature_names_in_"):
                df = df.reindex(columns=list(model_service.model.feature_names_in_), fill_value=0)
                preds = model_service.predict_teamexp(df)
            else:
                raise
        try:
            log_prediction(req.features, preds, notes=f"remote={request.client.host}")
        except Exception:
            pass
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analogous", response_model=AnalogousResponse)
async def analogous(r: AnalogousRequest, request: Request):
    try:
        res = model_service.analogous_cost(r.size)
        try:
            log_analogous(r.size, res.get("mean_cost_per_unit"), res.get("estimated_cost"), notes=f"remote={request.client.host}")
        except Exception:
            pass
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# -----------------------
# Admin: upload new model (protected)
# -----------------------
@app.post("/admin/upload-model")
async def upload_model(file: UploadFile = File(...), admin=Depends(get_current_admin)):
    """
    Upload a new model file (pickle). Protected by simple Bearer token.
    Save as models/rf_teamexp_classifier.pkl and reload model service.
    """
    try:
        if not file.filename.lower().endswith(('.pkl', '.joblib')):
            raise HTTPException(status_code=400, detail="Only .pkl or .joblib model files accepted.")

        # backup existing model if present
        if CURRENT_MODEL_PATH.exists():
            ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            backup_path = CURRENT_MODEL_PATH.with_name(f"rf_teamexp_classifier_backup_{ts}.pkl")
            shutil.copy2(CURRENT_MODEL_PATH, backup_path)

        # save uploaded file to CURRENT_MODEL_PATH
        with CURRENT_MODEL_PATH.open("wb") as out_f:
            content = await file.read()
            out_f.write(content)

        # reload model
        try:
            model_service.load_model()
        except Exception as e_load:
            if 'backup_path' in locals() and backup_path.exists():
                shutil.copy2(backup_path, CURRENT_MODEL_PATH)
                model_service.load_model()
            raise HTTPException(status_code=500, detail=f"Model load failed: {e_load}")

        return {"detail": "model uploaded and reloaded successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
