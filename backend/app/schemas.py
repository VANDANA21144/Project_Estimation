# backend/app/schemas.py
from pydantic import BaseModel
from typing import Dict, Any

# -----------------------
# Prediction (Effort)
# -----------------------
class PredictRequest(BaseModel):
    features: Dict[str, Any]

class PredictResponse(BaseModel):
    estimated_effort_hours: float


# -----------------------
# Analogous Cost Estimation
# -----------------------
class AnalogousRequest(BaseModel):
    size: int


class CostRange(BaseModel):
    min: float
    max: float


class AnalogousResponse(BaseModel):
    mean_cost_per_fp: float
    base_cost: float
    cost_range: CostRange
    recommended_bid: float
    risk_level: str
