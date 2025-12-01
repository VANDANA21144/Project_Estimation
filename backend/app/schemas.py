# backend/app/schemas.py
from pydantic import BaseModel
from typing import Dict, Any, List

class PredictRequest(BaseModel):
    features: Dict[str, Any]

class PredictResponse(BaseModel):
    predictions: List[Any]

class AnalogousRequest(BaseModel):
    size: int

class AnalogousResponse(BaseModel):
    mean_cost_per_unit: float
    estimated_cost: float
