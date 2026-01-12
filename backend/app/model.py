# backend/app/model.py
import os
import joblib
import warnings
from pathlib import Path

# -----------------------
# Correct model path (local + cloud safe)
# -----------------------
DEFAULT_MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(Path(__file__).resolve().parents[1] / "models" / "rf_teamexp_classifier.pkl")
)

class ModelService:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model = None

        # Try loading model at startup
        try:
            self.load_model()
        except FileNotFoundError:
            warnings.warn(
                f"Model not found at {self.model_path} — "
                "prediction endpoints will fail until model is uploaded."
            )
            self.model = None

    # -----------------------
    # Load ML model
    # -----------------------
    def load_model(self):
        p = Path(self.model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.model = joblib.load(str(p))
        return self.model

    # -----------------------
    # Predict Effort (Random Forest Regressor)
    # -----------------------
    def predict_effort(self, df):
        if self.model is None:
            raise RuntimeError("Model not loaded")

        preds = self.model.predict(df)

        # Return numeric effort (hours)
        return [round(float(p), 2) for p in preds]

    # -----------------------
    # Analogous Cost Estimation
    # -----------------------
    def analogous_cost(self, size):
        """
        Estimate project cost using analogous estimation.
        """
        mean_cost_per_fp = 38.19

        base_cost = mean_cost_per_fp * float(size)

        # risk buffer (industry-style)
        min_cost = base_cost * 1.10
        max_cost = base_cost * 1.25

        recommended_bid = (min_cost + max_cost) / 2

        return {
            "mean_cost_per_fp": mean_cost_per_fp,
            "base_cost": round(base_cost, 2),
            "cost_range": {
                "min": round(min_cost, 2),
                "max": round(max_cost, 2)
            },
            "recommended_bid": round(recommended_bid, 2),
            "risk_level": "Medium"
        }

    # -----------------------
    # Feature Importance (Explainability)
    # -----------------------
    def get_feature_importance(self):
        if self.model is None:
            raise RuntimeError("Model not loaded")

        if not hasattr(self.model, "feature_importances_"):
            return {}

        return dict(
            zip(
                self.model.feature_names_in_,
                self.model.feature_importances_.tolist()
            )
        )

    # -----------------------
    # Cost from Effort (optional helper)
    # -----------------------
    def estimate_cost_from_effort(self, effort_hours):
        cost_per_hour = 500  # example industry rate
        return round(effort_hours * cost_per_hour, 2)
