# backend/app/model.py
import os
import joblib
import warnings
from pathlib import Path

DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/rf_teamexp_classifier.pkl")

class ModelService:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.model = None
        # attempt to load model but do not raise on missing file (tests/CI friendly)
        try:
            self.load_model()
        except FileNotFoundError:
            warnings.warn(f"Model not found at {self.model_path} — continuing without model (health endpoints still work).")
            self.model = None
        except Exception as e:
            # If other exceptions occur during load, re-raise so real runs fail loudly
            raise

    def load_model(self):
        """
        Load the model from disk. Raises FileNotFoundError if file missing.
        """
        p = Path(self.model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        # Use joblib for loading sklearn models
        self.model = joblib.load(str(p))
        return self.model

    def predict_teamexp(self, df):
        """
        Predict using the loaded model. If model is not loaded, raise informative error.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot predict. Upload or provide a model file.")
        # original model.predict might return array-like
        preds = self.model.predict(df)
        # ensure JSON-serializable list
        return preds.tolist() if hasattr(preds, "tolist") else list(preds)

    def analogous_cost(self, size):
        """
        Example analogous method — uses internal data or fallback if model not present.
        """
        # Fallback simple calculation if data not available
        mean_cost_per_unit = 38.191011608301174
        estimated_cost = mean_cost_per_unit * float(size)
        return {"mean_cost_per_unit": mean_cost_per_unit, "estimated_cost": estimated_cost}
