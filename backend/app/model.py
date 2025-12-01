# backend/app/model.py
import os
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/rf_teamexp_classifier.pkl")
DATA_PATH = os.getenv("DATA_PATH", "/app/models/data_saved.csv")  # optional historical data CSV

class ModelService:
    def __init__(self):
        self.model = None
        self.historical_df = None
        self.load_model()

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        self.model = joblib.load(MODEL_PATH)
        if os.path.exists(DATA_PATH):
            self.historical_df = pd.read_csv(DATA_PATH)
            # simple fill for numeric columns
            self.historical_df = self.historical_df.fillna(self.historical_df.median(numeric_only=True))
        else:
            self.historical_df = None

    def predict_teamexp(self, input_df):
        preds = self.model.predict(input_df)
        return preds.tolist()

    def analogous_cost(self, new_size):
        if self.historical_df is None:
            raise RuntimeError("Historical dataset not available for analogous estimation.")
        if 'Effort' not in self.historical_df.columns or 'Transactions' not in self.historical_df.columns:
            raise RuntimeError("Historical dataset missing Effort or Transactions.")
        df = self.historical_df.copy()
        df['Project'] = df['Effort'] / df['Transactions'].replace(0, np.nan)
        mean_ppu = df['Project'].dropna().mean()
        est_cost = float(mean_ppu) * float(new_size)
        return {"mean_cost_per_unit": float(mean_ppu), "estimated_cost": est_cost}
