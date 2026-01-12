import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -----------------------
# Load dataset
# -----------------------
df = pd.read_csv("data_saved.csv")

# -----------------------
# Define target & features
# -----------------------
TARGET = "Effort"

FEATURES = [
    "Project",
    "ManagerExp",
    "YearEnd",
    "Length",
    "Transactions",
    "Entities",
    "PointsNonAdjust",
    "Adjustment",
    "PointsAjust",
    "Language"
]

X = df[FEATURES]
y = df[TARGET]

# -----------------------
# Train-test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# Train Random Forest Regressor
# -----------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------
# Save model
# -----------------------
joblib.dump(model, "rf_effort_regressor.pkl")

print("✅ Effort prediction model trained and saved successfully")
