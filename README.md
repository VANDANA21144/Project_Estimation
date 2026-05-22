# ⚡ EstimateIQ — AI-Powered Software Project Estimation

> Predict development effort, derive cost ranges, and surface similar historical projects — powered by a Random Forest model trained on the Desharnais dataset.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📸 Overview

EstimateIQ is a full-stack software project estimation platform that replaces spreadsheet guesswork with machine learning. You input 10 project attributes, and the system returns predicted effort in hours, a derived cost range, a recommended bid, and the 5 most similar historical projects — all logged to a persistent audit trail.

**Two estimation methods:**
- **ML Prediction** — Random Forest Regressor (300 trees, 10 features) trained on 81 real-world software projects from the Desharnais dataset
- **Analogous Estimation** — Finds the 5 nearest historical projects by Adjusted Function Points, scales their effort/FP ratio to your target size

---

## ✨ Features

| Feature | Description |
|---|---|
| 🧠 **ML Effort Prediction** | RF model predicts development hours from 10 project attributes |
| 📊 **Analogous Estimation** | Historical similarity search with proportional effort scaling |
| 💰 **Cost Derivation** | `Cost = Effort × Hourly Rate` — transparent, consistent, configurable |
| 📈 **Model Analysis Page** | Scatter plots, MAE/R² comparison vs COCOMO and Expert Judgment |
| 🎛 **Feature Importance** | Visual breakdown of which attributes drive the model most |
| 🗄️ **Prediction History** | SQLite audit log with search, export to CSV, and trend charts |
| 🔐 **Admin Model Upload** | Bearer-token protected hot-reload — upload a new `.pkl` without downtime |
| 🐳 **Docker Compose** | One-command full-stack deployment |

---

## 🏗️ Project Structure

```
Project_Estimation/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI app, all endpoints
│   │   ├── model.py         # ModelService — predict, analogous, metrics
│   │   ├── schemas.py       # Pydantic request/response models
│   │   ├── db.py            # SQLite helpers — init, log_prediction, log_analogous
│   │   └── auth.py          # Bearer token admin auth
│   ├── models/
│   │   ├── rf_effort_model.pkl   # Trained Random Forest
│   │   ├── data_saved.csv        # Desharnais dataset (81 projects)
│   │   └── model_metrics.json    # R², MAE, RMSE, feature importances
│   ├── database/            # SQLite DB volume (auto-created)
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── index.html           # Landing page
│   ├── dashboard.html       # Main estimator UI
│   ├── analysis.html        # Model benchmarking & scatter plots
│   ├── history.html         # Prediction audit log
│   └── styles.css           # Shared design system
├── tests/
├── data_saved.csv
├── train_effort_model.py    # Model training script
├── project.ipynb            # EDA notebook
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Quick Start

### Option 1 — Docker Compose (Recommended)

```bash
# Clone
git clone https://github.com/VANDANA21144/Project_Estimation.git
cd Project_Estimation

# Start backend + frontend
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:8080 |
| Backend API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

### Option 2 — Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt

# Copy model and data into place
mkdir -p models
cp ../rf_effort_model.pkl models/
cp ../data_saved.csv models/

uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
# Any static server works — e.g.:
cd frontend
python -m http.server 8080
# or
npx serve .
```

Then open http://localhost:8080.

---

## 🔌 API Reference

Base URL: `http://localhost:8000`

### `GET /health`
Returns `{"status": "ok"}`. Use for health checks.

---

### `POST /predict`
Predict development effort using the Random Forest model.

**Request:**
```json
{
  "features": {
    "Project": 10,
    "ManagerExp": 5,
    "YearEnd": 88,
    "Length": 12,
    "Transactions": 200,
    "Entities": 100,
    "PointsNonAdjust": 300,
    "Adjustment": 30,
    "PointsAjust": 310,
    "Language": 2
  }
}
```

**Response:**
```json
{
  "estimated_effort_hours": 4892.0,
  "cost_per_hour": 500,
  "base_cost": 2446000.0,
  "cost_range": { "min": 2690600.0, "max": 3057500.0 },
  "recommended_bid": 2874050.0,
  "risk_level": "Medium"
}
```

---

### `POST /analogous`
Analogous estimation — finds the 5 most similar historical projects by Adjusted FP and scales their effort/FP ratio to your target size.

**Request:**
```json
{ "size": 350 }
```

**Response:**
```json
{
  "size": 350,
  "mean_cost_per_fp": 14200.5,
  "estimated_effort_hours": 9940.35,
  "base_cost": 4970175.0,
  "cost_range": { "min": 5467192.5, "max": 6212718.75 },
  "recommended_bid": 5839955.6,
  "risk_level": "High",
  "similar_projects": [...]
}
```

---

### `GET /metrics`
Returns model accuracy metrics (R², MAE, RMSE, CV scores, feature importances).

### `GET /feature-importance`
Returns a dict of `{ feature_name: importance_score }`.

### `GET /history?limit=50`
Returns recent predictions from the SQLite log.

### `POST /admin/upload-model` *(protected)*
Upload a new `.pkl` model file. Requires `Authorization: Bearer <ADMIN_TOKEN>` header. Automatically backs up the current model before replacing it.

```bash
curl -X POST http://localhost:8000/admin/upload-model \
  -H "Authorization: Bearer supersecretadmintoken" \
  -F "file=@rf_effort_model.pkl"
```

---

## 🤖 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Trees | 300 |
| Dataset | Desharnais (81 software projects) |
| Target | Effort (development hours) |
| R² Score | 0.615 |
| MAE | 1,740 hrs |
| RMSE | 2,216 hrs |
| CV R² Mean | 0.174 (high variance — small dataset) |

**Feature Importances (top 5):**

| Feature | Importance |
|---|---|
| Length (months) | 30.7% |
| Adjusted FP | 20.2% |
| Unadjusted FP | 12.7% |
| Entities | 8.5% |
| Transactions | 8.1% |

**Cost formula:** `Base Cost = Effort × COST_PER_HOUR` (default ₹500/hr, configurable via env var)

**Bid range:** Base × 1.10 (min) to Base × 1.25 (max)

---

## 🔁 Retraining the Model

```bash
# Edit train_effort_model.py as needed, then:
python train_effort_model.py
# Outputs: rf_effort_model.pkl + model_metrics.json

# Upload without restarting the server:
curl -X POST http://localhost:8000/admin/upload-model \
  -H "Authorization: Bearer supersecretadmintoken" \
  -F "file=@rf_effort_model.pkl"
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `models/rf_effort_model.pkl` | Path to the trained model file |
| `COST_PER_HOUR` | `500` | Hourly rate in ₹ used for cost calculation |
| `ADMIN_TOKEN` | `supersecretadmintoken` | Bearer token for `/admin/upload-model` |

> ⚠️ Change `ADMIN_TOKEN` before any production deployment.

---

## 🧪 Running Tests

```bash
pip install -r requirements-ci.txt
pytest tests/ -v
```

CI runs automatically on every push via GitHub Actions (`.github/workflows/`).

---

## 📊 Dataset

This project uses the **Desharnais dataset** — a widely-cited benchmark in software cost estimation research, containing 81 real software projects with attributes including function points, team experience, project duration, and actual development effort.

- **81 projects**, **10 features**, **1 target** (Effort in hours)
- Used in academic literature for COCOMO comparisons and ML benchmarking
- See `project.ipynb` for full exploratory data analysis

---

## 📁 Frontend Pages

| Page | Purpose |
|---|---|
| `index.html` | Landing page — features, how-it-works, model stats |
| `dashboard.html` | Main estimator — ML prediction + analogous estimation |
| `analysis.html` | Model benchmarking vs COCOMO, scatter plots, feature importance |
| `history.html` | Prediction audit log — search, filter, CSV export, trend charts |

---

## 🛠️ Tech Stack

**Backend:** Python 3.10 · FastAPI · scikit-learn · pandas · SQLite · joblib  
**Frontend:** Vanilla HTML/CSS/JS · Chart.js · Syne + JetBrains Mono fonts  
**Infrastructure:** Docker · Docker Compose · Nginx · GitHub Actions CI

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙋 Author

**Vandana** — [@VANDANA21144](https://github.com/VANDANA21144)

*Built as a practical demonstration of ML-based software cost estimation using the Desharnais benchmark dataset.*
