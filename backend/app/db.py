# backend/app/db.py
import sqlite3
import json
from pathlib import Path
from typing import Any, Dict, Optional

DB_DIR = Path(__file__).resolve().parents[1] / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "software_estimation.db"

def get_connection(path: Optional[str] = None):
    p = DB_PATH if path is None else Path(path)
    conn = sqlite3.connect(str(p), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(path: Optional[str] = None):
    conn = get_connection(path)
    cur = conn.cursor()
    # Table to log prediction requests
    cur.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT (datetime('now')),
        features_json TEXT,
        prediction_json TEXT,
        notes TEXT
    )
    ''')
    # Table to log analogous cost requests
    cur.execute('''
    CREATE TABLE IF NOT EXISTS analogous_requests (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT DEFAULT (datetime('now')),
        size INTEGER,
        mean_cost_per_unit REAL,
        estimated_cost REAL,
        notes TEXT
    )
    ''')
    conn.commit()
    conn.close()
    return str(DB_PATH)

def log_prediction(features: Dict[str, Any], prediction: Any, notes: str = ""):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (features_json, prediction_json, notes) VALUES (?, ?, ?)",
        (json.dumps(features, ensure_ascii=False), json.dumps(prediction, ensure_ascii=False), notes)
    )
    conn.commit()
    last_id = cur.lastrowid
    conn.close()
    return last_id

def log_analogous(size: int, mean_cost_per_unit: float, estimated_cost: float, notes: str = ""):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO analogous_requests (size, mean_cost_per_unit, estimated_cost, notes) VALUES (?, ?, ?, ?)",
        (size, mean_cost_per_unit, estimated_cost, notes)
    )
    conn.commit()
    last_id = cur.lastrowid
    conn.close()
    return last_id
