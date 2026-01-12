import os, traceback
try:
    import joblib
except Exception as e:
    print("joblib import error:", e); raise

p = os.path.join(os.getcwd(), "backend", "models", "rf_teamexp_classifier.pkl")
print("MODEL_PATH we will test:", p)
print("Exists:", os.path.exists(p))

if os.path.exists(p):
    try:
        m = joblib.load(p)
        print("Loaded model object type:", type(m))
        print("Model load: OK")
    except Exception:
        print("Model load error (traceback below):")
        traceback.print_exc()
        raise
else:
    print("Model file missing. Create or copy a .pkl to backend/models/")
