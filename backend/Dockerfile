FROM python:3.10-slim

WORKDIR /app

# Copy backend code
COPY app/ ./app
COPY models/ ./models

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir fastapi uvicorn[standard] pandas scikit-learn joblib python-multipart

EXPOSE 8000

ENV MODEL_PATH=/app/models/rf_teamexp_classifier.pkl
ENV DATA_PATH=/app/models/data_saved.csv

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
