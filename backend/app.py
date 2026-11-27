"""Minimal API for loading saved student-project models."""
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException\nfrom fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.config import MODEL_DIR

app = FastAPI(title="Multi-Disease Risk Prediction - Academic Demo",
              description="Educational classification only; not for diagnosis or treatment.")

class PredictionInput(BaseModel):
    disease: str
    model: str = "random_forest"
    features: dict

@app.get("/")
def home():
    return {"message": "Academic ML demo API", "warning": "Not a medical diagnostic tool."}

@app.post("/predict")
def predict(payload: PredictionInput):
    allowed = {"diabetes", "heart", "kidney", "breast_cancer"}
    if payload.disease not in allowed:
        raise HTTPException(400, "Unsupported disease key.")
    model_path = MODEL_DIR / payload.disease / f"{payload.model}.joblib"
    if not model_path.exists():
        raise HTTPException(404, "Model not trained. Train it before requesting predictions.")
    saved = joblib.load(model_path)
    expected = saved["features"]
    missing = sorted(set(expected) - set(payload.features))
    extra = sorted(set(payload.features) - set(expected))
    if missing or extra:
        raise HTTPException(422, {"missing_features": missing, "unexpected_features": extra})
    row = pd.DataFrame([{k: payload.features[k] for k in expected}])
    pipeline = saved["pipeline"]
    prediction = pipeline.predict(row)[0]
    result = {"predicted_class": str(prediction),
              "notice": "Educational model output only; not a clinical risk estimate."}
    if hasattr(pipeline, "predict_proba"):
        result["model_score"] = float(pipeline.predict_proba(row).max())
    return result
