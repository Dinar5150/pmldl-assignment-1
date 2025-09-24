from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "model.joblib")

# Try to load model at startup
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)

# Define features explicitly
class PredictRequest(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    target: float

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        return {"error": "model not found"}
    
    # Create array in the correct order
    features = [
        req.age, req.sex, req.bmi, req.bp,
        req.s1, req.s2, req.s3, req.s4,
        req.s5, req.s6, req.target
    ]
    arr = np.array(features).reshape(1, -1)
    pred = model.predict(arr)
    return {"prediction": float(pred[0])}
