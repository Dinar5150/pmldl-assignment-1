import os
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_PATH = os.path.join(REPO_ROOT, "models", "linear_regression.joblib")

model = None


def get_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model


class Input(BaseModel):
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


app = FastAPI()


@app.post("/predict")
def predict(x: Input):
    model = get_model()
    features = [[x.age, x.sex, x.bmi, x.bp, x.s1, x.s2, x.s3, x.s4, x.s5, x.s6]]
    pred = float(model.predict(features)[0])
    return {"prediction": pred}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)
