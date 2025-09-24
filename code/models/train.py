import os
import joblib
import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROCESSED_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MLFLOW_TRACKING_DIR = os.path.join(ROOT, "mlruns")

mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")

train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

X_train = train.drop(columns=["target"]) if "target" in train.columns else train.iloc[:, :-1]
if "target" in train.columns:
    y_train = train["target"]
else:
    y_train = train.iloc[:, -1]

X_test = test.drop(columns=["target"]) if "target" in test.columns else test.iloc[:, :-1]
if "target" in test.columns:
    y_test = test["target"]
else:
    y_test = test.iloc[:, -1]

model = Ridge(alpha=1.0)

with mlflow.start_run():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)

    model_path = os.path.join(MODELS_DIR, "model.joblib")
    joblib.dump(model, model_path)
    mlflow.sklearn.log_model(model, "model")

    metrics = {"rmse": float(rmse), "mae": float(mae)}
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f)

print(f"Model saved to {model_path}. Metrics: {metrics}")
