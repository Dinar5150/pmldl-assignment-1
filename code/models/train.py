import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    proc_train = os.path.join(repo_root, "data", "processed", "train.csv")
    proc_test = os.path.join(repo_root, "data", "processed", "test.csv")

    df_train = pd.read_csv(proc_train)
    df_test = pd.read_csv(proc_test)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_test = df_test.drop(columns=["target"])
    y_test = df_test["target"]

    param_grid = {"alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    search = GridSearchCV(Ridge(), param_grid, cv=5, n_jobs=-1, scoring="neg_mean_squared_error")

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    preds = best_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("Best params:", search.best_params_)
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    models_dir = os.path.join(repo_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "linear_regression.joblib")
    joblib.dump(best_model, model_path)

    mlflow.set_experiment("diabetes_linear_regression")
    with mlflow.start_run():
        for k, v in search.best_params_.items():
            mlflow.log_param(k, v)

        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))

        mlflow.sklearn.log_model(best_model, "model")
        mlflow.log_artifact(model_path)

    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
