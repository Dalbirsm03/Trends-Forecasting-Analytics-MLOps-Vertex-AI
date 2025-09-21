import os
from logs.logger import get_logger
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

logger = get_logger(__name__, log_file="logs/project.log")

def evaluate_all(model_name, X, Y, version=None):
    if version:
        model_uri = f"models:/{model_name}/{version}"
    else:
        model_uri = f"models:/{model_name}/latest"

    model = mlflow.sklearn.load_model(model_uri)
    preds = model.predict(X)
    mse = mean_squared_error(Y, preds)
    r2 = r2_score(Y, preds)

    mlflow.set_experiment("Sales_Forecasting")
    with mlflow.start_run(run_name=f"{model_name}_eval"):
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

    logger.info(f"{model_name} evaluation → MSE: {mse:.4f}, R2: {r2:.4f}")
    return {"mse": mse, "r2": r2}
