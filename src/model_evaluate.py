import os
# from logs.logger import get_logger
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn

# logger = get_logger(__name__, log_file="logs/project.log")

def evaluate_all(model_name, X, Y, run_id = None,version=None):
    if version:
        model_uri = f"models:/{model_name}/{version}"
    else:
        model_uri = f"models:/{model_name}/latest"

    model = mlflow.sklearn.load_model(model_uri)
    preds = model.predict(X)
    mse = mean_squared_error(Y, preds)
    r2 = r2_score(Y, preds)

    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

    return {"mse": mse, "r2": r2}
