import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_register(model, model_name, X_train, Y_train):
    model.fit(X_train, Y_train)
    path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    joblib.dump(model, path)

    mlflow.set_experiment("Sales_Forecasting")
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=model_name
        )
        run_id = run.info.run_id   # capture the run ID

    return model, run_id

def train_all(X_train, Y_train):
    models = {}
    run_ids = {}

    # Linear Regression
    models['linear_regression'], run_ids['linear_regression'] = train_and_register(
        LinearRegression(), "linear_regression", X_train, Y_train
    )

    # Random Forest
    models['random_forest'], run_ids['random_forest'] = train_and_register(
        RandomForestRegressor(n_estimators=100, random_state=42),
        "random_forest", X_train, Y_train
    )

    # XGBoost
    models['xgboost'], run_ids['xgboost'] = train_and_register(
        XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "xgboost", X_train, Y_train
    )

    return models, run_ids
