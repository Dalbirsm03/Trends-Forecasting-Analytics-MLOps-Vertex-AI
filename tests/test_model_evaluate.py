import pandas as pd
import numpy as np
from src import model_train, model_evaluate

X = pd.DataFrame({"feature1": np.random.rand(10), "feature2": np.random.rand(10)})
Y = pd.Series(np.random.rand(10))

# Train a model so it exists in MLflow
def test_evaluate_after_training():
    model_train.train_and_register(model_train.LinearRegression(), "linear_test_eval", X, Y)
    metrics = model_evaluate.evaluate_all("linear_test_eval", X, Y)
    assert "mse" in metrics
    assert "r2" in metrics