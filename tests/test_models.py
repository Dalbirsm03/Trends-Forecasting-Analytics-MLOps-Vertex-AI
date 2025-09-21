import pandas as pd
import numpy as np
from src import model_train, model_evaluate
X = pd.DataFrame({"feature1": np.random.rand(10), "feature2": np.random.rand(10)})
Y = pd.Series(np.random.rand(10))
def test_train_all_models():
    models = model_train.train_all(X, Y)
    assert "linear" in models
    assert "random_forest" in models
    assert "xgboost" in models
    for m in models.values():
        assert m.score(X, Y) >= 0
