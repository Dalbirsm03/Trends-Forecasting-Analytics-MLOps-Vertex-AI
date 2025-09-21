import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from logs.logger import get_logger

logger = get_logger(__name__, log_file="logs/project.log")

def load_data():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df = pd.read_parquet(os.path.join(BASE_DIR, "data", "processed_file", "sales_feature_selected.parquet"))
    X = df.drop(columns=['sales'])
    Y = df['sales']
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, stratify=None, test_size=0.1, random_state=42
    )

    logger.info(f"Data loaded: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, X_test, Y_train, Y_test
