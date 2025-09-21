import os
import pandas as pd
from src import data_load

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
def test_load_data():
    df = pd.read_parquet(os.path.join(BASE_DIR, "data", "processed_file", "sales_feature_selected.parquet"))
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    for col in ["sales", "price_each", "product_mean_encoded"]:
        assert col in df.columns
