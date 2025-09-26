from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn

# FastAPI app
app = FastAPI(title="Sales Forecasting API")

# Pydantic model for input
class InputData(BaseModel):
    price_each: float
    product_mean_encoded: float
    city_mean_encoded: float
    SMA_3: float
    SMA_5: float
    quantity_ordered: float
    quarter: int
    month: int
    week: int
    year: int
    weekday_weekend_encoded: int

@app.get("/")
def root():
    return {"message": "Sales Forecasting API is running"}

@app.post("/predict/{model_name}")
def predict(model_name: str, data: InputData, version: str = "version-1"):
    """
    model_name: linear_regression / random_forest / xgboost
    version: version folder in GCS
    """
    # GCS bucket path
    model_uri = f"gs://trend-forecast-models/{model_name}/{version}/artifacts"
    
    # Load model from GCS
    model = mlflow.sklearn.load_model(model_uri)
    
    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])
    
    # Predict
    preds = model.predict(df)
    
    return {"prediction": preds.tolist()}
