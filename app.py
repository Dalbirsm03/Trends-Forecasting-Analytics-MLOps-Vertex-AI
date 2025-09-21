from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn

app = FastAPI(title="Sales Forecasting API")

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
def predict(model_name: str, data: InputData, version: str = None):
    model_uri = f"models:/{model_name}/{version}" if version else f"models:/{model_name}/latest"
    model = mlflow.sklearn.load_model(model_uri)
    
    df = pd.DataFrame([data.dict()])
    preds = model.predict(df)
    
    return {"prediction": preds.tolist()}
