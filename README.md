# Enterprise Sales Forecasting & ML Serving Platform

## 📌 Executive Summary
This repository contains an end-to-end Machine Learning pipeline and model serving architecture designed for high-throughput, multivariate time-series forecasting. The system ingests raw transactional data, automatically engineers temporal and statistical features, trains multiple machine learning models (with automated experiment tracking), and deploys the winning models via a robust, containerized REST API.

To ensure enterprise-readiness, the platform features a **dynamic Streamlit dashboard** for business stakeholders to interact with the models, and an automated **Data Drift Monitoring system (Evidently AI)** to detect when the model's accuracy degrades in production.

The platform handles forecasting for **171 distinct time-series** concurrently (representing 19 unique product lines across 9 geographic regions), processing over 185,000 historical sales records to predict future revenue and demand.

## 🏗️ Architecture & Technology Stack

### 1. Data Engineering & Processing
* **Libraries:** `Pandas`, `NumPy`, `PyArrow` (Parquet)
* **Workflow:** Raw CSV ingestion is transformed into optimized `.parquet` formats. Outlier detection, handling of missing values, and data type optimization are performed to ensure high-fidelity inputs for model training.

### 2. Feature Engineering & Time-Series Analysis
* **Libraries:** `Statsmodels`, `Prophet`, `Scikit-Learn`
* **Features Generated:**
  * **Temporal Extractors:** Quarter, Month, Week, Year, Weekday/Weekend binary encoding.
  * **Rolling Statistics:** Simple Moving Averages (`SMA_3`, `SMA_5`) to capture short-term demand trends.
  * **Categorical Encoding:** Target mean encoding for high-cardinality features like `Product` and `City`.
* **Exploratory Modeling:** Initial univariate baselines established using **ARIMA** and **Facebook Prophet** (documented in `notebooks/time_series_forecasting.ipynb`).

### 3. Machine Learning & MLOps Pipeline
* **Libraries:** `Scikit-Learn`, `XGBoost`, `MLflow`
* **Models Deployed:**
  * **Linear Regression** (Baseline Multivariate)
  * **Random Forest Regressor** (Non-linear Ensemble)
  * **XGBoost Regressor** (Gradient Boosted Trees)
* **MLflow Integration:** The `src/model_train.py` pipeline automatically logs experiment runs, model parameters, and evaluation metrics (MSE, $R^2$) to a centralized model registry. Trained artifacts are persisted as serialized `.pkl` objects.

### 4. API Development & Model Serving
* **Libraries:** `FastAPI`, `Pydantic`, `Uvicorn`
* **Architecture:** A dynamic routing layer allows clients to request predictions from any registered model via the `/predict/{model_name}` endpoint. `Pydantic` ensures strict schema validation for all incoming JSON payloads.

### 5. Interactive Frontend Dashboard (Streamlit)
* **Libraries:** `Streamlit`, `Plotly`, `Requests`
* **Features:** A dynamic, user-friendly UI that allows business stakeholders to select specific Product and City combinations. The dashboard automatically queries historical averages to pre-fill features, queries the FastAPI backend for predictions, and displays contextual historical sales trends using interactive Plotly charts.

### 6. Data Drift & Model Monitoring (Evidently AI)
* **Libraries:** `Evidently AI`
* **Features:** In production, models degrade over time (Concept Drift). The FastAPI backend securely logs every incoming prediction payload to `logs/api_logs.csv`. An automated monitoring script (`src/monitoring.py`) compares this live production traffic against the original training dataset to detect **Data Drift**, generating comprehensive HTML diagnostic reports to ensure model reliability.

### 7. Deployment & Cloud Integration
* **Infrastructure:** `Docker`, `Google Cloud Storage (GCS)`, `Vertex AI`
* **Containerization:** A lightweight `python:3.11-slim` Dockerfile packages the API, ensuring environment parity across development, staging, and production.
* **Cloud:** Integrates with GCP service accounts for seamless artifact retrieval and scalable deployment to Vertex AI endpoints.

## 📊 Scale, Testing & Performance Metrics

* **Data Volume:** 185,000+ historical transactions.
* **Time Series Complexity:** 171 concurrent product-region combinations.
* **API Latency:** Load testing via the custom `test_latency.py` suite demonstrates sub-50ms 95th percentile response times.
* **Throughput:** A comprehensive `locustfile.py` load test simulates high-concurrency traffic, proving the API's ability to sustain high Requests-Per-Second (RPS).

---

## 🚀 Local Development & Execution Guide

### 1. Environment Setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the Prediction API (Backend)
Launch the FastAPI server on port 8000:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Start the Interactive Dashboard (Frontend)
Launch the Streamlit app to interact with the models visually:
```bash
streamlit run frontend.py
```
*Navigate to `http://localhost:8501` in your browser.*

### 4. MLOps: Data Drift Monitoring
After generating some traffic on the API (via the dashboard or latency tests), check if the real-world data is drifting from the training data:
```bash
python src/monitoring.py
```
*Open `logs/drift_report.html` to view the Evidently AI diagnostics dashboard.*

### 5. Performance & Load Testing
**Measure Latency:**
```bash
python test_latency.py
```

**Measure Throughput:**
```bash
locust -f locustfile.py
```
*Navigate to `http://localhost:8089` to configure the swarm and monitor real-time RPS.*
