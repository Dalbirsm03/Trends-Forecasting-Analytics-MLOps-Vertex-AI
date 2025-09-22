# Base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and source
COPY app.py .
COPY src/ ./src/

# Expose port for FastAPI
EXPOSE 8000

# Set environment variable for MLflow tracking URI
# Replace with your MLflow server URI if needed
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

# Command to run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
