# Use official Python image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy files
COPY app.py requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Command to run FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
