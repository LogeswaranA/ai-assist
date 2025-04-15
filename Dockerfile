# Base stage for dependency installation
FROM python:3.9-slim AS base

WORKDIR /app

# Install system dependencies for FAISS, sentence-transformers, and curl
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY backend/requirements.txt backend_requirements.txt
COPY frontend/requirements.txt frontend_requirements.txt
RUN pip install --no-cache-dir -r backend_requirements.txt && \
    pip install --no-cache-dir -r frontend_requirements.txt

# Backend stage
FROM base AS backend
COPY backend/ /app/backend
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Frontend stage
FROM base AS frontend
COPY frontend/ /app/frontend
WORKDIR /app/frontend
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]