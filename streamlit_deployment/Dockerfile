# Dockerfile for Utility Inventory Detection Model
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY streamlit_deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY streamlit_deployment/ ./streamlit_deployment/
COPY models/ ./models/
COPY training/logs/training_log.json ./streamlit_deployment/training_log.json

# Expose Streamlit port
EXPOSE 8501

# Set working directory
WORKDIR /app/streamlit_deployment

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/_stcore/health')"

# Run Streamlit
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

