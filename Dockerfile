FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default to port 8080 for Cloud Run
ENV PORT=8080

# Expose the Cloud Run port
EXPOSE ${PORT}

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

# Command to run Streamlit - properly configured for Cloud Run
CMD streamlit run app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.baseUrlPath="" \
    --browser.serverAddress="0.0.0.0" \
    --server.headless=true
