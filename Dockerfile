FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories with proper permissions
RUN mkdir -p /app/app/models /app/app/static/samples /app/app/static/uploads && \
    chmod -R 755 /app/app/static

# Copy the application code
COPY . .

# Ensure static files have correct permissions
RUN chmod -R 755 /app/app/static/samples /app/app/models

# Set environment variables
ENV FLASK_APP=run.py \
    FLASK_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/app/models \
    PORT=8080 \
    PYTHONPATH=/app

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the application
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 run:app