FROM python:3.11-slim

WORKDIR /app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    for i in 1 2 3 4 5; do \
        pip install --no-cache-dir --timeout=600 -r requirements.txt && break || sleep 10; \
    done

# Copy application code
COPY . .

# Expose service port
EXPOSE 8003

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
