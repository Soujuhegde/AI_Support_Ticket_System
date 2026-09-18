FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Default command; overridden per-service in docker-compose.yml
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
