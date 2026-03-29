# Dockerfile
# Shared image for kb_agent, db_agent, and orchestrator.
FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the orchestrator (override via docker-compose command)
CMD ["uvicorn", "agents.orchestrator:app", "--host", "0.0.0.0", "--port", "8000"]
