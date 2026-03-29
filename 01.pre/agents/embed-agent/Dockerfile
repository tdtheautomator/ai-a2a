# Multi-stage build for Chunking & Embedding Agent with LiteLLM
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy agent script and entrypoint
COPY ./agents/embed-agent/agent.py agent.py
COPY ./agents/embed-agent/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

# Install Python dependencies
RUN pip install --no-cache-dir \
    solace-pubsubplus \
    minio \
    qdrant-client \
    psycopg2-binary \
    requests \
    litellm \
    PyPDF2 \
    python-docx

# This Dockerfile only sets Python runtime configuration
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import psycopg2; psycopg2.connect(host='${POSTGRES_HOST}', port=${POSTGRES_PORT}, database='${POSTGRES_DB}', user='${POSTGRES_USER}', password='${POSTGRES_PASSWORD}')" || exit 1

# Run the agent
ENTRYPOINT ["/app/entrypoint.sh"]