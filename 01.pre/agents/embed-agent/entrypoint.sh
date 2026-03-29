#!/bin/bash
# Entrypoint script for Chunking & Embedding Agent
# All configuration comes from docker-compose or environment variables

set -e

echo "🚀 Starting Chunking & Embedding Agent..."
echo "═══════════════════════════════════════════"
echo "Configuration from docker-compose:"
echo "  SOLACE_HOST: ${SOLACE_HOST}"
echo "  MINIO_HOST: ${MINIO_HOST}"
echo "  QDRANT_HOST: ${QDRANT_HOST}:${QDRANT_PORT}"
echo "  POSTGRES_HOST: ${POSTGRES_HOST}:${POSTGRES_PORT}"
echo "  LITELLM_URL: ${LITELLM_URL}"
echo "  EMBEDDING_MODEL: ${EMBEDDING_MODEL}"
echo "  VECTOR_SIZE: ${QDRANT_VECTOR_SIZE}"
echo "  CHUNKING_STRATEGY: ${CHUNKING_STRATEGY}"
echo "  POLL_INTERVAL: ${POLL_INTERVAL}"
echo "═══════════════════════════════════════════"
echo ""

# Run the agent - pass environment variables directly from docker-compose
python -u agent.py \
  --host "${SOLACE_HOST}" \
  --vpn "${SOLACE_VPN}" \
  --username "${SOLACE_USER}" \
  --password "${SOLACE_PASSWORD}" \
  --minio-host "${MINIO_HOST}" \
  --minio-user "${MINIO_USER}" \
  --minio-password "${MINIO_PASSWORD}" \
  --qdrant-host "${QDRANT_HOST}" \
  --qdrant-port "${QDRANT_PORT}" \
  --collection "${QDRANT_COLLECTION}" \
  --vector-size "${QDRANT_VECTOR_SIZE}" \
  --litellm-url "${LITELLM_URL}" \
  --litellm-key "${LITELLM_KEY}" \
  --model "${EMBEDDING_MODEL}" \
  --postgres-host "${POSTGRES_HOST}" \
  --postgres-port "${POSTGRES_PORT}" \
  --postgres-db "${POSTGRES_DB}" \
  --postgres-user "${POSTGRES_USER}" \
  --postgres-password "${POSTGRES_PASSWORD}" \
  --chunking-strategy "${CHUNKING_STRATEGY}" \
  --chunk-size "${CHUNK_SIZE}" \
  --chunk-overlap "${CHUNK_OVERLAP}" \
  --poll-interval "${POLL_INTERVAL}" \
  --batch-size "${BATCH_SIZE}"
