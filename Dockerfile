# Orchestration image for the thesis RAG project.
# Deliberately WITHOUT torch, FlagEmbedding, or model weights:
# all embedding/reranking goes to a GPU server configured via GPU_REMOTE_URL
# (see docker-compose.yml). Keep this image small and portable.
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/
