# Stage 1: build the frontend
FROM node:22-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# Stage 2: Python API serving the built frontend
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/var/data REPLAY_DIR=/app/backend/replays
WORKDIR /app
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ backend/
COPY pytest.ini NOTES.md README.md ./
COPY --from=web /web/dist frontend/dist
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.api.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
