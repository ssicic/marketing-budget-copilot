FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MBC_ARTIFACT_DIR=/app/artifacts/champion \
    PORT=8080

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ src/
COPY artifacts/champion/ artifacts/champion/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8080
# Cloud Run sets $PORT; default to 8080 locally.
CMD ["sh", "-c", "exec uvicorn mbc.api.app:app --host 0.0.0.0 --port ${PORT}"]
