FROM python:3.12-slim

WORKDIR /srv

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Thu muc du lieu ben vung (mount volume vao day neu muon giu lich su qua cac lan deploy)
ENV PORT=8000 DATA_DIR=/srv/data
RUN mkdir -p /srv/data
EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://127.0.0.1:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
