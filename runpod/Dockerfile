FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY runpod/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

COPY runpod/ /app/runpod/
COPY model/ /app/model/

WORKDIR /app/runpod

CMD ["python", "handler.py"]
