FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt

COPY . .

CMD ["sh", "-c", "uvicorn camera_counter_service:app --host 0.0.0.0 --port ${PORT}"]
