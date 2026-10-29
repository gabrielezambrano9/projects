FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    git build-essential \
    libgl1 libglfw3 libosmesa6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt