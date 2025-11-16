# Lightweight Python image
FROM python:3.11-slim

WORKDIR /app

# Install system libs for faiss, sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    libomp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install only pip first
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements
COPY server/requirements.txt /app/requirements.txt

# Install Python deps WITHOUT torch-cuda
# Force CPU-only torch (small)
RUN pip install --no-cache-dir torch==2.2.0+cpu torchvision==0.17.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install your other deps
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy backend code
COPY server/ /app/server/

EXPOSE 8000
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
