# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for FAISS & Transformers
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy server folder
COPY server/ /app/server/

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Expose port (Railway/Render will map automatically)
EXPOSE 8000

# Start FastAPI server using uvicorn
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
