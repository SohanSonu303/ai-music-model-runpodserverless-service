FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# System packages
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ffmpeg \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python deps
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt


# Copy handler
COPY runpod_handler.py .

# Start RunPod serverless
CMD ["python3", "-u", "runpod_handler.py"]