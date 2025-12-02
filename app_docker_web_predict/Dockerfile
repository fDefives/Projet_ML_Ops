FROM python:3.11-slim
WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch and torchvision from the official PyTorch wheel index to
# avoid dependency resolution conflicts inside pip resolver. If you want GPU support,
# replace this with a CUDA base image and install the matching CUDA wheels.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch" "torchvision"

COPY . /app
EXPOSE 5000
CMD ["python", "app.py"]
