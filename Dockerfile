# Documentation: https://huggingface.co/docs/hub/spaces-sdks-docker

# Start from an official lightweight Python image
FROM python:3.10-slim

# Prevents Python from writing .pyc files and buffering stdout/stderr
# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    unzip \
    vim-tiny \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Install Python dependencies early for caching
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port Gradio will run on inside Hugging Face Spaces
# EXPOSE 7860

# Command to run Gradio app
# Hugging Face Spaces will set PORT env var, so we use it
# CMD ["python", "app.py"]








# https://huggingface.co/docs/hub/spaces-sdks-docker

# https://huggingface.co/spaces/SpacesExamples/secret-example/tree/main
# - https://huggingface.co/spaces/SpacesExamples/secret-example/blob/main/Dockerfile