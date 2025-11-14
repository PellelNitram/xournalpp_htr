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
    xournalpp \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Create and set working directory
WORKDIR /app

# Create temp_code_mount folder
RUN mkdir -p /temp_code_mount

# Install Python dependencies early for caching
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the INSTALL_HF_DOCKER_SPACE.sh script
RUN bash INSTALL_HF_DOCKER_SPACE.sh
RUN pip install matplotlib bs4 pdf2image supabase python-dotenv
# ^- that should not be necessary!! TODO!!

# Expose the port Gradio will run on inside Hugging Face Spaces
EXPOSE 7860

ENV PYTHONUNBUFFERED=1

# Command to run Gradio app
# Hugging Face Spaces will set PORT env var, so we use it
CMD ["python", "scripts/demo.py"]








# https://huggingface.co/docs/hub/spaces-sdks-docker

# https://huggingface.co/spaces/SpacesExamples/secret-example/tree/main
# - https://huggingface.co/spaces/SpacesExamples/secret-example/blob/main/Dockerfile