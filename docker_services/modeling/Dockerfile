FROM tensorflow/tensorflow:latest

# Set the working directory
WORKDIR /workspace

# Copy requirements.txt into the container
COPY requirements.txt /workspace

# Install system-level dependencies required for OpenCV and git
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --ignore-installed -r requirements.txt


