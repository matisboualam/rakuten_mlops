FROM tensorflow/tensorflow:latest

# Set the working directory
WORKDIR /workspace

# Copy requirements.txt into the container
COPY requirements.txt /workspace

# Upgrade pip before installing packages
RUN pip install --upgrade pip setuptools

# Force reinstall problematic packages
RUN pip install --no-cache-dir --ignore-installed blinker Werkzeug

# Install all Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
