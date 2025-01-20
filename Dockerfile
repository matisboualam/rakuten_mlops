FROM tensorflow/tensorflow:latest

WORKDIR /workspace

COPY requirements.txt /workspace

EXPOSE 8000

RUN pip install -r requirements.txt