FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

VOLUME /app/models

RUN python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
RUN python -c "import ultralytics; print(f'Ultralytics version: {ultralytics.__version__}')"

CMD ["python", "run.py"]
