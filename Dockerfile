FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MODEL_VERSION=1.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py app.py ./

# Train the model while building the Docker image
RUN python train.py

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
Create .dockerignore:

text

.git
.venv
__pycache__
*.pyc
model.pkl
