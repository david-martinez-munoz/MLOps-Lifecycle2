FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY src/seeder/requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src /app/src

CMD ["python", "src/seeder/traffic_generator.py"]