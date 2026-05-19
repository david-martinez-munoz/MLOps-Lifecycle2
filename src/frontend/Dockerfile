FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm

WORKDIR /app

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY src/frontend/requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src /app/src

EXPOSE 8501

CMD ["streamlit", "run", "src/frontend/ui.py", "--server.address=0.0.0.0", "--server.port=8501"]