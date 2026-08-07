FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces runs on port 7860. We expose it for documentation.
EXPOSE 7860

# Run uvicorn on the port environment variable set by Hugging Face (defaulting to 7860 if not set)
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
