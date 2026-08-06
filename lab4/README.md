# Boston Housing FastAPI Serving API (lab4)

This lab serves the Boston Housing model using FastAPI and Docker.

## How to run locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the app:
   ```bash
   uvicorn app:app --reload
   ```

## How to run with Docker
1. Build the Docker image:
   ```bash
   docker build -t boston-housing-app .
   ```
2. Run the Docker container:
   ```bash
   docker run -d -p 8000:8000 boston-housing-app
   ```
