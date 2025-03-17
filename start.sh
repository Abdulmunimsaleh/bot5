#!/bin/bash
# Install Playwright browsers
python -m playwright install chromium --with-deps
# Start the FastAPI app
uvicorn main:app --host 0.0.0.0 --port $PORT
