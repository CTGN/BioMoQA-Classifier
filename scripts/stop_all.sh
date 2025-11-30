#!/bin/bash
# Stop all BioMoQA services

set -e

echo "🛑 Stopping BioMoQA Services..."

# Stop Celery workers
echo "Stopping Celery workers..."
uv run celery -A api.celery_app control shutdown 2>/dev/null || echo "No workers running"

# Stop API (find and kill uvicorn process)
echo "Stopping API server..."
pkill -f "uvicorn api.main:app" 2>/dev/null || echo "No API server running"

# Stop Streamlit
echo "Stopping Streamlit..."
pkill -f "streamlit run web/app_redis.py" 2>/dev/null || echo "No Streamlit running"

echo "✅ All services stopped"
