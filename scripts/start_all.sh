#!/bin/bash
# Start all services for local development (uv environment)

set -e

echo "🚀 Starting BioMoQA Services..."

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first:"
    echo "   redis-server"
    exit 1
fi

echo "✅ Redis is running"

# Create logs directory
mkdir -p logs

# Start Celery worker in background
echo "🔧 Starting Celery worker..."
uv run celery -A api.celery_app worker \
    --loglevel=info \
    --concurrency=1 \
    --pool=solo \
    --logfile=logs/worker.log \
    --detach

# Wait a bit for worker to start
sleep 3

# Start FastAPI in background
echo "🌐 Starting FastAPI server..."
uv run uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    > logs/api.log 2>&1 &

API_PID=$!
echo "API PID: $API_PID"

# Wait a bit for API to start
sleep 3

# Start Streamlit in foreground
echo "🎨 Starting Streamlit UI..."
echo ""
echo "Services started:"
echo "  - Redis: localhost:6379"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - UI: http://localhost:8501"
echo ""
echo "Logs:"
echo "  - Worker: logs/worker.log"
echo "  - API: logs/api.log"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

uv run streamlit run web/app_redis.py --server.maxUploadSize=1500

# Cleanup on exit
echo ""
echo "🛑 Stopping services..."
kill $API_PID 2>/dev/null || true
uv run celery -A api.celery_app control shutdown 2>/dev/null || true
echo "✅ All services stopped"
