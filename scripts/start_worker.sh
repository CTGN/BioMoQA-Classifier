#!/bin/bash
# Start Celery worker only (uv environment)

set -e

echo "🔧 Starting Celery worker..."

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first:"
    echo "   redis-server"
    exit 1
fi

# Start worker
uv run celery -A api.celery_app worker \
    --loglevel=info \
    --concurrency=1 \
    --pool=solo \
    --max-tasks-per-child=50

echo "✅ Worker started"
