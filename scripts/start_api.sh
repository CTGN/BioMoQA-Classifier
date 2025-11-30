#!/bin/bash
# Start FastAPI server only (uv environment)

set -e

echo "🌐 Starting FastAPI server..."

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first:"
    echo "   redis-server"
    exit 1
fi

# Start API
uv run uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload

echo "✅ API server started on http://localhost:8000"
echo "📚 API docs available at http://localhost:8000/docs"
