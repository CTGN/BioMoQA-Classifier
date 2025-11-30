#!/bin/bash
# Start Streamlit UI only (uv environment)

set -e

echo "🎨 Starting Streamlit UI..."

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first:"
    echo "   redis-server"
    exit 1
fi

# Start UI with increased upload limit
uv run streamlit run web/app_redis.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.maxUploadSize=1500

echo "✅ Streamlit started on http://localhost:8501"
