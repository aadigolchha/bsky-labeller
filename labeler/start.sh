#!/bin/bash
set -e

echo "============================================"
echo "🚀 BLUESKY MISINFORMATION LABELER STARTUP"
echo "============================================"

mkdir -p /data/cache

echo "🐍 Starting Python Analyzer..."
python3 -m uvicorn logic.analyzer:app \
    --uds /tmp/analyzer.sock \
    --workers 1 \
    --no-access-log &

PYTHON_PID=$!

echo "⏳ Waiting for analyzer socket..."

# Wait for socket file to exist (just the socket, not full init)
for i in {1..60}; do
    if ! kill -0 $PYTHON_PID 2>/dev/null; then
        echo "❌ Python analyzer died during socket creation!"
        exit 1
    fi
    
    if [ -S /tmp/analyzer.sock ]; then
        echo "✅ Socket created after ${i}s"
        break
    fi
    
    sleep 1
done

if [ ! -S /tmp/analyzer.sock ]; then
    echo "❌ Socket never created after 60s"
    exit 1
fi

# Start Node immediately - it will handle requests while Python initializes
# Python's /analyze endpoint returns {label: null} during initialization
echo "🤖 Starting Node.js Server (Python still initializing in background)..."
exec npm start