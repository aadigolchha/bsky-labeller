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

echo "⏳ Waiting for analyzer to initialize..."

for i in {1..300}; do
    if ! kill -0 $PYTHON_PID 2>/dev/null; then
        echo "❌ Python analyzer died!"
        exit 1
    fi
    
    if [ -S /tmp/analyzer.sock ]; then
        echo "✅ Socket ready after ${i}s"
        break
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "   Still initializing... (${i}s elapsed)"
    fi
    
    sleep 1
done

echo "🤖 Starting Node.js Server..."
exec npm start
