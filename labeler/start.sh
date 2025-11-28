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

READY=false
for i in {1..300}; do
    if ! kill -0 $PYTHON_PID 2>/dev/null; then
        echo "❌ Python analyzer died!"
        exit 1
    fi

    # Check if analyzer is actually ready (not just socket exists)
    if [ -S /tmp/analyzer.sock ]; then
        HEALTH=$(curl -s --unix-socket /tmp/analyzer.sock http://localhost/health 2>/dev/null || echo '{}')
        if echo "$HEALTH" | grep -q '"initialized": true'; then
            echo "✅ Analyzer ready after ${i}s"
            READY=true
            break
        fi
    fi

    if [ $((i % 10)) -eq 0 ]; then
        echo "   Still initializing... (${i}s elapsed)"
    fi

    sleep 1
done

if [ "$READY" = false ]; then
    echo "❌ Analyzer failed to initialize within 300s"
    exit 1
fi

echo "🤖 Starting Node.js Server..."
exec npm start
