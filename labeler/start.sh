#!/bin/bash
set -e

echo "🐍 Starting Python Logic..."
# Start Python in background
python3 -m uvicorn logic.analyzer:app --uds /tmp/analyzer.sock --workers 1 &
PYTHON_PID=$!

echo "⏳ Waiting for Python Socket..."
# Loop up to 30 times (3 seconds)
for i in {1..30}; do
  if [ -S /tmp/analyzer.sock ]; then
    echo "✅ Socket found!"
    break
  fi
  
  # Check if Python is still running
  if ! kill -0 $PYTHON_PID 2>/dev/null; then
    echo "❌ Python process died! Check logs above."
    exit 1
  fi
  
  sleep 0.1
done

echo "🤖 Starting Node.js Server..."
exec npm start