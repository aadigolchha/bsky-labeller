#!/bin/bash
set -e

echo "🚀 Starting Labeler System..."

# 1. Start Python (FastAPI) in background
# We use --uds to create the unix socket
# --forwarded-allow-ips='*' is needed for some internal proxies, but UDS ignores it safely.
echo "🐍 Starting Python Logic..."
python3 -m uvicorn python.analyzer:app --uds /tmp/analyzer.sock --workers 2 &

# 2. Wait for Python to wake up
echo "⏳ Waiting for Python Socket..."
while [ ! -S /tmp/analyzer.sock ]; do
  sleep 0.1
done
echo "✅ Python is listening on /tmp/analyzer.sock"

# 3. Start Node.js
echo "🤖 Starting Node.js Server..."
# We use 'exec' so Node takes over PID 1 (good for signal handling)
exec npm start