#!/bin/bash
set -e

echo "============================================"
echo "🚀 BLUESKY MISINFORMATION LABELER STARTUP"
echo "============================================"
echo ""

# Create data directory if it doesn't exist
mkdir -p /data/cache

echo "🐍 Starting Python Analyzer..."
echo "   - Loading Community Notes data"
echo "   - Generating embeddings (this may take a few minutes on first run)"
echo "   - Clustering misinformation themes"
echo "   - Generating dynamic labels"
echo ""

# Start Python analyzer in background
python3 -m uvicorn logic.analyzer:app \
    --uds /tmp/analyzer.sock \
    --workers 1 \
    --no-access-log \
    --log-level info &

PYTHON_PID=$!

# Trap signals for graceful shutdown
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    if kill -0 $PYTHON_PID 2>/dev/null; then
        echo "   Stopping Python analyzer (PID: $PYTHON_PID)"
        kill $PYTHON_PID 2>/dev/null
        wait $PYTHON_PID 2>/dev/null
    fi
    exit 0
}
trap cleanup EXIT SIGTERM SIGINT

echo "⏳ Waiting for analyzer to initialize..."

# Wait for socket with extended timeout (embedding generation can be slow)
MAX_WAIT=300  # 5 minutes max for initialization
WAITED=0

while [ $WAITED -lt $MAX_WAIT ]; do
    # Check if Python process is still running
    if ! kill -0 $PYTHON_PID 2>/dev/null; then
        echo "❌ Python analyzer process died!"
        echo "   Check the logs above for errors."
        exit 1
    fi
    
    # Check if socket exists
    if [ -S /tmp/analyzer.sock ]; then
        echo "✅ Socket created after ${WAITED}s"
        
        # Give it a moment to be ready for connections
        sleep 2
        
        # Try health check - must be initialized (not just responding)
        HEALTH_RESPONSE=$(curl --silent --unix-socket /tmp/analyzer.sock http://localhost/health 2>/dev/null)
        if echo "$HEALTH_RESPONSE" | grep -q '"initialized":true'; then
            echo "✅ Analyzer is fully initialized and ready"
            break
        elif echo "$HEALTH_RESPONSE" | grep -q '"initialized":false'; then
            # Socket is up but still initializing
            if [ $((WAITED % 10)) -eq 0 ] && [ $WAITED -gt 0 ]; then
                echo "   Analyzer responding but still initializing... (${WAITED}s elapsed)"
            fi
        fi
    fi
    
    # Progress indicator every 10 seconds
    if [ $((WAITED % 10)) -eq 0 ] && [ $WAITED -gt 0 ]; then
        echo "   Still initializing... (${WAITED}s elapsed)"
    fi
    
    sleep 1
    WAITED=$((WAITED + 1))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "❌ Timeout waiting for analyzer!"
    echo "   The analyzer may still be initializing embeddings."
    echo "   Check if Community Notes file exists and OPENAI_API_KEY is set."
    # Don't exit - Node.js will handle the error gracefully
fi

# Show analyzer status
echo ""
echo "📊 Analyzer Status:"
curl --silent --unix-socket /tmp/analyzer.sock http://localhost/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "   (Unable to fetch status)"
echo ""

echo "============================================"
echo "🤖 Starting Node.js Labeler Server..."
echo "============================================"
echo ""

# Replace this process with Node.js
exec npm start
