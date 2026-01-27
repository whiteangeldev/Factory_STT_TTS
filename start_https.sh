#!/bin/bash
# Start Factory STT/TTS Server with HTTPS

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 Starting Factory STT/TTS Server with HTTPS..."
echo ""
python3 run_server.py --https
