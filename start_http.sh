#!/bin/bash
# Start Factory STT/TTS Server with HTTP (no SSL)
# Note: Browsers require HTTPS for microphone access in production.
# For local development, you can use HTTP, but you'll need to use a tool like ngrok
# or access via localhost (some browsers allow microphone on localhost without HTTPS)

cd "$(dirname "$0")"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "🚀 Starting Factory STT/TTS Server with HTTP..."
echo ""
echo "⚠️  Note: Most browsers require HTTPS for microphone access."
echo "   For localhost, some browsers may allow HTTP."
echo "   For remote access, use HTTPS or a tunnel like ngrok."
echo ""
python3 run_server.py --host 0.0.0.0 --port 8000
