#!/bin/bash
# Script to run Fish Speech API server and test

set -e

echo "=========================================="
echo "Fish Speech v1.5 Test Runner"
echo "=========================================="
echo ""

# Check if fish-speech directory exists
if [ ! -d "fish-speech" ]; then
    echo "❌ fish-speech directory not found!"
    echo "Please run setup_fish_speech.sh first"
    exit 1
fi

# Check if models are downloaded
if [ ! -d "fish-speech/checkpoints/openaudio-s1-mini" ]; then
    echo "❌ Models not found!"
    echo "Please run setup_fish_speech.sh first"
    exit 1
fi

cd fish-speech

echo "Starting Fish Speech API server..."
echo "Server will run in the background"
echo "Access API at: http://127.0.0.1:8080"
echo ""

# Start API server in background
python -m tools.api_server \
    --listen 0.0.0.0:8080 \
    --llama-checkpoint-path "checkpoints/openaudio-s1-mini" \
    --decoder-checkpoint-path "checkpoints/openaudio-s1-mini/codec.pth" \
    --decoder-config-name modded_dac_vq \
    --compile \
    > ../fish_speech_server.log 2>&1 &

SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
echo ""

# Wait for server to start
echo "Waiting for server to start..."
sleep 10

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start. Check fish_speech_server.log for details"
    exit 1
fi

# Test health endpoint
if curl -s http://127.0.0.1:8080/v1/health > /dev/null; then
    echo "✓ Server is running!"
else
    echo "⚠ Server may not be ready yet. Waiting a bit more..."
    sleep 5
fi

echo ""
echo "Setting environment variable..."
export FISH_SPEECH_API_URL="http://127.0.0.1:8080"
echo "FISH_SPEECH_API_URL=$FISH_SPEECH_API_URL"
echo ""

cd ..

echo "Running test script..."
echo "-----------------------------------"
python test_fish_speech.py

echo ""
echo "Test completed!"
echo ""
echo "To stop the server, run:"
echo "  kill $SERVER_PID"
echo ""
