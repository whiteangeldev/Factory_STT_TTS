#!/bin/bash
# Quick start script for HTTPS server

echo "🚀 Starting Factory STT/TTS Server with HTTPS..."
echo ""

# Check if certificates exist
if [ ! -f "certs/cert.pem" ] || [ ! -f "certs/key.pem" ]; then
    echo "⚠️  SSL certificates not found. Generating..."
    bash generate_ssl_cert.sh
    echo ""
fi

# Start server with HTTPS
echo "Starting server on https://0.0.0.0:8000"
echo "Access at: https://51.15.25.197:8000"
echo ""
echo "⚠️  Chrome will show a security warning for self-signed certificate."
echo "   Click 'Advanced' → 'Proceed to 51.15.25.197 (unsafe)'"
echo ""

python run_server.py --https
