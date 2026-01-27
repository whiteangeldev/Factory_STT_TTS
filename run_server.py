#!/usr/bin/env python3
"""Server entry point"""
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.server import app, socketio

def main():
    parser = argparse.ArgumentParser(description='Factory STT/TTS Server')
    parser.add_argument('--https', action='store_true', help='Enable HTTPS')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    args = parser.parse_args()
    
    if args.https:
        certfile = os.path.join(os.path.dirname(__file__), 'certs/cert.pem')
        keyfile = os.path.join(os.path.dirname(__file__), 'certs/key.pem')
        
        if os.path.exists(certfile) and os.path.exists(keyfile):
            print(f"Starting server on https://{args.host}:{args.port}")
            print(f"Access at: https://51.15.25.197:{args.port}")
            print("\n⚠️  Chrome will show a security warning for self-signed certificate.")
            print("   Click 'Advanced' → 'Proceed to 51.15.25.197 (unsafe)'")
            socketio.run(app, host=args.host, port=args.port, certfile=certfile, keyfile=keyfile)
        else:
            print(f"SSL certificates not found at {certfile} and {keyfile}")
            print("Running without HTTPS...")
            socketio.run(app, host=args.host, port=args.port)
    else:
        print(f"Starting server on http://{args.host}:{args.port}")
        socketio.run(app, host=args.host, port=args.port)

if __name__ == '__main__':
    main()
