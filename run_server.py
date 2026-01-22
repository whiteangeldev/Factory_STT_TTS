#!/usr/bin/env python3
"""Startup script for the Factory STT/TTS server"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.server import run_server

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Factory STT/TTS Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to (default: 8000)')
    parser.add_argument('--https', action='store_true', help='Enable HTTPS (requires --ssl-key and --ssl-cert)')
    parser.add_argument('--ssl-key', default='certs/key.pem', help='Path to SSL private key (default: certs/key.pem)')
    parser.add_argument('--ssl-cert', default='certs/cert.pem', help='Path to SSL certificate (default: certs/cert.pem)')
    
    args = parser.parse_args()
    
    if args.https:
        run_server(
            host=args.host, 
            port=args.port,
            use_https=True,
            ssl_keyfile=args.ssl_key,
            ssl_certfile=args.ssl_cert
        )
    else:
        run_server(host=args.host, port=args.port)
