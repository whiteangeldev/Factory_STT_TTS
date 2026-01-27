#!/usr/bin/env python3
"""Server entry point - auto-detects HTTPS if certificates exist"""
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.server import app, socketio

def main():
    parser = argparse.ArgumentParser(description='Factory STT/TTS Server')
    parser.add_argument('--https', action='store_true', help='Force HTTPS (requires certs)')
    parser.add_argument('--http', action='store_true', help='Force HTTP (disable HTTPS)')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    args = parser.parse_args()
    
    certfile = os.path.join(os.path.dirname(__file__), 'certs/cert.pem')
    keyfile = os.path.join(os.path.dirname(__file__), 'certs/key.pem')
    certs_exist = os.path.exists(certfile) and os.path.exists(keyfile)
    
    # Determine if we should use HTTPS
    use_https = False
    if args.http:
        use_https = False
    elif args.https:
        use_https = True
        if not certs_exist:
            print("❌ SSL certificates not found!")
            print(f"   Expected: {certfile}")
            print(f"   Expected: {keyfile}")
            print("\n💡 Generate certificates with:")
            print("   python3 generate_certs.py")
            sys.exit(1)
    else:
        # Auto-detect: use HTTPS if certs exist
        use_https = certs_exist
    
    if use_https:
        print(f"🔒 Starting HTTPS server on https://{args.host}:{args.port}")
        print(f"   Access at: https://localhost:{args.port}")
        print("\n⚠️  Self-signed certificate - browser will show security warning")
        print("   Click 'Advanced' → 'Proceed to localhost (unsafe)'")
        socketio.run(app, host=args.host, port=args.port, certfile=certfile, keyfile=keyfile)
    else:
        print(f"🌐 Starting HTTP server on http://{args.host}:{args.port}")
        print(f"   Access at: http://localhost:{args.port}")
        if not args.http and not certs_exist:
            print("\n💡 To enable HTTPS, generate certificates:")
            print("   python3 generate_certs.py")
        socketio.run(app, host=args.host, port=args.port)

if __name__ == '__main__':
    main()
