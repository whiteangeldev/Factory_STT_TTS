#!/usr/bin/env python3
"""Generate self-signed SSL certificates for HTTPS"""
import os
import subprocess
import sys

def generate_certs():
    """Generate self-signed SSL certificates"""
    certs_dir = os.path.join(os.path.dirname(__file__), 'certs')
    certfile = os.path.join(certs_dir, 'cert.pem')
    keyfile = os.path.join(certs_dir, 'key.pem')
    
    # Create certs directory if it doesn't exist
    os.makedirs(certs_dir, exist_ok=True)
    
    # Check if certificates already exist
    if os.path.exists(certfile) and os.path.exists(keyfile):
        print("✅ SSL certificates already exist")
        print(f"   Certificate: {certfile}")
        print(f"   Private key: {keyfile}")
        response = input("\nRegenerate? (y/N): ").strip().lower()
        if response != 'y':
            print("Keeping existing certificates")
            return
    
    print("🔐 Generating self-signed SSL certificates...")
    print("   (This may take a few seconds)")
    
    try:
        # Generate certificate and key using openssl
        subprocess.run([
            'openssl', 'req', '-x509', '-newkey', 'rsa:4096',
            '-keyout', keyfile,
            '-out', certfile,
            '-days', '365',
            '-nodes',
            '-subj', '/C=US/ST=State/L=City/O=Factory/CN=localhost'
        ], check=True, capture_output=True)
        
        print(f"\n✅ Certificates generated successfully!")
        print(f"   Certificate: {certfile}")
        print(f"   Private key: {keyfile}")
        print(f"\n⚠️  These are self-signed certificates for development only.")
        print(f"   Your browser will show a security warning - this is normal.")
        print(f"   Click 'Advanced' → 'Proceed to localhost (unsafe)' to continue.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error generating certificates: {e}")
        print("\nMake sure OpenSSL is installed:")
        print("  macOS: brew install openssl")
        print("  Linux: apt-get install openssl (or yum install openssl)")
        print("  Windows: Download from https://slproweb.com/products/Win32OpenSSL.html")
        sys.exit(1)
    except FileNotFoundError:
        print("\n❌ OpenSSL not found!")
        print("\nInstall OpenSSL:")
        print("  macOS: brew install openssl")
        print("  Linux: apt-get install openssl (or yum install openssl)")
        print("  Windows: Download from https://slproweb.com/products/Win32OpenSSL.html")
        sys.exit(1)

if __name__ == '__main__':
    generate_certs()
