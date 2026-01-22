#!/bin/bash
# Generate self-signed SSL certificate for HTTPS

echo "Generating SSL certificate for HTTPS..."

# Create certs directory if it doesn't exist
mkdir -p certs

# Generate private key
openssl genrsa -out certs/key.pem 2048

# Generate certificate (valid for 365 days)
openssl req -new -x509 -key certs/key.pem -out certs/cert.pem -days 365 -subj "/CN=51.15.25.197"

echo "✅ SSL certificate generated successfully!"
echo "Certificate: certs/cert.pem"
echo "Private Key: certs/key.pem"
echo ""
echo "⚠️  Note: This is a self-signed certificate. Chrome will show a security warning."
echo "   Click 'Advanced' → 'Proceed to 51.15.25.197 (unsafe)' to continue."
