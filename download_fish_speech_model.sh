#!/bin/bash
# Script to download Fish Speech model with authentication

set -e

echo "=========================================="
echo "Fish Speech Model Download"
echo "=========================================="
echo ""

cd fish-speech

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub CLI..."
    pip install "huggingface_hub[cli]"
fi

# Check if user is logged in
echo "Checking Hugging Face authentication..."
if ! huggingface-cli whoami &>/dev/null; then
    echo ""
    echo "❌ Not logged in to Hugging Face"
    echo ""
    echo "Please log in first:"
    echo "  huggingface-cli login"
    echo ""
    echo "You'll need:"
    echo "  1. A Hugging Face account (create at https://huggingface.co/join)"
    echo "  2. Access to the model (request at https://huggingface.co/fishaudio/openaudio-s1-mini)"
    echo "  3. Your access token (get from https://huggingface.co/settings/tokens)"
    echo ""
    read -p "Press Enter after logging in and requesting access, or Ctrl+C to cancel..."
    
    # Try again
    if ! huggingface-cli whoami &>/dev/null; then
        echo "❌ Still not logged in. Please run: huggingface-cli login"
        exit 1
    fi
fi

USERNAME=$(huggingface-cli whoami)
echo "✓ Logged in as: $USERNAME"
echo ""

# Create checkpoints directory
mkdir -p checkpoints

# Download model
echo "Downloading OpenAudio S1-mini model..."
echo "This may take a few minutes..."
echo ""

python -c "
from huggingface_hub import snapshot_download
import os
import sys

os.makedirs('checkpoints/openaudio-s1-mini', exist_ok=True)

try:
    print('Starting download...')
    snapshot_download(
        repo_id='fishaudio/openaudio-s1-mini',
        local_dir='checkpoints/openaudio-s1-mini',
    )
    print('')
    print('✓ Model downloaded successfully!')
except Exception as e:
    error_msg = str(e)
    print('')
    print(f'❌ Download failed: {error_msg}')
    print('')
    if '401' in error_msg or 'Unauthorized' in error_msg or 'gated' in error_msg.lower():
        print('This model requires:')
        print('  1. Hugging Face account')
        print('  2. Access approval (request at: https://huggingface.co/fishaudio/openaudio-s1-mini)')
        print('  3. Authentication (run: huggingface-cli login)')
        print('')
        print('Steps:')
        print('  1. Visit: https://huggingface.co/fishaudio/openaudio-s1-mini')
        print('  2. Click \"Request access\" or \"Agree and access repository\"')
        print('  3. Wait for approval (usually instant)')
        print('  4. Run: huggingface-cli login')
        print('  5. Run this script again')
    else:
        print('Please check your internet connection and try again.')
    sys.exit(1)
"

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo ""
echo "Model location: fish-speech/checkpoints/openaudio-s1-mini"
echo ""
echo "Next steps:"
echo "  1. Start the API server (see QUICK_START_FISH_SPEECH.md)"
echo "  2. Run the test script"
