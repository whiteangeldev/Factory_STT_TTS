#!/bin/bash
# Setup script for Fish Speech v1.5

set -e

echo "=========================================="
echo "Fish Speech v1.5 Setup Script"
echo "=========================================="
echo ""

# Check if fish-speech directory exists
if [ ! -d "fish-speech" ]; then
    echo "❌ fish-speech directory not found!"
    echo "Please clone it first:"
    echo "  git clone https://github.com/fishaudio/fish-speech.git"
    exit 1
fi

cd fish-speech

echo "Step 1: Installing dependencies..."
echo "-----------------------------------"

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv to install dependencies..."
    uv pip install -e .
else
    echo "Using pip to install dependencies..."
    pip install -e .
fi

echo ""
echo "Step 2: Downloading models..."
echo "-----------------------------------"

# Check if huggingface_hub is installed
if ! python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface_hub..."
    pip install "huggingface_hub[cli]"
fi

# Create checkpoints directory
mkdir -p checkpoints

# Download model
echo "Downloading OpenAudio S1-mini model..."
echo ""
echo "⚠️  Note: This model requires Hugging Face authentication."
echo "   1. Create a Hugging Face account at https://huggingface.co/join"
echo "   2. Request access to: https://huggingface.co/fishaudio/openaudio-s1-mini"
echo "   3. Log in: huggingface-cli login"
echo "   OR set token: export HF_TOKEN=your_token_here"
echo ""

# Check if user is logged in
if ! huggingface-cli whoami &>/dev/null; then
    echo "❌ Not logged in to Hugging Face"
    echo ""
    echo "Please log in first:"
    echo "  huggingface-cli login"
    echo ""
    echo "Or set your token:"
    echo "  export HF_TOKEN=your_token_here"
    echo ""
    echo "Then request access to the model:"
    echo "  https://huggingface.co/fishaudio/openaudio-s1-mini"
    echo ""
    read -p "Press Enter after logging in and requesting access..."
fi

# Download with authentication
python -c "
from huggingface_hub import snapshot_download
import os
os.makedirs('checkpoints/openaudio-s1-mini', exist_ok=True)
try:
    snapshot_download(
        repo_id='fishaudio/openaudio-s1-mini',
        local_dir='checkpoints/openaudio-s1-mini',
    )
    print('✓ Model downloaded successfully!')
except Exception as e:
    print(f'❌ Download failed: {e}')
    print('')
    print('Please ensure:')
    print('  1. You are logged in: huggingface-cli login')
    print('  2. You have requested access: https://huggingface.co/fishaudio/openaudio-s1-mini')
    print('  3. Your access has been approved')
    exit(1)
"

echo ""
echo "Step 3: Setup complete!"
echo "-----------------------------------"
echo ""
echo "Next steps:"
echo "1. Start the Fish Speech API server:"
echo "   cd fish-speech"
echo "   python -m tools.api_server \\"
echo "       --listen 0.0.0.0:8080 \\"
echo "       --llama-checkpoint-path \"checkpoints/openaudio-s1-mini\" \\"
echo "       --decoder-checkpoint-path \"checkpoints/openaudio-s1-mini/codec.pth\" \\"
echo "       --decoder-config-name modded_dac_vq"
echo ""
echo "2. In another terminal, set the API URL:"
echo "   export FISH_SPEECH_API_URL=\"http://127.0.0.1:8080\""
echo ""
echo "3. Run the test script:"
echo "   cd .."
echo "   python test_fish_speech.py"
echo ""
