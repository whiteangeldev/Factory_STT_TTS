#!/bin/bash
# Install TTS dependencies for Factory STT/TTS

echo "🔊 Installing TTS dependencies for Factory STT/TTS..."
echo ""

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not detected."
    echo "   It's recommended to activate your virtual environment first:"
    echo "   source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Choose TTS backend to install:"
echo "1) Chinese/Japanese TTS (gTTS) - Lightweight, recommended"
echo "2) English TTS (MMS-TTS) - Requires PyTorch, larger download"
echo "3) Both"
echo ""
read -p "Enter choice (1/2/3): " choice

case $choice in
    1)
        echo ""
        echo "📦 Installing gTTS and pydub for Chinese/Japanese TTS..."
        pip install gtts pydub
        echo "✅ Chinese/Japanese TTS dependencies installed!"
        ;;
    2)
        echo ""
        echo "📦 Installing PyTorch, transformers, and datasets for English TTS..."
        echo "   This may take a while and requires significant disk space..."
        pip install torch transformers datasets soundfile
        echo "✅ English TTS dependencies installed!"
        ;;
    3)
        echo ""
        echo "📦 Installing all TTS dependencies..."
        echo "   Installing gTTS for Chinese/Japanese..."
        pip install gtts pydub
        echo "   Installing PyTorch and transformers for English..."
        pip install torch transformers datasets soundfile
        echo "✅ All TTS dependencies installed!"
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "🎉 TTS installation complete!"
echo ""
echo "Optional: Install librosa for speed adjustment:"
echo "  pip install librosa"
echo ""
