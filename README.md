# Factory STT/TTS - Real-time Speech Recognition and Text-to-Speech

A comprehensive real-time speech-to-text and text-to-speech system with multi-language support (English, Chinese, Japanese), designed for factory and industrial environments.

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup TTS (Optional - for Text-to-Speech functionality)

**For English TTS:**
```bash
pip install pykokoro spacy
python -m spacy download en_core_web_sm
```

**For Chinese TTS:**
```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
cd ..
pip install nltk
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

**For Japanese TTS:**
```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
python -m unidic download
cd ..
```

### 3. Run Server

```bash
python3 run_server.py
```

**Access the application:**
- HTTPS: `https://localhost:5421` (if SSL certificates exist)
- HTTP: `http://localhost:5421` (default)

**To generate SSL certificates (optional):**
```bash
python3 generate_certs.py
```

## Features

### Speech-to-Text (STT)
- ✅ **Real-time Transcription**: See transcriptions as you speak
- ✅ **Multi-language Support**: Automatically detects English, Chinese, and Japanese
- ✅ **Offline Operation**: Works completely offline after initial setup
- ✅ **Dual Input Modes**: 
  - Microphone mode (browser-based)
  - System audio mode (captures system sound directly)
- ✅ **Noise Reduction**: Optimized for factory environments
- ✅ **Visual Feedback**: Language badges, confidence scores, and latency metrics

### Text-to-Speech (TTS)
- ✅ **Multi-language Support**: English, Chinese, and Japanese
- ✅ **Automatic Language Detection**: No manual language selection needed
- ✅ **Mixed-language Support**: English words in Chinese/Japanese text are automatically handled
- ✅ **Natural Speech**: Variable pauses, optimized speed, smooth transitions
- ✅ **Adjustable Speed**: Control playback speed from 0.5x to 2.0x
- ✅ **Save Audio**: Download synthesized audio as WAV file
- ✅ **Offline Operation**: Works completely offline after initial setup

## Usage Guide

### Speech-to-Text

1. **Select Input Mode**:
   - **Microphone**: Click microphone icon to use your microphone
   - **System Audio**: Click system audio icon to capture system sound

2. **Start Recording**: Click **"Start Recording"** button

3. **View Results**:
   - Real-time transcriptions appear as you speak
   - Final transcriptions appear when speech ends
   - Language is automatically detected and displayed

4. **Stop Recording**: Click **"Stop Recording"** when done

### Text-to-Speech

1. **Enter Text**: Type or paste text in the text area
   - Language is automatically detected (English, Chinese, or Japanese)

2. **Adjust Speed** (optional): Use the speed slider (0.5x - 2.0x)

3. **Play**: Click **"Play"** button to synthesize and play audio

4. **Stop**: Click **"Stop"** to stop playback

5. **Save**: Click **"Save"** button to download audio as WAV file
   - Button is enabled after all text is synthesized

## System Requirements

- **Python 3.8 or higher**
- **Modern browser** (Chrome/Edge recommended)
- **Internet connection** (only for initial model download)

## System Audio Setup (Optional)

**macOS:**
1. Install BlackHole: `brew install blackhole-2ch`
2. Open Audio MIDI Setup
3. Create Multi-Output Device
4. Add your speakers and BlackHole 2ch
5. Select Multi-Output Device as system output

**Windows:**
1. Open Sound settings
2. Enable "Stereo Mix" in Recording devices
3. Set as default recording device

**Linux:**
```bash
pactl load-module module-loopback
```

## Troubleshooting

### TTS Not Working

**English TTS:**
```bash
pip install pykokoro spacy
python -m spacy download en_core_web_sm
```

**Chinese TTS:**
```bash
# Install MeloTTS
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
cd ..

# Install NLTK for mixed-language support
pip install nltk
python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

**Japanese TTS:**
```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
python -m unidic download
cd ..
```

**After installation, restart the server.**

### Common Issues

- **No audio playback**: Check browser console, verify audio permissions, try Chrome/Edge
- **English words not spoken in Chinese/Japanese text**: Install PyKokoro and spaCy (see English TTS setup above)
- **TTS errors**: Check server logs for specific error messages
- **Model download fails**: Ensure internet connection for initial download

### STT Not Working

- **No transcriptions**: 
  - Verify Whisper is installed: `pip install openai-whisper`
  - Model downloads automatically on first use (requires internet)
  - Check server logs for errors

- **Slow transcription**: 
  - This is normal for high-accuracy models
  - Smaller models are faster but less accurate

## Technical Details

### TTS Engines
- **English**: PyKokoro-82M (neural TTS, offline-capable)
- **Chinese**: MeloTTS (neural TTS with natural pauses and optimized speed)
- **Japanese**: MeloTTS (neural TTS with natural pauses)

### STT Engine
- **Whisper**: OpenAI Whisper (offline-capable, automatic language detection)

All engines work completely offline after initial model download.

## Support

For issues or questions, check the server logs for detailed error messages.
