# Factory STT/TTS

**Real-time Speech Recognition & Text-to-Speech System**

Transform speech to text and text to speech in real-time with support for **English**, **Chinese**, and **Japanese**. Perfect for factory environments, multilingual communication, and voice-enabled applications.

---

## 🚀 What Can It Do?

### Speech-to-Text (STT)
- 🎤 **Real-time transcription** as you speak
- 🌍 **Auto-detects language** (English, Chinese, Japanese)
- 🔇 **Works offline** after initial setup
- 🎯 **High accuracy** with confidence scores
- 🏭 **Noise reduction** optimized for factory environments

### Text-to-Speech (TTS)
- 🔊 **Natural-sounding voices** for 3 languages
- 🔄 **Automatic language detection** from text
- 🌐 **Mixed-language support** (English words in Chinese/Japanese text)
- ⚡ **Adjustable speed** (0.5x - 2.0x)
- 💾 **Save audio** as WAV files
- 🔇 **Fully offline** after setup

---

## 📋 Prerequisites

- **Python 3.11**
- **Modern web browser** (Chrome, Edge, or Firefox)
- **Internet connection** (only for initial setup)

---

## ⚡ Quick Installation

### Step 1: Clone and Setup

```bash
# Navigate to project directory
cd Factory_STT_TTS

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Mac/Linux
# OR
.venv\Scripts\activate      # Windows

# Install core dependencies
pip install -r requirements.txt
```

### Step 2: Install TTS Engines (Optional)

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

> **💡 Tip:** You only need to install the TTS engines for languages you plan to use. STT works without any TTS setup.

### GPU Acceleration (NVIDIA, Optional but Recommended)

If your PC has an NVIDIA GPU and you want faster TTS/STT:

```bash
# Activate venv first
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Mac/Linux

# Switch runtime variants (safe to run even if not present)
pip uninstall -y onnxruntime onnxruntime-gpu torch torchvision torchaudio

# Install CUDA-enabled PyTorch wheels (example: CUDA 12.4)
pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# Install ONNX Runtime GPU provider (required by PyKokoro GPU path)
pip install onnxruntime-gpu
```

Verify GPU availability:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
```

Expected:
- `torch.cuda.is_available()` is `True`
- ONNX providers include `CUDAExecutionProvider`

### Step 3: Run the Server

```bash
python3 run_server.py
```

**Access the application:**
- Open your browser and go to: `http://localhost:5421`
- For HTTPS (if certificates exist): `https://localhost:5421`

**Generate SSL certificates (optional):**
```bash
python3 generate_certs.py
```

---

## 🎯 How to Use

### Using Speech-to-Text

1. **Choose Input Source**
   - 🎤 **Microphone**: Click microphone icon (requires browser permission)
   - 🔊 **System Audio**: Click system audio icon (captures computer sound)

2. **Start Recording**
   - Click **"Start Recording"** button
   - Speak naturally - you'll see transcriptions appear in real-time

3. **View Results**
   - **Interim results** appear as you speak (marked with ⟳)
   - **Final results** appear when you stop (marked with ✓)
   - Language is automatically detected and shown with flags

4. **Stop Recording**
   - Click **"Stop Recording"** when finished

### Using Text-to-Speech

1. **Enter Your Text**
   - Type or paste text in the text area
   - Language is automatically detected

2. **Adjust Settings** (Optional)
   - Use the speed slider to control playback speed (0.5x - 2.0x)

3. **Play Audio**
   - Click **"Play"** to synthesize and hear the text
   - Audio plays immediately after synthesis

4. **Save Audio** (Optional)
   - Click **"Save"** button to download the audio as WAV file
   - Button is enabled after all text is converted

5. **Stop Playback**
   - Click **"Stop"** to stop audio at any time

---

## 🔧 System Audio Setup (Optional)

If you want to capture system audio (e.g., from video calls, music, etc.):

### macOS
1. Install BlackHole: `brew install blackhole-2ch`
2. Open **Audio MIDI Setup** (Applications > Utilities)
3. Click **+** → **Create Multi-Output Device**
4. Add your speakers and **BlackHole 2ch**
5. Select the Multi-Output Device as your system output

### Windows
1. Right-click sound icon → **Sounds**
2. Go to **Recording** tab
3. Right-click empty space → **Show Disabled Devices**
4. Enable **Stereo Mix**
5. Set it as default recording device

### Linux
```bash
pactl load-module module-loopback
```

---

## 🛠️ Troubleshooting

### TTS Not Working?

**English TTS Issues:**
```bash
pip install pykokoro spacy
python -m spacy download en_core_web_sm
```

**Chinese TTS Issues:**
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

**Japanese TTS Issues:**
```bash
git clone https://github.com/myshell-ai/MeloTTS.git
cd MeloTTS
pip install -e .
python -m unidic download
cd ..
```

**After installing, restart the server!**

### STT Not Working?

- **No transcriptions appearing?**
  - Install Whisper: `pip install openai-whisper`
  - Model downloads automatically on first use (needs internet)
  - Check server logs for errors

- **Slow transcription?**
  - This is normal for high-accuracy models
  - The system uses optimized settings for best balance

### Common Issues

- **No audio playback?**
  - Check browser console for errors
  - Verify browser audio permissions
  - Try Chrome or Edge browser

- **English words not spoken in Chinese/Japanese text?**
  - Install PyKokoro: `pip install pykokoro spacy`
  - Download spaCy model: `python -m spacy download en_core_web_sm`

- **Server won't start?**
  - Check Python version: `python3 --version` (needs 3.8+)
  - Verify dependencies: `pip list`
  - Check server logs for specific errors

---

## 📚 Technical Information

### Supported Languages
- **English** (en)
- **Chinese** (zh, cmn, zho)
- **Japanese** (ja, jpn)

### TTS Engines
- **English**: PyKokoro-82M (neural TTS, offline)
- **Chinese**: MeloTTS (neural TTS, natural pauses, optimized speed)
- **Japanese**: MeloTTS (neural TTS, natural pauses)

### STT Engine
- **Whisper**: OpenAI Whisper (offline-capable, auto language detection)

### Features
- ✅ **Fully offline** after initial setup
- ✅ **Model caching** for fast performance
- ✅ **Mixed-language support** (English in CJK text)
- ✅ **Natural speech** with variable pauses
- ✅ **Real-time processing** with low latency

---

## 📖 Project Structure

```
Factory_STT_TTS/
├── backend/          # Server-side code
│   ├── audio/        # Audio processing (STT, TTS, VAD)
│   ├── config.py     # Configuration
│   └── server.py     # Flask server
├── frontend/         # Web interface
│   ├── index.html    # Main page
│   └── static/      # CSS and JavaScript
├── MeloTTS/         # MeloTTS engine (for Chinese/Japanese)
├── requirements.txt  # Python dependencies
└── run_server.py    # Server entry point
```

---

## 🔒 Security Note

The server uses self-signed SSL certificates by default. Your browser will show a security warning - this is normal for local development. Click "Advanced" → "Proceed to localhost" to continue.

---

## 💡 Tips

- **First time setup?** Start with English TTS only - it's the simplest to set up
- **Need all languages?** Install TTS engines one at a time to avoid confusion
- **Offline use?** All models download automatically on first use, then work offline
- **Performance?** Models are cached in memory for fast subsequent requests

---

## 📞 Getting Help

1. **Check server logs** - Most errors are logged with helpful messages
2. **Verify installation** - Make sure all dependencies are installed
3. **Check browser console** - Frontend errors appear in browser DevTools
4. **Restart server** - After installing new dependencies, always restart

---

## 🎉 You're Ready!

Once installed, you can:
- ✅ Transcribe speech in real-time
- ✅ Convert text to natural-sounding speech
- ✅ Work with multiple languages seamlessly
- ✅ Use everything offline after initial setup

**Start the server and open `http://localhost:5421` to begin!**
