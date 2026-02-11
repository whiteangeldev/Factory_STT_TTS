# Factory STT/TTS - Real-time Speech Recognition and Text-to-Speech

A comprehensive real-time speech-to-text and text-to-speech system with multi-language support, designed for factory and industrial environments.

## Milestones Completed

### ✅ Milestone 1: Real-time Audio Capture
- Real-time audio capture with Voice Activity Detection (VAD)
- Dual input modes (Microphone and System Audio)
- Noise reduction using RNNoise
- Real-time audio level monitoring

### ✅ Milestone 2: Speech-to-Text (STT) Integration
- **Offline Whisper STT** - Fully offline speech recognition using OpenAI Whisper
- Real-time interim and final transcription display
- Automatic language detection (English, Chinese, Japanese)
- Language badge display in transcriptions
- Confidence scores (95-100% for final transcriptions) and latency metrics
- Optimized for speed and accuracy (beam_size=3, best_of=3, fp16 acceleration)
- Optimized for noisy factory conditions
- Works completely offline after model download

### ✅ Milestone 3: Language Handling & Text-to-Speech (TTS)
- Automatic language detection from text input
- Multi-language TTS support (all offline-capable after initial download):
  - **English**: PyKokoro-82M (neural TTS, offline-capable)
  - **Chinese**: PyKokoro (offline, neural TTS, faster than Piper, no g2pw overhead)
  - **Japanese**: PyKokoro (offline, neural TTS, natural-sounding)
- Adjustable playback speed (0.5x - 2.0x)
- Real-time audio playback in browser
- Model caching for fast subsequent requests

## Quick Start

### 1. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Server

```bash
python3 run_server.py
```

**Server URL:**
- HTTPS: `https://localhost:5421` (if SSL certificates exist in `certs/` directory)
- HTTP: `http://localhost:5421` (if no SSL certificates)

**Note:** The server automatically detects and uses HTTPS if certificates are present. To generate SSL certificates, run:
```bash
python3 generate_certs.py
```

## Features

### Audio Capture & Processing
- ✅ **Microphone Mode**: Record audio from your microphone
- ✅ **System Audio Mode**: Capture system audio directly on server (no browser permissions needed)
- ✅ **Voice Activity Detection (VAD)**: Real-time speech detection with visual indicators
- ✅ **Noise Reduction**: RNNoise-based noise suppression for factory environments
- ✅ **Audio Level Monitoring**: Real-time audio level display with RMS and peak levels

### Speech-to-Text (STT)
- ✅ **Offline Transcription**: Fully offline STT using OpenAI Whisper (no internet required)
- ✅ **Real-time Transcription**: Low-latency transcription with streaming support (optimized for speed)
- ✅ **Interim Results**: See transcription as you speak (real-time updates)
- ✅ **Final Results**: Accurate final transcriptions with confidence scores (95-100%)
- ✅ **Multi-language Support**: Automatic detection of English, Chinese, and Japanese
- ✅ **Language Badges**: Visual indicators showing detected language (🇬🇧 English, 🇨🇳 Chinese, 🇯🇵 Japanese)
- ✅ **Latency Metrics**: Track transcription latency for performance monitoring
- ✅ **Confidence Scores**: Display transcription confidence levels (optimized calculation for final results)

### Text-to-Speech (TTS)
- ✅ **Automatic Language Detection**: Detects language from input text automatically
- ✅ **Multi-language TTS**: Supports English, Chinese, and Japanese (all offline-capable)
- ✅ **Offline Operation**: All TTS engines work completely offline after initial model download
- ✅ **Adjustable Speed**: Control playback speed from 0.5x to 2.0x
- ✅ **Real-time Playback**: Instant audio playback in browser
- ✅ **High-quality Voices**: Natural-sounding neural speech synthesis
- ✅ **Model Caching**: Fast subsequent requests with in-memory model caching

## Usage

### Starting the Server

1. Start the server: `python3 run_server.py`
   - Server automatically uses HTTPS if SSL certificates exist in `certs/` directory
   - Otherwise, server runs on HTTP
   - To force HTTPS: `python3 run_server.py --https` (requires certificates)
   - To force HTTP: `python3 run_server.py --http`
2. Open browser:
   - HTTPS: `https://localhost:5421` (if certificates are configured)
   - HTTP: `http://localhost:5421` (if no certificates)

### Speech-to-Text (STT)

1. **Select Input Mode**:
   - **Microphone**: Browser captures from your mic (requires browser permissions)
   - **System Audio**: Server captures system audio (no browser permissions needed)

2. **Start Recording**: Click **"Start Recording"** button

3. **View Transcriptions**:
   - See real-time interim transcriptions as you speak (marked with "⟳ Processing")
   - Final transcriptions appear when speech ends (marked with "✓ Final")
   - Language badges show detected language automatically
   - Confidence scores (95-100% for final transcriptions) and latency metrics are displayed

4. **Stop Recording**: Click **"Stop Recording"** when done

### Text-to-Speech (TTS)

1. **Enter Text**: Type or paste text in the "Enter Text" textarea
   - Language is automatically detected from the text (English, Chinese, or Japanese)

2. **Adjust Speed** (optional): Use the speed slider (0.5x - 2.0x)
   - Default speed is 1.0x (normal speed)

3. **Play**: Click **"Play"** button
   - Language is automatically detected from the text
   - Audio plays immediately after synthesis
   - Status message shows detected language and playback status

4. **Stop**: Click **"Stop"** to stop playback at any time

## System Audio Setup

**macOS:**
- Install BlackHole: `brew install blackhole-2ch`
- Create Multi-Output Device in Audio MIDI Setup
- Add your speakers and BlackHole 2ch to the Multi-Output Device
- Select Multi-Output Device as system output

**Windows:**
- Enable "Stereo Mix" in Sound settings (Recording devices)
- Set as default recording device

**Linux:**
- Configure PulseAudio loopback module:
  ```bash
  # Load loopback module
  pactl load-module module-loopback
  
  # Or configure in /etc/pulse/default.pa
  ```
- Alternatively, use JACK or ALSA loopback devices

## Requirements

- **Python 3.8 or higher** (tested with Python 3.8, 3.9, 3.10, 3.11, 3.12, 3.13)
- Modern browser (Chrome/Edge recommended)
- Whisper model (for STT functionality) - Downloaded automatically on first use, or pre-download with `download_whisper_model.py`

## Setup & Configuration

### STT (Speech-to-Text) Setup

**Offline Whisper STT (Recommended - Works Offline):**

1. **Install Whisper**:
   ```bash
   pip install openai-whisper
   ```

2. **Pre-download Model (Optional, but recommended)**:
   ```bash
   # Download small model (default - optimized for accuracy and speed)
   python download_whisper_model.py --model small
   
   # Or download other models:
   # python download_whisper_model.py --model tiny   # Fastest, lower accuracy
   # python download_whisper_model.py --model base   # Fast, good accuracy
   # python download_whisper_model.py --model small  # Better accuracy (default)
   # python download_whisper_model.py --model medium # High accuracy, slower
   # python download_whisper_model.py --model large   # Highest accuracy, slowest
   ```

3. **Model Sizes**:
   - **tiny**: ~39MB, fastest, lower accuracy (good for testing)
   - **base**: ~74MB, fast, good accuracy
   - **small**: ~244MB, optimized speed/accuracy balance (**default, recommended**)
   - **medium**: ~769MB, high accuracy, slower
   - **large**: ~1550MB, highest accuracy, slowest
   
   **Note**: The default model is `small`, which is optimized with `beam_size=3`, `best_of=3`, and `fp16` acceleration (if available) for the best balance of speed and accuracy.

4. **STT Features**:
   - ✅ **Fully offline** - No internet connection required after model download
   - ✅ Automatically detects language (English, Chinese, Japanese)
   - ✅ Low-latency real-time transcription with interim results (optimized for speed)
   - ✅ Works with both microphone and system audio modes
   - ✅ Models are cached locally after first download
   - ✅ Optimized transcription parameters for speed (beam_size=3, best_of=3)
   - ✅ GPU/MPS acceleration support (fp16) for faster processing when available
   - ✅ Accurate confidence scoring (95-100% for final transcriptions)

**Note**: The model will be downloaded automatically on first use if not pre-downloaded, but this requires an internet connection. Pre-downloading ensures offline operation from the start.

### TTS (Text-to-Speech) Setup

TTS is optional. To enable TTS functionality:

**Install all TTS dependencies:**
```bash
pip install -r requirements.txt
```

**Pre-download models for offline use (requires internet once):**
```bash
# Download all models (English, Chinese, Japanese)
python download_tts_model.py --lang all

# Or download individually:
# English TTS uses PyKokoro-82M (no separate download needed, auto-downloads on first use)
python download_tts_model.py --lang zh  # Chinese (PyKokoro - faster than Piper)
python download_tts_model.py --lang ja  # Japanese (PyKokoro)
```

**TTS Engines:**
- ✅ **English**: PyKokoro-82M (neural TTS, offline-capable) - Auto-downloads on first use
- ✅ **Chinese**: PyKokoro (neural TTS, faster than Piper, no g2pw overhead) - Offline after download
- ✅ **Japanese**: PyKokoro (neural TTS, natural-sounding) - Offline after download

**Additional dependencies for Chinese/Japanese TTS (PyKokoro):**
```bash
# PyKokoro requires spaCy language models and Chinese-specific dependencies
pip install spacy cn2an jieba
python -m spacy download en_core_web_sm  # Required for both Chinese and Japanese
python -m spacy download zh_core_web_sm   # Required for Chinese TTS
python -m spacy download ja_core_news_sm  # Recommended for better Japanese processing
```

**Note**: 
- `cn2an` is required for Chinese TTS to handle number conversion (Chinese numerals ↔ Arabic numerals)
- `jieba` is required for Chinese TTS to perform word segmentation (breaking Chinese text into words)

**Note**: Chinese TTS now uses PyKokoro instead of Piper TTS for better performance and lower latency. The old Piper TTS dependencies (g2pw, unicode-rbnf, sentence-stream) are no longer required.

**Offline Operation:**
- ✅ **All TTS engines**: Work completely offline after initial model download
- ✅ **Model caching**: Models are cached in memory for fast subsequent requests
- ✅ **No internet required**: After initial setup, all TTS works without internet

**Note**: TTS automatically detects language from input text - no manual language selection needed!

## Troubleshooting

### Audio Capture Issues
- **System audio not working**: Check system audio device setup (see System Audio Setup section)
- **VAD not working**: Ensure audio is playing and system audio device is configured correctly
- **Audio level not updating**: Verify audio is being captured and check browser console for errors

### STT Issues
- **No transcriptions appearing**: 
  - Check that Whisper is installed: `pip install openai-whisper`
  - Verify model is downloaded (check `~/.cache/whisper/` directory)
  - Model downloads automatically on first use (requires internet once)
  - Check server logs for model loading errors
  - Check browser console for errors
- **Transcriptions cut off**: 
  - Wait for speech to complete - final transcription appears after speech ends
  - Check that audio is being captured properly
  - Whisper processes audio in chunks, so there may be a slight delay
  - The system uses pre-buffering and extended stopping periods to capture complete speech
- **Wrong language detected**: 
  - Language is auto-detected from audio content
  - For mixed-language audio, the dominant language will be detected
- **Slow transcription**: 
  - The default `small` model is already optimized for speed (beam_size=3, best_of=3)
  - Use a smaller model (e.g., `tiny` or `base` instead of `small`) for even faster processing
  - Edit `backend/audio/pipeline.py` to change model size in `WhisperOfflineSTT` initialization
  - GPU/MPS acceleration (fp16) is automatically enabled if available for faster processing
- **Model download fails**: 
  - Check internet connection (required only for first download)
  - Manually download: `python download_whisper_model.py --model base`
  - Models are cached in `~/.cache/whisper/` after download

### TTS Issues
- **TTS not available error**: 
  - Install TTS dependencies: `pip install -r requirements.txt`
  - For Chinese/Japanese: Install spaCy models:
    ```bash
    pip install spacy cn2an jieba
    python -m spacy download en_core_web_sm  # Required for both Chinese and Japanese
    python -m spacy download zh_core_web_sm   # Required for Chinese TTS
    python -m spacy download ja_core_news_sm  # Recommended for Japanese TTS
    ```
  - Restart server after installing dependencies
- **No audio playback**: 
  - Check browser console for errors
  - Verify audio permissions in browser
  - Try a different browser (Chrome/Edge recommended)
- **Wrong language for TTS**: 
  - Language is automatically detected from text
  - English text uses PyKokoro-82M (offline-capable)
  - Chinese text uses PyKokoro (offline, faster than Piper)
  - Japanese text uses PyKokoro (offline)
- **Model download errors**:
  - Ensure internet connection for initial download
  - Run `python download_tts_model.py --lang <language>` to pre-download models
  - Models are cached locally after download for offline use
- **Japanese TTS not working**:
  - Ensure PyKokoro is installed: `pip install pykokoro`
  - Install required spaCy models:
    ```bash
    pip install spacy
    python -m spacy download en_core_web_sm  # Required
    python -m spacy download ja_core_news_sm  # Recommended for better Japanese processing
    ```
  - Check server logs for specific error messages
- **Chinese TTS not working**:
  - Ensure PyKokoro and Chinese dependencies are installed:
    ```bash
    pip install pykokoro cn2an jieba spacy
    python -m spacy download en_core_web_sm  # Required
    python -m spacy download zh_core_web_sm   # Required for Chinese
    ```
  - Check server logs for specific error messages