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
- Confidence scores and latency metrics
- Optimized for noisy factory conditions
- Works completely offline after model download

### ✅ Milestone 3: Language Handling & Text-to-Speech (TTS)
- Automatic language detection from text input
- Multi-language TTS support:
  - **English**: MMS-TTS (PyTorch-based)
  - **Chinese**: gTTS (Google Text-to-Speech)
  - **Japanese**: gTTS (Google Text-to-Speech)
- Adjustable playback speed (0.5x - 2.0x)
- Real-time audio playback in browser

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

Server runs on `https://localhost:5421`.

## Features

### Audio Capture & Processing
- ✅ **Microphone Mode**: Record audio from your microphone
- ✅ **System Audio Mode**: Capture system audio directly on server (no browser permissions needed)
- ✅ **Voice Activity Detection (VAD)**: Real-time speech detection with visual indicators
- ✅ **Noise Reduction**: RNNoise-based noise suppression for factory environments
- ✅ **Audio Level Monitoring**: Real-time audio level display with RMS and peak levels

### Speech-to-Text (STT)
- ✅ **Offline Transcription**: Fully offline STT using OpenAI Whisper (no internet required)
- ✅ **Real-time Transcription**: Low-latency transcription with streaming support
- ✅ **Interim Results**: See transcription as you speak (real-time updates)
- ✅ **Final Results**: Accurate final transcriptions with confidence scores
- ✅ **Multi-language Support**: Automatic detection of English, Chinese, and Japanese
- ✅ **Language Badges**: Visual indicators showing detected language (🇬🇧 English, 🇨🇳 Chinese, 🇯🇵 Japanese)
- ✅ **Latency Metrics**: Track transcription latency for performance monitoring
- ✅ **Confidence Scores**: Display transcription confidence levels

### Text-to-Speech (TTS)
- ✅ **Automatic Language Detection**: Detects language from input text automatically
- ✅ **Multi-language TTS**: Supports English, Chinese, and Japanese
- ✅ **Adjustable Speed**: Control playback speed from 0.5x to 2.0x
- ✅ **Real-time Playback**: Instant audio playback in browser
- ✅ **High-quality Voices**: Natural-sounding speech synthesis

## Usage

### Starting the Server

1. Start the server: `python3 run_server.py`
2. Open browser: `https://localhost:5421`

### Speech-to-Text (STT)

1. **Select Input Mode**:
   - **Microphone**: Browser captures from your mic (requires browser permissions)
   - **System Audio**: Server captures system audio (no browser permissions needed)

2. **Start Recording**: Click **"Start Recording"** button

3. **View Transcriptions**:
   - See real-time interim transcriptions as you speak (marked with "⟳ Processing")
   - Final transcriptions appear when speech ends (marked with "✓ Final")
   - Language badges show detected language automatically
   - Confidence scores and latency metrics are displayed

4. **Stop Recording**: Click **"Stop Recording"** when done

### Text-to-Speech (TTS)

1. **Enter Text**: Type or paste text in the "Text to Speak" textarea

2. **Adjust Speed** (optional): Use the speed slider (0.5x - 2.0x)

3. **Play**: Click **"🔊 Play"** button
   - Language is automatically detected from the text
   - Audio plays immediately after synthesis

4. **Stop**: Click **"⏹ Stop"** to stop playback

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
- Configure PulseAudio loopback module

## Requirements

- Python 3.8+
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
   # Download base model (recommended - good balance of speed and accuracy)
   python download_whisper_model.py --model base
   
   # Or download other models:
   # python download_whisper_model.py --model tiny   # Fastest, lower accuracy
   # python download_whisper_model.py --model small  # Better accuracy
   # python download_whisper_model.py --model medium # High accuracy
   # python download_whisper_model.py --model large   # Highest accuracy
   ```

3. **Model Sizes**:
   - **tiny**: ~39MB, fastest, lower accuracy (good for testing)
   - **base**: ~74MB, fast, good accuracy (**recommended**)
   - **small**: ~244MB, medium speed, better accuracy
   - **medium**: ~769MB, slow, high accuracy
   - **large**: ~1550MB, slowest, highest accuracy

4. **STT Features**:
   - ✅ **Fully offline** - No internet connection required after model download
   - ✅ Automatically detects language (English, Chinese, Japanese)
   - ✅ Low-latency real-time transcription with interim results
   - ✅ Works with both microphone and system audio modes
   - ✅ Models are cached locally after first download

**Note**: The model will be downloaded automatically on first use if not pre-downloaded, but this requires an internet connection. Pre-downloading ensures offline operation from the start.

### TTS (Text-to-Speech) Setup

TTS is optional. To enable TTS functionality:

**For English TTS (Offline-capable, Recommended):**
```bash
# Install dependencies
pip install torch transformers datasets soundfile

# Pre-download model for offline use (requires internet once)
python download_tts_model.py
```

**For Chinese/Japanese TTS (Requires Internet):**
```bash
pip install gtts pydub
```

**For speed adjustment (Optional):**
```bash
pip install librosa
```

**Quick install script:**
```bash
./install_tts.sh
```

**Or install all at once:**
```bash
pip install gtts pydub torch transformers datasets soundfile librosa
python download_tts_model.py  # Pre-download English model for offline use
```

**Offline Operation:**
- ✅ **English TTS**: Works offline after model is downloaded (run `download_tts_model.py` once with internet)
- ❌ **Chinese/Japanese TTS**: Requires internet (uses gTTS/Google API)

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
- **Wrong language detected**: 
  - Language is auto-detected from audio content
  - For mixed-language audio, the dominant language will be detected
- **Slow transcription**: 
  - Use a smaller model (e.g., `tiny` or `base` instead of `large`)
  - Edit `backend/audio/pipeline.py` to change model size in `WhisperOfflineSTT` initialization
- **Model download fails**: 
  - Check internet connection (required only for first download)
  - Manually download: `python download_whisper_model.py --model base`
  - Models are cached in `~/.cache/whisper/` after download

### TTS Issues
- **TTS not available error**: 
  - Install TTS dependencies (see TTS Setup section)
  - Restart server after installing dependencies
- **No audio playback**: 
  - Check browser console for errors
  - Verify audio permissions in browser
  - Try a different browser (Chrome/Edge recommended)
- **Wrong language for TTS**: 
  - Language is automatically detected from text
  - Chinese/Japanese characters will use gTTS
  - English text will use MMS-TTS (if installed) or gTTS