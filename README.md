# Factory STT/TTS - Milestone 1

Real-time audio capture with Voice Activity Detection (VAD) and dual input modes.

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

## Features (Milestone 1)

- ✅ **Microphone Mode**: Record audio from your microphone
- ✅ **System Audio Mode**: Capture system audio directly on server (no browser permissions)
- ✅ **Voice Activity Detection**: Visual indicators for speech detection
- ✅ **Audio Level Monitoring**: Real-time audio level display

## Usage

1. Start the server: `python3 run_server.py`
2. Open browser: `https://localhost:5421`
3. Select input mode:
   - **Microphone**: Browser captures from your mic
   - **System Audio**: Server captures system audio (no browser permissions needed)
4. Click **"Start Recording"**
5. View VAD status and audio level indicators

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

## TTS (Text-to-Speech) Setup

TTS is optional. To enable TTS functionality:

**For Chinese/Japanese TTS (Recommended - Lightweight):**
```bash
pip install gtts pydub
```

**For English TTS (Requires PyTorch):**
```bash
pip install torch transformers datasets soundfile
```

**For speed adjustment (Optional):**
```bash
pip install librosa
```

**Quick install script:**
```bash
./install_tts.sh
```

Or install all at once:
```bash
pip install gtts pydub torch transformers datasets soundfile librosa
```

## Troubleshooting

- **System audio not working**: Check system audio device setup (see above)
- **VAD not working**: Ensure audio is playing and system audio device is configured correctly
- **Audio level not updating**: Verify audio is being captured and check browser console for errors