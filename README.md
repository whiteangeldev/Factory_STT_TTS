# Factory STT/TTS

Real-time Speech-to-Text server using OpenAI Realtime API with VAD and noise suppression.

## Quick Start

### 1. Generate SSL Certificates (for HTTPS)

```bash
python3 generate_certs.py
```

### 2. Run Server

```bash
python3 run_server.py
```

The server will automatically use HTTPS if certificates exist, otherwise HTTP.

**Options:**
- `python3 run_server.py` - Auto-detect HTTPS/HTTP
- `python3 run_server.py --https` - Force HTTPS (requires certs)
- `python3 run_server.py --http` - Force HTTP
- `python3 run_server.py --port 8080` - Custom port

### Manual Setup (Optional)

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate.bat   # Windows

# Install dependencies
pip install -r requirements.txt

# Run server
python3 run_server.py
```

## Configuration

Set environment variables or create `.env` file:
```
OPENAI_API_KEY=your-api-key-here
HOST=0.0.0.0
PORT=8000
```

## Features

- ✅ **Microphone Mode**: Record from your microphone
- ✅ **System Audio Mode**: Capture system audio (no browser permissions needed!)
- ✅ Real-time speech-to-text using OpenAI Realtime API
- ✅ Voice Activity Detection (VAD) using WebRTC
- ✅ Noise suppression using RNNoise
- ✅ WebSocket-based audio streaming
- ✅ Automatic HTTPS if certificates exist in `certs/`

## Usage

1. Run: `python3 run_server.py`
2. Open browser: `http://localhost:8000` (or `https://` if SSL certs exist)
3. Select input mode:
   - **Microphone**: Records from your mic
   - **System Audio**: Captures system audio from server (no browser permissions!)
4. Click "Start Recording"
5. See real-time transcriptions

## System Audio Mode

**No browser permissions needed!** Captures system audio directly on the server.

1. **Select "System Audio"** from dropdown
2. **Click "Start Recording"**
3. **Done!** No browser dialogs, no screen sharing needed

**Setup (one-time):**
```bash
pip install sounddevice
```

**Platform-specific setup:**
- **macOS**: May need to enable "Monitor" device in Audio MIDI Setup, or install BlackHole
- **Windows**: Enable "Stereo Mix" in Sound settings (Recording devices)
- **Linux**: Configure PulseAudio loopback module

### Troubleshooting

- **System audio not working**: Install `sounddevice` and check system audio device setup
- **No system audio device found**: See platform-specific setup above

## Requirements

- Python 3.8+
- OpenAI API key
- Modern browser (Chrome/Edge recommended for system audio)

## Troubleshooting

- **No transcription**: Check `OPENAI_API_KEY` is set
- **System audio not working**: Use Chrome/Edge, enable "Share system audio" in browser
- **Connection errors**: Verify network connectivity to OpenAI API
