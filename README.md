# Factory STT/TTS - Voice AI for Industrial Chatbot

Industrial-grade Speech-to-Text and Text-to-Speech system with noise robustness for factory environments (85dB+ background noise).

## 🎯 Overview

This system implements a complete voice AI pipeline optimized for industrial factory settings with:
- **Real-time audio preprocessing** for noisy environments (85dB+)
- **WebRTC VAD** (Voice Activity Detection) for fast, accurate speech detection
- **RNNoise** noise suppression for high-noise environments
- **Multi-language support** (Japanese, English, Chinese)
- **Low-latency processing** for real-time interaction

## 🏗️ Architecture

### Audio Processing Pipeline

The system uses an optimized pipeline order for maximum effectiveness in high-noise environments:

```
Input Audio (noisy, 85dB+)
    ↓
RNNoise (Noise Suppression) ← Cleans audio FIRST
    ↓
Normalize (Boost signal level)
    ↓
Noise Gate (Reject very low-level noise)
    ↓
WebRTC VAD (Fast detection on CLEAN audio) ← Key improvement
    ↓
Output (Clean speech ready for STT)
```

**Why this order matters:**
- VAD sees **clean audio** (after noise suppression), dramatically improving detection accuracy
- Noise suppression happens **before** VAD, not after
- This is critical for 85dB+ factory noise environments

### Key Components

#### 1. Voice Activity Detection (VAD) - WebRTC
- **Technology**: WebRTC VAD (DSP-based, not ML)
- **Why WebRTC**: 
  - 2-3x faster than Silero VAD (~10-30ms vs ~30-50ms)
  - Lower CPU usage (no PyTorch overhead)
  - Better for high noise environments
  - Optimized for real-world conditions
- **Configuration**: `VAD_AGGRESSIVENESS` (0-3, 3 = most aggressive for factory noise)
- **Frame Size**: 30ms (WebRTC supports 10, 20, or 30ms)

#### 2. Noise Suppression - RNNoise
- **Technology**: RNNoise (neural network-based)
- **Why RNNoise**: 
  - Better quality than DSP-based methods for 85dB+ noise
  - Hybrid DSP + ML approach
  - Proven in production (used in NoiseTorch, video conferencing apps)
- **Fallback**: noisereduce (spectral subtraction) if RNNoise unavailable
- **Configuration**: `NOISE_REDUCTION_STRENGTH` (0-1, default 0.8)

#### 3. Audio Normalization
- **Purpose**: Boost quiet speech to consistent level
- **Target**: -20 dBFS (configurable)
- **Max Gain**: Limited to 100x to prevent artifacts

#### 4. Noise Gating
- **Purpose**: Reject very low-level noise before processing
- **Default**: Disabled for quiet environments
- **Factory Mode**: Enable for 85dB+ environments

## 📋 Prerequisites

### System Requirements
- Python 3.10+
- Linux/Ubuntu (tested on Ubuntu 20.04+)
- Microphone access
- Modern web browser (Chrome, Firefox, Edge)

### System Dependencies

```bash
# Install Python development headers and audio libraries
sudo apt-get update
sudo apt-get install -y python3-dev python3.10-dev portaudio19-dev
```

## 🚀 Installation

1. **Clone or navigate to the project directory:**
```bash
cd Factory_STT_TTS
```

2. **Create virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**Note:** If PyAudio installation fails, ensure you've installed the system dependencies above.

## 🏃 Running the Application

### Start the Server

#### Option 1: HTTP (for localhost only)

```bash
# Activate virtual environment (if not already active)
source .venv/bin/activate

# Run the server
python run_server.py

# Or with custom host/port
python run_server.py --host 0.0.0.0 --port 8000
```

**Note:** HTTP only works for `localhost`. For VPS/remote access, use HTTPS (Option 2).

#### Option 2: HTTPS (Required for VPS/Remote Access)

```bash
# Activate virtual environment
source .venv/bin/activate

# Generate SSL certificate (first time only)
bash generate_ssl_cert.sh

# Start server with HTTPS
python run_server.py --https

# Or use the quick start script
./start_https.sh
```

The server will start on `https://0.0.0.0:8000`

**Access:** `https://YOUR_IP:8000` (replace with your VPS IP)

**Important:** Chrome will show a security warning for self-signed certificates. Click "Advanced" → "Proceed to site" to continue.

### Access the Frontend

**For Localhost (HTTP):**
```
http://localhost:8000
```

**For VPS/Remote Access (HTTPS required):**
```
https://YOUR_IP:8000
```

**⚠️ Important for VPS:**
- Chrome requires HTTPS for microphone access on non-localhost addresses
- Use `--https` flag when starting the server
- Chrome will show a security warning - click "Advanced" → "Proceed" to continue

## 🎮 Usage

1. **Connect to Server:**
   - The frontend will automatically connect via WebSocket
   - Check the connection status indicator (top right)

2. **Start Recording:**
   - Click "Start Listening" button
   - Grant microphone permissions when prompted
   - The status will change to "Listening..."

3. **Monitor Audio:**
   - Watch the audio level meter (shows dB values)
   - Check system status indicators
   - View processing log for system messages

4. **Save Recording:**
   - While recording, click "Save Recording" button
   - Recording will be saved as WAV file
   - File will be downloaded automatically

5. **Stop Recording:**
   - Click "Stop" to end the session

## 🔧 Configuration

Edit `backend/config.py` to adjust settings:

### Audio Settings
- `SAMPLE_RATE`: Audio sample rate (default: 16000 Hz)
- `CHANNELS`: Audio channels (default: 1 = mono)
- `CHUNK_SIZE`: Audio buffer size (default: 1024)

### VAD Settings (WebRTC)
- `VAD_AGGRESSIVENESS`: 0-3 (0=least, 3=most aggressive)
  - **0-1**: Quiet environments
  - **2**: Moderate noise
  - **3**: High noise (85dB+ factory) - **Recommended**
- `VAD_FRAME_MS`: Frame size in milliseconds (10, 20, or 30)

### Noise Suppression
- `ENABLE_NOISE_SUPPRESSION`: Enable/disable noise reduction
- `NOISE_REDUCTION_STRENGTH`: Aggressiveness (0-1, default: 0.8)
- `USE_RNNOISE`: Use RNNoise if available (better for 85dB+)

### Audio Normalization
- `ENABLE_NORMALIZATION`: Enable/disable audio level normalization
- `TARGET_DBFS`: Target audio level in dBFS (default: -20)

### Noise Gating
- `ENABLE_NOISE_GATING`: Enable/disable noise gate
  - **False**: Quiet environments (default)
  - **True**: 85dB+ factory noise environments
- `NOISE_GATE_THRESHOLD`: RMS threshold (default: 0.005)

### Speech Detection
- `HANGOVER_MS`: Milliseconds to keep speech active after detection ends (default: 500)
- `MIN_SPEECH_MS`: Minimum speech duration to trigger speech_start (default: 100)

## 🏗️ Project Structure

```
Factory_STT_TTS/
├── backend/
│   ├── __init__.py
│   ├── server.py              # FastAPI server with WebSocket
│   ├── config.py              # Audio configuration
│   └── audio/
│       ├── __init__.py
│       ├── capture.py         # Audio capture module
│       ├── vad.py             # WebRTC Voice Activity Detection
│       ├── noise_suppression.py  # RNNoise noise reduction
│       ├── pipeline.py        # Main preprocessing pipeline
│       └── aec.py             # Acoustic Echo Cancellation (interface)
├── frontend/
│   ├── index.html            # Main HTML page
│   └── static/
│       ├── style.css         # Styling
│       └── app.js            # Frontend JavaScript
├── requirements.txt          # Python dependencies
├── run_server.py            # Server startup script
├── generate_ssl_cert.sh     # SSL certificate generator
├── start_https.sh           # Quick HTTPS start script
└── README.md               # This file
```

## 🔬 Technical Details

### WebRTC VAD Implementation

**Why WebRTC VAD instead of Silero?**
- **Speed**: 2-3x faster (10-30ms vs 30-50ms)
- **CPU**: Lower usage (no PyTorch overhead)
- **Noise**: Better performance in high-noise environments
- **Real-time**: Optimized for real-world conditions

**How it works:**
1. Converts audio to 16-bit PCM
2. Processes in fixed-size frames (10/20/30ms)
3. Uses DSP-based detection (energy, spectral features)
4. Returns binary decision (speech/no-speech)

**Configuration:**
- Aggressiveness 0-3 controls sensitivity
- Higher aggressiveness = more sensitive (catches more speech in noise)
- Frame size affects latency vs accuracy trade-off

### RNNoise Noise Suppression

**Why RNNoise?**
- **Quality**: Better than pure DSP methods for 85dB+ noise
- **Hybrid**: Combines DSP features with small neural network
- **Proven**: Used in production (NoiseTorch, video conferencing)
- **Efficient**: Low latency, reasonable CPU usage

**How it works:**
1. Requires 48kHz sample rate (resamples if needed)
2. Processes in 480-sample frames (10ms at 48kHz)
3. Uses neural network to predict per-band gains
4. Applies gains to suppress noise while preserving speech

**Fallback:**
- If RNNoise unavailable, uses noisereduce (spectral subtraction)
- Still effective but lower quality in extreme noise

### Pipeline Order (Critical for High Noise)

**Old (Problematic) Order:**
```
Input → Normalize → VAD → Noise Suppression → Output
```
**Problem**: VAD tries to detect speech in raw noisy audio

**New (Optimized) Order:**
```
Input → Noise Suppression → Normalize → VAD → Output
```
**Benefit**: VAD sees clean audio, dramatically improving accuracy

This reordering is the **most critical improvement** for 85dB+ factory noise.

### Speech State Machine

The system uses a state machine to track speech:

```
SILENCE → SPEECH_START → SPEECH → SPEECH_END → SILENCE
```

**Features:**
- **Hangover Logic**: Keeps speech active for 500ms after VAD stops detecting (prevents truncation)
- **Minimum Duration**: Requires 100ms of speech before triggering speech_start (prevents false triggers)
- **Event Callbacks**: Emits `speech_start` and `speech_end` events for STT integration

## 🧪 Testing

### Test Audio Pipeline (Backend Only)

```bash
python test_audio_pipeline.py
```

This will:
- Test microphone capture
- Process audio through the pipeline
- Display processing statistics

### Test Factory Noise Scenarios

```bash
python test_factory_noise.py
```

This will:
- Test with synthetic factory noise
- Measure SNR improvement
- Calculate false trigger rate
- Measure latency
- Detect clipping

### Test Web Interface

1. Start the server: `python run_server.py --https`
2. Open browser: `https://localhost:8000`
3. Click "Start Listening" and speak into microphone
4. Monitor the processing in real-time
5. Click "Save Recording" to save audio as WAV file

## 🐛 Troubleshooting

### Microphone Not Working

#### Issue: "Failed to access microphone" or Permission Denied

**Problem:** Chrome blocks microphone access on HTTP for non-localhost addresses.

**Solutions:**

1. **Use Localhost (Easiest):**
   - Instead of `http://YOUR_IP:8000`
   - Use `http://localhost:8000` or `http://127.0.0.1:8000`
   - Localhost works with HTTP (no HTTPS needed)

2. **Grant Permissions in Chrome:**
   - Click lock icon (🔒) in address bar → Site settings
   - OR go to: `chrome://settings/content/microphone`
   - Find your site → Set Microphone to "Allow"
   - If blocked, click "Reset permissions" first

3. **For Remote Access:**
   - Use SSH tunnel: `ssh -L 8000:localhost:8000 user@server`
   - Then access via `http://localhost:8000`

4. **Set Up HTTPS (For Production):**
   - Run: `bash generate_ssl_cert.sh`
   - Start server: `python run_server.py --https`
   - Access: `https://YOUR_IP:8000`

**Quick Reference:**
- ✅ `http://localhost:8000` - Works (no HTTPS needed)
- ✅ `http://127.0.0.1:8000` - Works (no HTTPS needed)
- ❌ `http://YOUR_IP:8000` - Blocked (needs HTTPS)
- ✅ `https://YOUR_IP:8000` - Works (with HTTPS setup)

### Speech Not Detected

**Problem:** Speech not being detected in quiet office or noisy factory.

**Solutions:**

1. **Check VAD Aggressiveness:**
   - For quiet environments: `VAD_AGGRESSIVENESS = 2`
   - For factory noise: `VAD_AGGRESSIVENESS = 3` (most aggressive)

2. **Check Audio Level:**
   - Look at audio level meter (should show -40 to -20 dB for normal speech)
   - If consistently < -50 dB, microphone may be too quiet

3. **Enable Noise Gate:**
   - For factory noise: `ENABLE_NOISE_GATING = True`
   - For quiet office: `ENABLE_NOISE_GATING = False`

4. **Check Noise Suppression:**
   - Ensure `ENABLE_NOISE_SUPPRESSION = True`
   - Ensure `USE_RNNOISE = True` (if available)

### WebSocket Connection Failed
- Ensure server is running
- Check firewall settings
- Verify port 8000 is not in use

### PyAudio Installation Issues
```bash
# Reinstall system dependencies
sudo apt-get install --reinstall python3-dev portaudio19-dev
pip install --upgrade pip
pip install pyaudio
```

### WebRTC VAD Not Working
```bash
# Install webrtcvad
pip install webrtcvad==2.0.10
```

If WebRTC VAD is not available, the system falls back to energy-based VAD (less accurate).

## 📊 Performance

- **Latency:** < 50ms per audio chunk (processing)
- **CPU Usage:** Low-Moderate (WebRTC VAD is lightweight)
- **Memory:** ~100-200MB (no PyTorch overhead)
- **Accuracy:** High in 85dB+ factory noise (with optimized pipeline)

## 🔜 Next Milestones

- **Milestone 2:** Speech-to-Text Integration (STT)
- **Milestone 3:** Text-to-Speech (TTS) & Barge-in
- **Milestone 4:** Chatbot Integration & Factory Validation

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributors here]
