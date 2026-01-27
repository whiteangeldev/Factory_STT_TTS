# OpenAI Realtime API Migration

## Overview

The project has been migrated from AssemblyAI to **OpenAI Realtime API** for streaming transcription, while keeping **RNNoise** noise suppression and **VAD** (Voice Activity Detection).

## What Changed

### 1. **Removed AssemblyAI**
   - Deleted `backend/audio/stt_assemblyai.py`
   - Deleted `backend/audio/stt_streaming.py`
   - Removed `assemblyai` from requirements.txt

### 2. **Added OpenAI Realtime STT** (`backend/audio/stt_openai_realtime.py`)
   - Uses OpenAI Realtime API WebSocket for streaming transcription
   - Supports real-time interim and final transcriptions
   - Uses `gpt-4o-realtime-preview-2024-10-01` model

### 3. **Implemented RNNoise** (`backend/audio/rnnoise.py`)
   - RNNoise-based noise suppression
   - Falls back to simple noise gate if RNNoise library not available
   - Handles sample rate conversion (RNNoise works at 48kHz internally)

### 4. **Updated Pipeline** (`backend/audio/pipeline.py`)
   - Integrated VAD + RNNoise + OpenAI Realtime STT
   - Processing flow: Audio → RNNoise → VAD → OpenAI Realtime API
   - Handles sample rate conversion (16kHz → 24kHz for OpenAI)

## Architecture

```
Microphone → Frontend
    ↓ (WebSocket - audio chunks)
Backend Server
    ↓
Audio Pipeline
    ↓ RNNoise (noise suppression)
    ↓ VAD (speech detection)
    ↓ Resample to 24kHz
    ↓ OpenAI Realtime API (WebSocket)
    ↓ transcriptions
Backend Server
    ↓ (WebSocket - events)
Frontend
```

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Optional: Install RNNoise** (for better noise suppression)
```bash
# RNNoise requires building from source
# See: https://github.com/xiph/rnnoise
# Or use: pip install rnnoise-python (if available)
```

If RNNoise is not installed, the system will use a simple noise gate fallback.

### 2. Set OpenAI API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Get your API key at: https://platform.openai.com/api-keys

### 3. Run the Server
```bash
python backend/server.py
```

## Key Features

✅ **RNNoise Noise Suppression**: Advanced noise reduction before transcription  
✅ **VAD Gate**: Voice Activity Detection to only transcribe when speech is detected  
✅ **OpenAI Realtime API**: Real-time streaming transcription with low latency  
✅ **Auto Resampling**: Handles sample rate conversion (16kHz → 24kHz)  
✅ **Fallback Support**: Works even if RNNoise is not installed  

## Processing Flow

1. **Audio Capture**: Frontend captures audio at 16kHz
2. **RNNoise**: Noise suppression applied to audio
3. **VAD Detection**: Detects when speech starts/stops
4. **Resampling**: Audio resampled to 24kHz for OpenAI Realtime API
5. **Streaming**: Audio chunks streamed to OpenAI via WebSocket
6. **Transcription**: OpenAI returns real-time transcriptions
7. **Events**: Transcriptions sent to frontend via WebSocket

## API Events

The system emits the following events:

- `speech_start`: When speech is detected (VAD)
- `transcription_interim`: Partial transcription (while speaking)
- `transcription`: Final transcription (when speech ends)
- `speech_end`: When speech stops (VAD)

## Requirements

### Required
- `openai`: OpenAI Python SDK
- `websocket-client`: WebSocket client for OpenAI Realtime API
- `numpy`: Audio processing
- `webrtcvad`: Voice Activity Detection
- `scipy`: Audio resampling (optional, has fallback)

### Optional
- `rnnoise` or `rnnoise-python`: Advanced noise suppression (falls back to simple noise gate if not available)

## Troubleshooting

### No Transcriptions
- Check that `OPENAI_API_KEY` is set
- Verify API key is valid and has Realtime API access
- Check network connectivity

### RNNoise Not Working
- RNNoise is optional - system will use simple noise gate fallback
- To use RNNoise, install the library (may require building from source)
- Check logs for RNNoise initialization messages

### Connection Errors
- Ensure internet connection is stable
- Check OpenAI service status
- Verify API key has Realtime API access

## Notes

- OpenAI Realtime API requires 24kHz sample rate (system handles conversion)
- RNNoise works internally at 48kHz (system handles conversion)
- VAD works at configured sample rate (default 16kHz)
- All conversions are handled automatically
