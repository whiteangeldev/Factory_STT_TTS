# Streaming STT Migration - Complete Implementation

## Overview

The project has been fully converted from local Whisper model to **AssemblyAI Realtime Streaming STT API**. This provides true real-time transcription with minimal latency.

## What Changed

### 1. **New Streaming STT Implementation** (`backend/audio/stt_streaming.py`)
   - Uses AssemblyAI Realtime API for streaming transcription
   - Streams audio chunks directly via WebSocket
   - Provides real-time interim and final transcriptions
   - No file I/O - pure streaming

### 2. **Updated Pipeline** (`backend/audio/pipeline.py`)
   - Completely rewritten to use streaming STT
   - Starts streaming when speech is detected
   - Sends audio chunks directly to streaming API
   - Handles transcriptions via callbacks
   - Stops streaming when speech ends

### 3. **Removed Dependencies**
   - No longer uses local Whisper model
   - No file-based transcription
   - No temporary WAV file creation

## Key Features

✅ **Real-time Streaming**: Audio is streamed directly to AssemblyAI as it's captured  
✅ **Low Latency**: No file I/O overhead, immediate transcription  
✅ **Interim Results**: Get partial transcriptions while speaking  
✅ **Final Results**: Get complete transcriptions when speech ends  
✅ **Auto Language Detection**: Automatically detects spoken language  
✅ **No Local Models**: No need to download or manage Whisper models  

## Setup

### 1. Install Dependencies
```bash
pip install assemblyai
```

### 2. Set API Key
```bash
export ASSEMBLYAI_API_KEY="your-api-key-here"
```

Get your free API key at: https://www.assemblyai.com/
- Free tier includes 5 hours of transcription per month

### 3. Run the Server
```bash
python backend/server.py
```

## How It Works

1. **Audio Capture**: Frontend captures audio from microphone
2. **VAD Detection**: Backend detects when speech starts/stops
3. **Stream Start**: When speech is detected, streaming STT connection is opened
4. **Audio Streaming**: Audio chunks are sent directly to AssemblyAI in real-time
5. **Transcription**: AssemblyAI returns interim and final transcriptions via callbacks
6. **Stream Stop**: When speech ends, streaming connection is closed

## Architecture

```
Frontend (Browser)
    ↓ (WebSocket - audio chunks)
Backend Server
    ↓ (VAD - speech detection)
Audio Pipeline
    ↓ (Streaming STT)
AssemblyAI Realtime API
    ↓ (WebSocket - transcriptions)
Backend Server
    ↓ (WebSocket - events)
Frontend (Browser)
```

## Benefits Over Local Whisper

| Feature | Local Whisper | Streaming STT |
|---------|---------------|---------------|
| Latency | High (file-based) | Low (real-time streaming) |
| Setup | Model download required | API key only |
| Accuracy | Good | Excellent |
| Language Detection | Manual | Automatic |
| Resource Usage | High (GPU/CPU) | Low (cloud-based) |
| Scalability | Limited | Unlimited |

## API Events

The system emits the following events:

- `speech_start`: When speech is detected
- `transcription_interim`: Partial transcription (while speaking)
- `transcription`: Final transcription (when speech ends)
- `speech_end`: When speech stops

## Troubleshooting

### No Transcriptions
- Check that `ASSEMBLYAI_API_KEY` is set
- Verify API key is valid
- Check network connectivity

### Connection Errors
- Ensure internet connection is stable
- Check AssemblyAI service status
- Verify API key has streaming access

## Migration Notes

- Old file-based STT (`stt_assemblyai.py`) is no longer used
- Pipeline no longer buffers audio for file transcription
- All transcription is now real-time streaming
- No changes needed to frontend - it works with existing WebSocket events
