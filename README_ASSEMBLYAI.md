# AssemblyAI Streaming STT Integration

This project uses **AssemblyAI Realtime Streaming API** for real-time Speech-to-Text transcription.

## Setup

1. **Install dependencies:**
   ```bash
   pip install assemblyai
   ```

2. **Set your AssemblyAI API key:**
   ```bash
   export ASSEMBLYAI_API_KEY="your-api-key-here"
   ```
   
   Or create a `.env` file:
   ```
   ASSEMBLYAI_API_KEY=your-api-key-here
   ```

3. **Get your API key:**
   - Sign up at https://www.assemblyai.com/
   - Get your API key from the dashboard
   - Free tier includes 5 hours of transcription per month

## Changes Made

- Created `backend/audio/stt_streaming.py` - AssemblyAI Realtime Streaming STT implementation
- Updated `backend/audio/pipeline.py` - Now uses streaming STT instead of file-based transcription
- Removed dependency on local Whisper model
- No model download needed - AssemblyAI uses cloud-based streaming transcription

## How It Works

1. **Audio Capture**: Frontend captures audio from microphone
2. **VAD Detection**: Backend detects when speech starts/stops using WebRTC VAD
3. **Stream Start**: When speech is detected, streaming STT connection is opened
4. **Real-time Streaming**: Audio chunks are sent directly to AssemblyAI via WebSocket
5. **Transcription**: AssemblyAI returns interim and final transcriptions in real-time
6. **Stream Stop**: When speech ends, streaming connection is closed

## Benefits

- **Real-time Streaming**: Audio is streamed directly as it's captured (no file I/O)
- **Low Latency**: Minimal delay between speech and transcription
- **No Local Model**: No need to download or manage Whisper models
- **High Accuracy**: Professional-grade transcription from AssemblyAI
- **Auto Language Detection**: Automatically detects the spoken language
- **Interim Results**: Get partial transcriptions while speaking
- **Final Results**: Get complete transcriptions when speech ends

## Architecture

```
Frontend (Browser)
    ↓ WebSocket (audio chunks)
Backend Server
    ↓ VAD (speech detection)
Audio Pipeline
    ↓ Streaming STT
AssemblyAI Realtime API (WebSocket)
    ↓ transcriptions
Backend Server
    ↓ WebSocket (events)
Frontend (Browser)
```

## API Events

The system emits the following events via WebSocket:

- `speech_start`: When speech is detected
- `transcription_interim`: Partial transcription (while speaking)
- `transcription`: Final transcription (when speech ends)
- `speech_end`: When speech stops

## Notes

- Requires internet connection (uses cloud API)
- API key required (free tier available)
- Streaming transcription happens in real-time via WebSocket
- No file I/O overhead - pure streaming

## Troubleshooting

### No Transcriptions
- Check that `ASSEMBLYAI_API_KEY` is set
- Verify API key is valid
- Check network connectivity

### Connection Errors
- Ensure internet connection is stable
- Check AssemblyAI service status
- Verify API key has streaming access
