# Fixes Applied for VAD and STT Issues

## Problems Identified

1. **VAD Frame Size Mismatch**: VAD expected exactly 480 samples (30ms), but frontend was sending 512-sample chunks. VAD only processed the first frame, ignoring the rest.

2. **VAD Too Strict**: Aggressiveness was set to 3 (most aggressive), which was too strict and missing speech.

3. **Audio Buffer Validation**: Missing validation for buffer size alignment (must be multiple of 2 bytes for int16).

4. **Limited VAD Processing**: VAD only checked the first frame in each chunk, not all frames.

## Fixes Applied

### 1. VAD Improvements (`backend/audio/vad.py`)
- **Fixed**: VAD now processes ALL complete frames in each chunk, not just the first one
- **Result**: Better speech detection across variable-sized chunks

### 2. VAD Sensitivity (`backend/config.py`)
- **Changed**: VAD aggressiveness from 3 → 2 (less aggressive = more sensitive to speech)
- **Result**: More likely to detect speech, especially quiet speech

### 3. Audio Validation (`backend/server.py`)
- **Added**: Buffer size validation (must be multiple of 2 bytes)
- **Added**: Audio range clipping to ensure valid [-1, 1] range
- **Added**: Diagnostic logging for first 10 chunks

### 4. Better Logging (`backend/server.py`, `backend/audio/pipeline.py`)
- **Changed**: Reduced verbose logging (INFO → DEBUG for routine messages)
- **Added**: Debug logging for first 10 chunks to diagnose issues
- **Result**: Less log spam, better diagnostics

### 5. HTTP Option (`start_http.sh`)
- **Created**: Simple HTTP startup script for development
- **Note**: HTTPS is still recommended for microphone access in browsers

## About SSL/HTTPS

**Why SSL is needed:**
- Modern browsers (Chrome, Firefox, Edge) require HTTPS for microphone access
- This is a security feature to prevent malicious websites from accessing your microphone
- **Exception**: `localhost` and `127.0.0.1` may work with HTTP in some browsers

**Options:**
1. **Use HTTPS** (recommended): `./start_https.sh`
   - Requires self-signed certificate (browser will show warning)
   - Works for remote access

2. **Use HTTP for localhost**: `./start_http.sh`
   - Only works on localhost
   - May not work in all browsers

3. **Use a tunnel** (ngrok, cloudflare tunnel, etc.)
   - Provides HTTPS endpoint
   - Good for testing remote access

## Testing

After applying these fixes:

1. **Restart the server**: `./start_https.sh` or `./start_http.sh`

2. **Check logs**: You should see:
   - `[DEBUG] Chunk X: ... samples, level=..., is_speech=..., vad_prob=...` for first 10 chunks
   - `[VAD] 🟢 Speech detected` when speech is detected
   - `🔔 Speech event callback: speech_start` when speech starts
   - `📝 ✅ Emitted transcription` when transcription completes

3. **Test with speech**: Speak clearly into the microphone
   - VAD should detect speech (check logs)
   - STT should transcribe after speech ends

## Next Steps if Still Not Working

1. **Check audio levels**: Look for `level=` in debug logs - should be > 0.01 for normal speech
2. **Check VAD probability**: Should be 1.0 when speech is detected
3. **Check microphone**: Ensure microphone is working and not muted
4. **Check browser console**: Look for any JavaScript errors
5. **Try different VAD aggressiveness**: Can adjust in `backend/config.py` (0-3, lower = more sensitive)
