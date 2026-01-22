"""FastAPI server with WebSocket support for real-time audio processing"""
import base64
import json
import logging
import os
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

from backend.audio.pipeline import AudioPipeline, SpeechState
from backend.config import AudioConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Factory STT/TTS Server")

# Initialize audio pipeline
config = AudioConfig()
active_connections = []

# Store speech events per connection
connection_queues = {}

def speech_event_callback(event_type: str, data: dict):
    """Handle speech events from pipeline - store for websocket delivery"""
    logger.info(f"Speech event: {event_type}")
    # Store event to be sent to all active connections
    event_data = {
        "type": "speech_event",
        "event": event_type,
        "data": {k: v for k, v in data.items() if k != "audio_segment"}
    }
    # Add to each connection's queue
    for ws in active_connections:
        if ws not in connection_queues:
            connection_queues[ws] = []
        connection_queues[ws].append(event_data)

audio_pipeline = AudioPipeline(config, event_callback=speech_event_callback)

@app.get("/")
async def get_index():
    """Serve the frontend HTML"""
    with open("frontend/index.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    websocket.client_state = "open"  # Track connection state
    active_connections.append(websocket)
    connection_queues[websocket] = []  # Initialize queue for this connection
    logger.info(f"New WebSocket connection established (total: {len(active_connections)})")
    
    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            msg_type = msg.get("type")
            
            if msg_type == "audio_chunk":
                # Decode and process audio
                audio_bytes = base64.b64decode(msg.get("audio"))
                audio_float = (np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0)
                
                # Optional reference audio for AEC
                reference_audio = None
                if msg.get("reference_audio"):
                    ref_bytes = base64.b64decode(msg.get("reference_audio"))
                    reference_audio = (np.frombuffer(ref_bytes, dtype=np.int16).astype(np.float32) / 32768.0)
                
                import time
                chunk_start_time = time.time()
                
                processed = audio_pipeline.process_chunk(audio_float, reference_audio=reference_audio)
                
                processing_time = (time.time() - chunk_start_time) * 1000  # Convert to ms
                
                # Get VAD probability and metrics (stored by pipeline)
                vad_prob = getattr(audio_pipeline, '_last_vad_prob', 0.0)
                audio_metrics = getattr(audio_pipeline, '_last_audio_metrics', {})
                
                # Get current speech state
                speech_state = audio_pipeline.speech_state.value
                
                # Calculate input audio level
                audio_rms = np.sqrt(np.mean(audio_float ** 2))
                input_db = 20 * np.log10(audio_rms) if audio_rms > 0 else -float('inf')
                
                # Get processing metrics from pipeline
                input_db_metric = audio_metrics.get('input_db', input_db)
                denoised_db = audio_metrics.get('denoised_db', input_db)
                normalized_db = audio_metrics.get('normalized_db', input_db)
                ns_improvement = audio_metrics.get('ns_improvement', 0.0)
                gate_passed = audio_metrics.get('gate_passed', True)
                vad_reason = audio_metrics.get('vad_reason', 'UNKNOWN')
                
                # Determine VAD decision
                is_speech = vad_prob > 0.5
                vad_decision = "SPEECH" if is_speech else "NOISE"
                
                # Calculate output level if processed
                if processed is not None:
                    output_rms = np.sqrt(np.mean(processed ** 2))
                    output_db = 20 * np.log10(output_rms) if output_rms > 0 else -float('inf')
                else:
                    output_db = -float('inf')
                
                # Log speech state changes with detailed info
                if not hasattr(audio_pipeline, '_last_speech_state'):
                    audio_pipeline._last_speech_state = speech_state
                elif audio_pipeline._last_speech_state != speech_state:
                    logger.info(f"[STATE] {audio_pipeline._last_speech_state} → {speech_state} | "
                              f"VAD: {vad_prob:.3f} ({vad_decision}, {vad_reason}) | "
                              f"Audio: In={input_db_metric:.1f}dB → Denoised={denoised_db:.1f}dB (NS: {ns_improvement:+.1f}dB) → Norm={normalized_db:.1f}dB → Out={output_db:.1f}dB | "
                              f"Gate: {'PASS' if gate_passed else 'REJECT'} | "
                              f"Latency: {processing_time:.1f}ms")
                    audio_pipeline._last_speech_state = speech_state
                
                # Log detailed processing info (every 20 chunks for active speech, every 100 for silence)
                log_interval = 20 if processed is not None else 100
                if not hasattr(audio_pipeline, '_log_counter'):
                    audio_pipeline._log_counter = 0
                audio_pipeline._log_counter += 1
                
                if audio_pipeline._log_counter % log_interval == 0:
                    vad_decision = "SPEECH" if vad_prob > 0.5 else "NOISE"
                    status = "ACTIVE" if processed is not None else "SILENT"
                    logger.info(f"[{status}] State: {speech_state} | "
                              f"VAD: {vad_prob:.3f} ({vad_decision}, {vad_reason}) | "
                              f"Audio: In={input_db_metric:.1f}dB → Denoised={denoised_db:.1f}dB (NS: {ns_improvement:+.1f}dB) → Norm={normalized_db:.1f}dB → Out={output_db:.1f}dB | "
                              f"Gate: {'PASS' if gate_passed else 'REJECT'} | "
                              f"Latency: {processing_time:.1f}ms")
                
                # Send any pending speech events first (before processed_audio)
                if websocket in connection_queues:
                    while connection_queues[websocket]:
                        event = connection_queues[websocket].pop(0)
                        try:
                            await websocket.send_json(event)
                            logger.debug(f"Sent speech event: {event.get('event')}")
                        except Exception as e:
                            logger.error(f"Error sending speech event: {e}")
                            break  # Stop if connection is broken
                
                # Always send speech_state update immediately (even if no processed audio)
                # This ensures frontend gets state changes without delay
                if processed is not None:
                    processed_bytes = (processed * 32768.0).astype(np.int16).tobytes()
                    await websocket.send_json({
                        "type": "processed_audio",
                        "audio": base64.b64encode(processed_bytes).decode('utf-8'),
                        "has_speech": True,
                        "speech_state": speech_state,
                        "audio_level_db": round(input_db_metric, 2),
                        "vad_probability": round(vad_prob, 3)
                    })
                else:
                    # Send state update even when no speech (for immediate UI feedback)
                    await websocket.send_json({
                        "type": "processed_audio",
                        "has_speech": False,
                        "speech_state": speech_state,
                        "audio_level_db": round(input_db_metric, 2),
                        "vad_probability": round(vad_prob, 3)
                    })
            
            elif msg_type == "calibrate_noise":
                noise_bytes = base64.b64decode(msg.get("audio"))
                noise_float = (np.frombuffer(noise_bytes, dtype=np.int16).astype(np.float32) / 32768.0)
                audio_pipeline.calibrate_noise_profile(noise_float)
                await websocket.send_json({"type": "calibration_complete"})
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)
            logger.info(f"Removed client from active connections (remaining: {len(active_connections)})")
        if websocket in connection_queues:
            del connection_queues[websocket]
        if hasattr(websocket, 'client_state'):
            websocket.client_state = "closed"

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "connections": len(active_connections)}

@app.get("/api/config")
async def get_config():
        return {
            "sample_rate": config.SAMPLE_RATE,
            "vad_aggressiveness": getattr(config, 'VAD_AGGRESSIVENESS', 3),
            "vad_frame_ms": config.VAD_FRAME_MS,
            "noise_suppression_enabled": config.ENABLE_NOISE_SUPPRESSION,
            "noise_reduction_strength": config.NOISE_REDUCTION_STRENGTH
        }

# Mount static files (CSS, JS)
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

def run_server(host="0.0.0.0", port=8000, use_https=False, ssl_keyfile=None, ssl_certfile=None):
    if use_https:
        if not (ssl_keyfile and ssl_certfile and os.path.exists(ssl_keyfile) and os.path.exists(ssl_certfile)):
            logger.error("SSL certificates not found")
            return
        logger.info(f"Starting HTTPS server on {host}:{port}")
        uvicorn.run(app, host=host, port=port, ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
    else:
        logger.info(f"Starting HTTP server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()
