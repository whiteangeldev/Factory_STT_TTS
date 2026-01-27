"""Flask-SocketIO server for real-time STT/TTS"""
import eventlet
eventlet.monkey_patch()

import base64
import logging
import numpy as np
from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import os
import ssl

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger_env = logging.getLogger(__name__)
    logger_env.info("Loaded environment variables from .env file (if present)")
except ImportError:
    # python-dotenv not installed, skip loading .env
    pass

from .config import AudioConfig
from .audio.pipeline import AudioPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress SSL errors from eventlet (browsers rejecting self-signed certs)
# These are harmless - browsers reject certs but Socket.IO connections still work
import warnings
import sys
import io
warnings.filterwarnings('ignore', category=UserWarning)

# Suppress eventlet SSL error logging
logging.getLogger('eventlet').setLevel(logging.ERROR)
logging.getLogger('eventlet.wsgi').setLevel(logging.ERROR)
logging.getLogger('eventlet.hubs').setLevel(logging.ERROR)

# Suppress SSL errors in stderr (they're just browsers rejecting self-signed certs)
class SSLFilter:
    """Filter out SSL certificate errors from logs"""
    def filter(self, record):
        # Filter out SSL certificate unknown errors
        if 'SSLV3_ALERT_CERTIFICATE_UNKNOWN' in str(record.getMessage()):
            return False
        if 'ssl.SSLError' in str(record.getMessage()) and 'certificate' in str(record.getMessage()).lower():
            return False
        return True

# Add filter to root logger
logging.getLogger().addFilter(SSLFilter())

# Intercept stderr to filter SSL certificate error tracebacks
# These happen when browsers reject self-signed certs during HTTP handshake
# but Socket.IO WebSocket connections still work fine
_original_stderr = sys.stderr
_ssl_error_active = False

class FilteredStderr:
    """Filter stderr to suppress SSL certificate error tracebacks"""
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    
    def write(self, text):
        global _ssl_error_active
        text_str = str(text)
        
        # Detect SSL certificate error
        if 'SSLV3_ALERT_CERTIFICATE_UNKNOWN' in text_str:
            _ssl_error_active = True
            return  # Suppress
        
        # If we're in an SSL error traceback, suppress traceback lines
        if _ssl_error_active:
            # Suppress traceback lines (File, Traceback, etc.)
            if (text_str.strip().startswith('Traceback') or
                text_str.strip().startswith('File ') or
                'eventlet' in text_str.lower() or
                'ssl.SSLError' in text_str or
                'Removing descriptor' in text_str or
                not text_str.strip()):
                return  # Suppress
            # If we see a normal log line (starts with timestamp or our logger format), reset
            if ' - ' in text_str and ('INFO' in text_str or 'ERROR' in text_str or 'WARNING' in text_str):
                _ssl_error_active = False
                # Write this line (it's a real log)
                self.original_stderr.write(text)
            else:
                return  # Still in traceback, suppress
        
        # Normal output
        self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()
    
    def __getattr__(self, name):
        return getattr(self.original_stderr, name)

# Replace stderr with filtered version (only suppress SSL cert errors)
# Note: This suppresses harmless SSL cert rejection errors from browsers
# The Socket.IO WebSocket connections still work fine
sys.stderr = FilteredStderr(_original_stderr)

# Initialize Flask app
app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend')
app.config['SECRET_KEY'] = 'factory-stt-tts-secret-key'
CORS(app)

# Initialize SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    logger=False,
    engineio_logger=False
)

# Global state
config = AudioConfig()
connected_clients = set()

# Per-client pipelines and state
client_pipelines = {}  # {client_id: AudioPipeline} - One pipeline per client
client_audio_buffers = {}  # {client_id: np.ndarray}
client_recording_state = {}  # {client_id: bool} - Track if client is recording
MIN_CHUNK_SIZE_FOR_VAD = 480  # 30ms at 16kHz
MIN_AUDIO_LEVEL_FOR_VAD = 0.0005  # Minimum audio level to consider for VAD (very low threshold - only filter out complete silence)

def speech_event_callback(event_type: str, data: dict):
    """Callback for speech events from pipeline"""
    logger.info(f"🔔 Speech event callback: {event_type}, data keys: {list(data.keys())}")
    
    if event_type == "transcription":
        event_data = {
            "text": data.get("text", ""),
            "language": data.get("language"),
            "confidence": float(data.get("confidence", 0.0)),
            "is_final": True
        }
        try:
            socketio.emit("transcription", event_data, namespace="/")
            logger.info(f"📝 ✅ Emitted transcription: '{event_data['text'][:50]}...'")
        except Exception as e:
            logger.error(f"❌ Error emitting transcription: {e}")
        return
    
    elif event_type == "transcription_interim":
        event_data = {
            "text": data.get("text", ""),
            "language": data.get("language"),
            "confidence": float(data.get("confidence", 0.0)),
            "is_final": False
        }
        try:
            socketio.emit("transcription_interim", event_data, namespace="/")
            logger.info(f"📝 ✅ Emitted transcription_interim: '{event_data['text'][:50]}...'")
        except Exception as e:
            logger.error(f"❌ Error emitting transcription_interim: {e}")
        return
    
    elif event_type == "transcription_processing":
        event_data = {
            "status": data.get("status", "processing"),
            "audio_duration": float(data.get("audio_duration", 0.0)),
            "error": data.get("error")
        }
        try:
            socketio.emit("transcription_processing", event_data, namespace="/")
            logger.info(f"📝 ✅ Emitted transcription_processing: {event_data['status']}")
        except Exception as e:
            logger.error(f"❌ Error emitting transcription_processing: {e}")
        return
    
    else:
        # Generic speech event
        event_data = {
            "type": "speech_event",
            "event": event_type,
            "data": {k: v for k, v in data.items() if k != "audio_segment"}
        }
        try:
            socketio.emit("speech_event", event_data, namespace="/")
            logger.info(f"🔊 ✅ Emitted speech_event: {event_type}")
        except Exception as e:
            logger.error(f"❌ Error emitting speech_event: {e}")

# Pipelines are created per-client (see handle_connect and handle_start_recording)

@app.route('/')
def index():
    """Serve main page"""
    frontend_dir = os.path.join(os.path.dirname(__file__), '../frontend')
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    static_dir = os.path.join(os.path.dirname(__file__), '../frontend/static')
    return send_from_directory(static_dir, path)

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    connected_clients.add(client_id)
    client_audio_buffers[client_id] = np.array([], dtype=np.float32)  # Initialize buffer
    client_recording_state[client_id] = False  # Not recording initially
    # Pipeline will be created when recording starts
    logger.info(f"✅ Client connected: {client_id} (total: {len(connected_clients)})")
    
    emit('connected', {
        'status': 'ready',
        'message': 'WebSocket ready - VAD and noise reduction active'
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    connected_clients.discard(client_id)
    client_audio_buffers.pop(client_id, None)  # Clean up buffer
    client_recording_state.pop(client_id, None)  # Clean up recording state
    client_pipelines.pop(client_id, None)  # Clean up pipeline
    logger.info(f"🔌 Client disconnected: {client_id} (remaining: {len(connected_clients)})")

@socketio.on('start_recording')
def handle_start_recording(data=None):
    """Handle recording start"""
    client_id = request.sid
    client_recording_state[client_id] = True
    client_audio_buffers[client_id] = np.array([], dtype=np.float32)  # Clear buffer on start
    
    # Create a fresh pipeline instance for this client
    # This ensures clean state for each recording session
    client_pipelines[client_id] = AudioPipeline(config, event_callback=speech_event_callback)
    client_pipelines[client_id].reset()  # Ensure clean state
    logger.info(f"🎙️ Recording started for client: {client_id} - new pipeline created and reset")
    
    emit('recording_status', {
        'is_recording': True,
        'status': 'Recording...'
    })

@socketio.on('stop_recording')
def handle_stop_recording(data=None):
    """Handle recording stop"""
    client_id = request.sid
    
    # Set recording state to False IMMEDIATELY to stop processing any incoming chunks
    client_recording_state[client_id] = False
    
    # Clear audio buffer to discard any pending audio
    client_audio_buffers[client_id] = np.array([], dtype=np.float32)
    
    # Remove pipeline for this client (will be recreated on next start)
    client_pipelines.pop(client_id, None)
    
    logger.info(f"🛑 Recording stopped for client: {client_id} - pipeline removed, audio processing disabled")
    
    emit('recording_status', {
        'is_recording': False,
        'status': 'Ready'
    })

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    """Handle incoming audio chunk - only process if client is recording"""
    client_id = request.sid
    
    # Only process audio if client is recording
    if not client_recording_state.get(client_id, False):
        # Silently ignore audio chunks when not recording
        # Log occasionally to detect if chunks are still arriving after stop
        if not hasattr(handle_audio_chunk, '_ignored_count'):
            handle_audio_chunk._ignored_count = {}
        if client_id not in handle_audio_chunk._ignored_count:
            handle_audio_chunk._ignored_count[client_id] = 0
        handle_audio_chunk._ignored_count[client_id] += 1
        if handle_audio_chunk._ignored_count[client_id] <= 3:
            logger.warning(f"⚠️ Ignoring audio chunk from client {client_id} - not recording (chunk #{handle_audio_chunk._ignored_count[client_id]})")
        return
    
    # Reset ignored count when recording is active
    if hasattr(handle_audio_chunk, '_ignored_count'):
        handle_audio_chunk._ignored_count.pop(client_id, None)
    
    try:
        # Extract audio data
        if isinstance(data, dict):
            audio_base64 = data.get('audio', '')
        elif isinstance(data, str):
            audio_base64 = data
        else:
            logger.error(f"Invalid audio chunk format: {type(data)}")
            return
        
        if not audio_base64:
            return
        
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(audio_base64)
            
            # Validate buffer size (must be multiple of 2 bytes for int16)
            if len(audio_bytes) % 2 != 0:
                logger.warning(f"Invalid audio buffer size: {len(audio_bytes)} bytes (must be multiple of 2). Skipping chunk.")
                return
            
            if len(audio_bytes) == 0:
                return
            
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Debug first few chunks
            if not hasattr(handle_audio_chunk, '_decode_count'):
                handle_audio_chunk._decode_count = 0
            handle_audio_chunk._decode_count += 1
            if handle_audio_chunk._decode_count <= 5:
                int16_max = np.abs(audio_int16).max()
                int16_mean = np.abs(audio_int16).mean()
                logger.info(f"[Decode Debug {handle_audio_chunk._decode_count}] Int16: {len(audio_int16)} samples, max={int16_max}, mean_abs={int16_mean:.2f}")
            
            # Convert to float32 [-1, 1]
            audio_float = audio_int16.astype(np.float32) / 32768.0
            
            # Ensure audio is in valid range [-1, 1]
            audio_float = np.clip(audio_float, -1.0, 1.0)
            
            # Check if audio is actually silent (all zeros or very quiet)
            audio_max = np.abs(audio_float).max()
            audio_mean = np.abs(audio_float).mean()
            
            if handle_audio_chunk._decode_count <= 5:
                logger.info(f"[Decode Debug {handle_audio_chunk._decode_count}] Float: max={audio_max:.6f}, mean_abs={audio_mean:.6f}")
            
            if audio_max < 1e-6:
                if handle_audio_chunk._decode_count <= 10:
                    logger.warning(f"Received silent audio chunk: {len(audio_float)} samples, max={audio_max:.6f}, int16_max={np.abs(audio_int16).max()}")
                return  # Skip processing silent chunks
            
            # Log chunk info occasionally (every 100 chunks to avoid spam)
            if not hasattr(handle_audio_chunk, '_chunk_count'):
                handle_audio_chunk._chunk_count = 0
            handle_audio_chunk._chunk_count += 1
            if handle_audio_chunk._chunk_count % 100 == 0:
                logger.debug(f"Audio chunk: {len(audio_float)} samples, range: [{audio_float.min():.3f}, {audio_float.max():.3f}], max_abs={audio_max:.4f}")
        except Exception as e:
            logger.error(f"Error decoding audio: {e}")
            return
        
        # Buffer audio chunks until we have enough for VAD
        # VAD needs at least 480 samples (30ms at 16kHz), but we're getting 171 sample chunks
        client_id = request.sid
        if client_id not in client_audio_buffers:
            client_audio_buffers[client_id] = np.array([], dtype=np.float32)
        
        # Add new chunk to buffer
        client_audio_buffers[client_id] = np.concatenate([client_audio_buffers[client_id], audio_float])
        
        # Process when we have enough samples, or if buffer is getting too large
        max_buffer_size = config.SAMPLE_RATE * 0.5  # Max 500ms buffer
        if len(client_audio_buffers[client_id]) >= MIN_CHUNK_SIZE_FOR_VAD:
            # Process accumulated buffer
            audio_to_process = client_audio_buffers[client_id].copy()
            # Keep remainder if buffer is larger than needed
            if len(audio_to_process) > MIN_CHUNK_SIZE_FOR_VAD * 2:
                # Process in chunks of MIN_CHUNK_SIZE_FOR_VAD, keep remainder
                process_size = (len(audio_to_process) // MIN_CHUNK_SIZE_FOR_VAD) * MIN_CHUNK_SIZE_FOR_VAD
                audio_to_process = audio_to_process[:process_size]
                client_audio_buffers[client_id] = client_audio_buffers[client_id][process_size:]
            else:
                # Process all and clear buffer
                client_audio_buffers[client_id] = np.array([], dtype=np.float32)
            
            # Get client's pipeline
            if client_id not in client_pipelines:
                logger.warning(f"⚠️ No pipeline for client {client_id}, creating one")
                try:
                    client_pipelines[client_id] = AudioPipeline(config, event_callback=speech_event_callback)
                except Exception as e:
                    logger.error(f"❌ Failed to create pipeline for client {client_id}: {e}")
                    # Don't keep retrying - mark this client as failed
                    client_pipelines[client_id] = None
                    return
            
            audio_pipeline = client_pipelines[client_id]
            if audio_pipeline is None:
                # Pipeline creation failed, skip processing
                return
            
            # Check audio level - only filter out complete silence
            audio_level = np.abs(audio_to_process).max()
            if audio_level < MIN_AUDIO_LEVEL_FOR_VAD:
                # Audio too quiet (complete silence), skip processing
                processed = None
                input_db = 20 * np.log10(audio_level + 1e-10)
                emit('processed_audio', {
                    'audio': '',
                    'has_speech': False,
                    'speech_state': 'silence',
                    'audio_level_db': float(round(input_db, 2)),
                    'vad_probability': 0.0
                })
                return
            
            # Log audio level for debugging (first few chunks)
            if not hasattr(handle_audio_chunk, '_process_count'):
                handle_audio_chunk._process_count = {}
            if client_id not in handle_audio_chunk._process_count:
                handle_audio_chunk._process_count[client_id] = 0
            handle_audio_chunk._process_count[client_id] += 1
            if handle_audio_chunk._process_count[client_id] <= 10:
                logger.info(f"[Process] Client {client_id[:8]}... chunk #{handle_audio_chunk._process_count[client_id]}: {len(audio_to_process)} samples, level={audio_level:.4f}, dB={20*np.log10(audio_level+1e-10):.1f}")
            
            # Process through pipeline
            processed = audio_pipeline.process_chunk(audio_to_process)
            
            # Calculate metrics on the processed chunk
            input_db = 20 * np.log10(np.abs(audio_to_process).max() + 1e-10)
            vad_prob = audio_pipeline.vad.get_probability(audio_to_process)
            is_speech = audio_pipeline.vad.is_speech(audio_to_process)
            speech_state = "speech" if audio_pipeline.is_speaking else "silence"
        elif len(client_audio_buffers[client_id]) > max_buffer_size:
            # Buffer too large, process what we have
            audio_to_process = client_audio_buffers[client_id].copy()
            client_audio_buffers[client_id] = np.array([], dtype=np.float32)
            
            # Get client's pipeline
            if client_id not in client_pipelines:
                logger.warning(f"⚠️ No pipeline for client {client_id}, creating one")
                try:
                    client_pipelines[client_id] = AudioPipeline(config, event_callback=speech_event_callback)
                except Exception as e:
                    logger.error(f"❌ Failed to create pipeline for client {client_id}: {e}")
                    # Don't keep retrying - mark this client as failed
                    client_pipelines[client_id] = None
                    return
            
            audio_pipeline = client_pipelines[client_id]
            if audio_pipeline is None:
                # Pipeline creation failed, skip processing
                return
            
            # Check audio level
            audio_level = np.abs(audio_to_process).max()
            if audio_level < MIN_AUDIO_LEVEL_FOR_VAD:
                processed = None
                input_db = 20 * np.log10(audio_level + 1e-10)
                emit('processed_audio', {
                    'audio': '',
                    'has_speech': False,
                    'speech_state': 'silence',
                    'audio_level_db': float(round(input_db, 2)),
                    'vad_probability': 0.0
                })
                return
            
            processed = audio_pipeline.process_chunk(audio_to_process)
            
            # Calculate metrics
            input_db = 20 * np.log10(np.abs(audio_to_process).max() + 1e-10)
            vad_prob = audio_pipeline.vad.get_probability(audio_to_process)
            is_speech = audio_pipeline.vad.is_speech(audio_to_process)
            speech_state = "speech" if audio_pipeline.is_speaking else "silence"
        else:
            # Not enough samples yet, skip processing but send metrics for UI feedback
            processed = None
            input_db = 20 * np.log10(np.abs(audio_float).max() + 1e-10)
            emit('processed_audio', {
                'audio': '',  # No processed audio yet
                'has_speech': False,
                'speech_state': 'buffering',
                'audio_level_db': float(round(input_db, 2)),
                'vad_probability': 0.0
            })
            return
        
        # Log VAD status (only on state changes or occasionally during speech)
        # Track previous state to detect changes
        if not hasattr(handle_audio_chunk, '_prev_speech_state'):
            handle_audio_chunk._prev_speech_state = {}
        if not hasattr(handle_audio_chunk, '_speech_log_counter'):
            handle_audio_chunk._speech_log_counter = {}
        
        prev_state = handle_audio_chunk._prev_speech_state.get(client_id, None)
        handle_audio_chunk._prev_speech_state[client_id] = speech_state
        
        # Log on state change or every 50 chunks during speech
        if prev_state != speech_state:
            # State changed
            if is_speech:
                logger.info(f"[VAD] 🟢 Speech START - Samples: {len(audio_to_process)}, VAD: {vad_prob:.3f}, Input: {input_db:.1f}dB")
            elif prev_state == "speech":
                logger.info(f"[VAD] 🔴 Speech END - VAD: {vad_prob:.3f}, Input: {input_db:.1f}dB")
        elif speech_state == "speech":
            # Log occasionally during continuous speech (every 50 chunks)
            if client_id not in handle_audio_chunk._speech_log_counter:
                handle_audio_chunk._speech_log_counter[client_id] = 0
            handle_audio_chunk._speech_log_counter[client_id] += 1
            if handle_audio_chunk._speech_log_counter[client_id] % 50 == 0:
                logger.debug(f"[VAD] 🟢 Speech ongoing - Chunks: {handle_audio_chunk._speech_log_counter[client_id]}, VAD: {vad_prob:.3f}, Input: {input_db:.1f}dB")
        
        # Send processed audio back (optional)
        if processed is not None:
            processed_bytes = (processed * 32768.0).astype(np.int16).tobytes()
            emit('processed_audio', {
                'audio': base64.b64encode(processed_bytes).decode('utf-8'),
                'has_speech': bool(is_speech),
                'speech_state': str(speech_state),
                'audio_level_db': float(round(input_db, 2)),
                'vad_probability': float(round(vad_prob, 3))
            })
    
    except Exception as e:
        logger.error(f"❌ Error processing audio chunk: {e}", exc_info=True)

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 8000))
    
    logger.info(f"Starting HTTPS server on {host}:{port}")
    
    # SSL certificates
    certfile = os.path.join(os.path.dirname(__file__), '../certs/cert.pem')
    keyfile = os.path.join(os.path.dirname(__file__), '../certs/key.pem')
    
    if os.path.exists(certfile) and os.path.exists(keyfile):
        socketio.run(app, host=host, port=port, certfile=certfile, keyfile=keyfile)
    else:
        logger.warning("SSL certificates not found, running without HTTPS")
        socketio.run(app, host=host, port=port)
