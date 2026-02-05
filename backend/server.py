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
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .config import AudioConfig
from .audio.pipeline import AudioPipeline
from .audio.system_audio import SystemAudioCapture

# Try to import TTS (optional - server can run without it)
try:
    from .audio.tts import synthesize_speech
    _HAS_TTS = True
except (ImportError, RuntimeError, PermissionError) as e:
    # Logger not yet defined, use print for early import errors
    print(f"Warning: TTS not available: {e}. TTS feature will be disabled.")
    _HAS_TTS = False
    synthesize_speech = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress SSL errors and eventlet noise
logging.getLogger('eventlet').setLevel(logging.CRITICAL)
logging.getLogger('eventlet.wsgi').setLevel(logging.CRITICAL)
logging.getLogger('eventlet.hubs').setLevel(logging.CRITICAL)
logging.getLogger('eventlet.greenthread').setLevel(logging.CRITICAL)

class SSLFilter:
    def filter(self, record):
        msg = str(record.getMessage())
        # Filter out all SSL-related errors that are harmless
        ssl_patterns = [
            'SSLV3_ALERT_CERTIFICATE_UNKNOWN', '[SSL: HTTP_REQUEST]', 
            'HTTP_REQUEST', 'ssl.SSLError', 'Bad request version',
            'code 400, message Bad request'
        ]
        return not any(pattern in msg for pattern in ssl_patterns)

logging.getLogger().addFilter(SSLFilter())

# Filter stderr for SSL errors (suppress harmless HTTP_REQUEST errors)
_ssl_error_active = False
_ssl_error_lines = 0
_original_stderr = sys.stderr

class FilteredStderr:
    def __init__(self, original):
        self.original = original
    
    def write(self, text):
        global _ssl_error_active, _ssl_error_lines
        text_str = str(text)
        
        # Detect SSL errors (various patterns)
        ssl_error_patterns = [
            '[SSL: HTTP_REQUEST]', 'HTTP_REQUEST', 'ssl.SSLError',
            'SSLV3_ALERT_CERTIFICATE_UNKNOWN', 'Bad request version',
            'code 400, message Bad request', '_ssl.c:', 'recv_into',
            'ssl.py', 'green/ssl.py'
        ]
        
        # Detect eventlet traceback patterns that indicate SSL errors
        # These patterns appear in tracebacks from SSL mismatches
        eventlet_ssl_patterns = [
            'eventlet/wsgi.py', 'eventlet/hubs', 'eventlet/greenthread',
            '_read_request_line', 'readline', 'handle_one_request',
            'process_request', 'protocol', 'wait', 'cb(fileno)',
            'kqueue.py', 'greenthread.py', 'wsgi.py', 'eventlet'
        ]
        
        # Check if this contains an SSL error
        is_ssl_error = any(pattern in text_str for pattern in ssl_error_patterns)
        
        # Check if this is a traceback from eventlet that's likely SSL-related
        # We suppress tracebacks that involve eventlet + request handling (common SSL error pattern)
        is_eventlet_traceback = (
            ('Traceback' in text_str) or  # Any traceback
            ('File "' in text_str and any(p in text_str for p in eventlet_ssl_patterns)) or
            ('line ' in text_str and any(p in text_str for p in eventlet_ssl_patterns))
        )
        
        # Activate filter if we see SSL error or eventlet traceback
        # In HTTPS mode, eventlet tracebacks during request handling are usually SSL mismatches
        if is_ssl_error or is_eventlet_traceback:
            _ssl_error_active = True
            _ssl_error_lines = 0
            return  # Suppress this line
        
        # If we're in an SSL error traceback, suppress traceback lines
        if _ssl_error_active:
            _ssl_error_lines += 1
            
            # Suppress traceback components (comprehensive list)
            suppress_patterns = [
                'Traceback', 'File "', 'File ', 'eventlet', 'ssl.SSLError',
                'Removing descriptor', 'HTTP_REQUEST', '_ssl.c:', 'socket.py',
                'green/ssl.py', 'wsgi.py', 'kqueue.py', 'greenthread.py',
                'recv_into', 'read', '_read_request_line', 'handle_one_request',
                'process_request', 'protocol', 'wait', 'cb(fileno)', 'code 400',
                'readline', 'handle', '__init__', 'main', 'result = function'
            ]
            
            # Suppress if it matches any pattern, is empty, or looks like traceback
            is_traceback_line = (
                any(pattern in text_str for pattern in suppress_patterns) or
                not text_str.strip() or
                text_str.strip().startswith('File ') or
                'line ' in text_str and ('eventlet' in text_str or 'ssl' in text_str.lower())
            )
            
            if is_traceback_line:
                # Reset after 30 lines (traceback should be done by then)
                if _ssl_error_lines > 30:
                    _ssl_error_active = False
                    _ssl_error_lines = 0
                return
            
            # Safety: reset if we've processed many lines without seeing traceback patterns
            if _ssl_error_lines > 30:
                _ssl_error_active = False
                _ssl_error_lines = 0
            
            # If we see a normal log line (our format), reset and show it
            if ' - ' in text_str and any(x in text_str for x in ['INFO', 'ERROR', 'WARNING', 'DEBUG']):
                _ssl_error_active = False
                _ssl_error_lines = 0
                self.original.write(text)
                return
            
            # Still in traceback, suppress
            return
        
        # Normal output
        self.original.write(text)
    
    def flush(self):
        self.original.flush()
    
    def __getattr__(self, name):
        return getattr(self.original, name)

sys.stderr = FilteredStderr(_original_stderr)

app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend')
app.config['SECRET_KEY'] = 'factory-stt-tts-secret-key'
CORS(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet", logger=False, engineio_logger=False)

config = AudioConfig()
connected_clients = set()
client_pipelines = {}
client_audio_buffers = {}
client_recording_state = {}
client_system_audio = {}  # Server-side system audio capture per client
client_audio_queues = {}  # Queues for passing audio from background threads to eventlet
MIN_CHUNK_SIZE = 480
MIN_AUDIO_LEVEL = 0.0005

def speech_event_callback(event_type: str, data: dict):
    try:
        if event_type == "transcription":
            socketio.emit("transcription", {
                "text": data.get("text", ""),
                "language": data.get("language"),
                "confidence": float(data.get("confidence", 0.0)),
                "is_final": True
            })
        elif event_type == "transcription_interim":
            socketio.emit("transcription_interim", {
                "text": data.get("text", ""),
                "language": data.get("language"),
                "confidence": float(data.get("confidence", 0.0)),
                "is_final": False
            })
        else:
            socketio.emit("speech_event", {
                "type": "speech_event",
                "event": event_type,
                "data": {k: v for k, v in data.items() if k != "audio_segment"}
            })
    except Exception as e:
        logger.error(f"Error emitting event: {e}")

@app.route('/')
def index():
    return send_from_directory(os.path.join(os.path.dirname(__file__), '../frontend'), 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(os.path.join(os.path.dirname(__file__), '../frontend/static'), path)

@socketio.on('connect')
def handle_connect():
    client_id = request.sid
    connected_clients.add(client_id)
    client_audio_buffers[client_id] = np.array([], dtype=np.float32)
    client_recording_state[client_id] = False
    logger.info(f"✅ Client connected: {client_id} (total: {len(connected_clients)})")
    emit('connected', {'status': 'ready', 'message': 'WebSocket ready'})

@socketio.on('disconnect')
def handle_disconnect():
    client_id = request.sid
    connected_clients.discard(client_id)
    client_audio_buffers.pop(client_id, None)
    client_recording_state.pop(client_id, None)
    client_pipelines.pop(client_id, None)
    
    # Stop server-side system audio if active
    if client_id in client_system_audio:
        client_system_audio[client_id].stop()
        client_system_audio.pop(client_id, None)
    
    logger.info(f"🔌 Client disconnected: {client_id} (remaining: {len(connected_clients)})")

@socketio.on('start_recording')
def handle_start_recording(data=None):
    client_id = request.sid
    input_mode = data.get('input_mode', 'microphone') if isinstance(data, dict) else 'microphone'
    
    client_recording_state[client_id] = True
    client_audio_buffers[client_id] = np.array([], dtype=np.float32)
    
    # Create client-specific callback that emits to the right client
    def client_speech_callback(event_type: str, data: dict):
        """Client-specific callback that emits to the correct client"""
        try:
            if event_type == "transcription":
                socketio.emit("transcription", {
                    "text": data.get("text", ""),
                    "language": data.get("language"),
                    "confidence": float(data.get("confidence", 0.0)),
                    "is_final": True
                }, room=client_id)
            elif event_type == "transcription_interim":
                socketio.emit("transcription_interim", {
                    "text": data.get("text", ""),
                    "language": data.get("language"),
                    "confidence": float(data.get("confidence", 0.0)),
                    "is_final": False
                }, room=client_id)
            else:
                socketio.emit("speech_event", {
                    "type": "speech_event",
                    "event": event_type,
                    "data": {k: v for k, v in data.items() if k != "audio_segment"}
                }, room=client_id)
        except Exception as e:
            logger.error(f"Error emitting event to {client_id}: {e}")
    
    client_pipelines[client_id] = AudioPipeline(config, event_callback=client_speech_callback)
    client_pipelines[client_id].reset()
    
    # System audio: always use server-side capture
    if input_mode == 'system':
        try:
            # Use a queue to pass audio from background thread to eventlet context
            import queue
            audio_event_queue = queue.Queue(maxsize=100)  # Limit queue size
            client_audio_queues[client_id] = audio_event_queue
            
            def on_audio_chunk(audio_data):
                """Callback for server-side system audio - queues for eventlet processing"""
                if not client_recording_state.get(client_id, False):
                    return
                # Queue the audio data for processing in eventlet context
                try:
                    audio_event_queue.put_nowait((client_id, audio_data))
                except queue.Full:
                    pass  # Drop if queue is full (backpressure)
            
            # Start background task to process queued audio in eventlet context
            def process_audio_queue():
                while client_recording_state.get(client_id, False) and client_id in client_audio_queues:
                    try:
                        queued_client_id, audio_data = audio_event_queue.get(timeout=0.1)
                        if queued_client_id in client_pipelines and client_pipelines[queued_client_id]:
                            _process_audio_chunk(queued_client_id, audio_data)
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logger.error(f"Error processing queued audio: {e}")
            
            socketio.start_background_task(process_audio_queue)
            
            system_capture = SystemAudioCapture(
                sample_rate=config.SAMPLE_RATE,
                chunk_size=MIN_CHUNK_SIZE,
                on_audio=on_audio_chunk
            )
            if system_capture.start():
                client_system_audio[client_id] = system_capture
                logger.info(f"🎙️ System audio started: {client_id}")
                emit('recording_status', {'is_recording': True, 'status': 'Recording system audio...'})
            else:
                logger.error(f"⚠️ Failed to start system audio for {client_id}")
                emit('recording_status', {'is_recording': False, 'status': 'Failed to start system audio. Check server logs.'})
        except Exception as e:
            logger.error(f"Failed to start system audio: {e}")
            emit('recording_status', {'is_recording': False, 'status': f'System audio error: {str(e)}'})
    else:
        # Microphone mode: browser sends audio
        logger.info(f"🎙️ Microphone recording started: {client_id}")
        emit('recording_status', {'is_recording': True, 'status': 'Recording...'})

@socketio.on('stop_recording')
def handle_stop_recording(data=None):
    client_id = request.sid
    client_recording_state[client_id] = False
    client_audio_buffers[client_id] = np.array([], dtype=np.float32)
    
    # Stop server-side system audio if active
    if client_id in client_system_audio:
        client_system_audio[client_id].stop()
        client_system_audio.pop(client_id, None)
        logger.info(f"🛑 Server-side system audio stopped: {client_id}")
    
    # Clean up audio queue
    if client_id in client_audio_queues:
        # Clear remaining items
        queue = client_audio_queues.pop(client_id, None)
        if queue:
            while not queue.empty():
                try:
                    queue.get_nowait()
                except:
                    break
    
    client_pipelines.pop(client_id, None)
    logger.info(f"🛑 Recording stopped: {client_id}")
    emit('recording_status', {'is_recording': False, 'status': 'Ready'})

def _process_audio_chunk(client_id, audio_to_process):
    """Process audio chunk through pipeline"""
    if client_id not in client_pipelines:
        try:
            # Create a client-specific callback
            def client_speech_callback(event_type: str, data: dict):
                """Client-specific callback that emits to the correct client"""
                try:
                    if event_type == "transcription":
                        socketio.emit("transcription", {
                            "text": data.get("text", ""),
                            "language": data.get("language"),
                            "confidence": float(data.get("confidence", 0.0)),
                            "is_final": True
                        }, room=client_id)
                    elif event_type == "transcription_interim":
                        socketio.emit("transcription_interim", {
                            "text": data.get("text", ""),
                            "language": data.get("language"),
                            "confidence": float(data.get("confidence", 0.0)),
                            "is_final": False
                        }, room=client_id)
                    else:
                        socketio.emit("speech_event", {
                            "type": "speech_event",
                            "event": event_type,
                            "data": {k: v for k, v in data.items() if k != "audio_segment"}
                        }, room=client_id)
                except Exception as e:
                    logger.error(f"Error emitting event to {client_id}: {e}")
            
            client_pipelines[client_id] = AudioPipeline(config, event_callback=client_speech_callback)
        except Exception as e:
            logger.error(f"Failed to create pipeline: {e}")
            client_pipelines[client_id] = None
            return None
    
    pipeline = client_pipelines[client_id]
    if pipeline is None:
        return None
    
    # Check recording state
    if not client_recording_state.get(client_id, False):
        return None
    
    audio_level = np.abs(audio_to_process).max()
    input_db = 20 * np.log10(audio_level + 1e-10)
    
    # Log first few chunks to debug
    if not hasattr(_process_audio_chunk, '_debug_count'):
        _process_audio_chunk._debug_count = {}
    if client_id not in _process_audio_chunk._debug_count:
        _process_audio_chunk._debug_count[client_id] = 0
    
    if _process_audio_chunk._debug_count[client_id] < 5:
        _process_audio_chunk._debug_count[client_id] += 1
        logger.info(f"[Audio Debug {_process_audio_chunk._debug_count[client_id]}] Client {client_id}: level={audio_level:.6f}, dB={input_db:.2f}, samples={len(audio_to_process)}")
    
    # Process through pipeline (VAD, noise reduction, STT)
    if audio_level < MIN_AUDIO_LEVEL:
        # Below threshold - send original audio for saving, but mark as silence
        audio_bytes = (audio_to_process * 32768.0).astype(np.int16).tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Debug: Log emission attempt
        if not hasattr(_process_audio_chunk, '_emit_count'):
            _process_audio_chunk._emit_count = {}
        if client_id not in _process_audio_chunk._emit_count:
            _process_audio_chunk._emit_count[client_id] = 0
        _process_audio_chunk._emit_count[client_id] += 1
        
        if _process_audio_chunk._emit_count[client_id] <= 3:
            logger.info(f"[Emit Debug {_process_audio_chunk._emit_count[client_id]}] Emitting processed_audio to {client_id[:8]}... (silence, {len(audio_b64)} bytes)")
        
        try:
            socketio.emit('processed_audio', {
                'audio': audio_b64,
                'has_speech': False, 'speech_state': 'silence',
                'audio_level_db': float(round(input_db, 2)), 'vad_probability': 0.0
            }, room=client_id)
        except Exception as e:
            logger.error(f"Error emitting processed_audio: {e}")
        return None
    
    processed = pipeline.process_chunk(audio_to_process)
    input_db = 20 * np.log10(np.abs(audio_to_process).max() + 1e-10)
    vad_prob = pipeline.vad.get_probability(audio_to_process)
    is_speech = pipeline.vad.is_speech(audio_to_process)
    speech_state = "speech" if pipeline.is_speaking else "silence"
    
    # Always send audio for saving (use processed if available, otherwise original)
    audio_to_save = processed if processed is not None else audio_to_process
    audio_bytes = (audio_to_save * 32768.0).astype(np.int16).tobytes()
    audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    # Debug: Log emission attempt
    if not hasattr(_process_audio_chunk, '_emit_count'):
        _process_audio_chunk._emit_count = {}
    if client_id not in _process_audio_chunk._emit_count:
        _process_audio_chunk._emit_count[client_id] = 0
    _process_audio_chunk._emit_count[client_id] += 1
    
    if _process_audio_chunk._emit_count[client_id] <= 3:
        logger.info(f"[Emit Debug {_process_audio_chunk._emit_count[client_id]}] Emitting processed_audio to {client_id[:8]}... (has_speech={is_speech}, {len(audio_b64)} bytes)")
    
    try:
        socketio.emit('processed_audio', {
            'audio': audio_b64,
            'has_speech': bool(is_speech), 'speech_state': str(speech_state),
            'audio_level_db': float(round(input_db, 2)), 'vad_probability': float(round(vad_prob, 3))
        }, room=client_id)
    except Exception as e:
        logger.error(f"Error emitting processed_audio: {e}")
    
    return processed

@socketio.on('audio_chunk')
def handle_audio_chunk(data):
    client_id = request.sid
    # Skip browser audio if system audio is active (server handles it)
    if client_id in client_system_audio:
        return
    if not client_recording_state.get(client_id, False):
        return
    
    # Skip browser audio if server-side system audio is active
    if client_id in client_system_audio:
        return  # Server is handling audio capture
    
    try:
        audio_base64 = data.get('audio', '') if isinstance(data, dict) else (data if isinstance(data, str) else '')
        if not audio_base64:
            return
        
        audio_bytes = base64.b64decode(audio_base64)
        if len(audio_bytes) % 2 != 0 or len(audio_bytes) == 0:
            return
        
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = np.clip(audio_int16.astype(np.float32) / 32768.0, -1.0, 1.0)
        
        if np.abs(audio_float).max() < 1e-6:
            return
        
        if client_id not in client_audio_buffers:
            client_audio_buffers[client_id] = np.array([], dtype=np.float32)
        
        client_audio_buffers[client_id] = np.concatenate([client_audio_buffers[client_id], audio_float])
        
        max_buffer_size = config.SAMPLE_RATE * 0.5
        if len(client_audio_buffers[client_id]) >= MIN_CHUNK_SIZE:
            audio_to_process = client_audio_buffers[client_id].copy()
            if len(audio_to_process) > MIN_CHUNK_SIZE * 2:
                process_size = (len(audio_to_process) // MIN_CHUNK_SIZE) * MIN_CHUNK_SIZE
                audio_to_process = audio_to_process[:process_size]
                client_audio_buffers[client_id] = client_audio_buffers[client_id][process_size:]
            else:
                client_audio_buffers[client_id] = np.array([], dtype=np.float32)
            
            _process_audio_chunk(client_id, audio_to_process)
        elif len(client_audio_buffers[client_id]) > max_buffer_size:
            audio_to_process = client_audio_buffers[client_id].copy()
            client_audio_buffers[client_id] = np.array([], dtype=np.float32)
            _process_audio_chunk(client_id, audio_to_process)
        else:
            input_db = 20 * np.log10(np.abs(audio_float).max() + 1e-10)
            emit('processed_audio', {
                'audio': '', 'has_speech': False, 'speech_state': 'buffering',
                'audio_level_db': float(round(input_db, 2)), 'vad_probability': 0.0
            })
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}", exc_info=True)

@socketio.on('synthesize_speech')
def handle_synthesize_speech(data):
    """Handle TTS synthesis request"""
    client_id = request.sid
    try:
        if not _HAS_TTS or synthesize_speech is None:
            emit('tts_error', {
                'message': 'TTS is not available. Please install TTS dependencies:\n\n' +
                          'For English TTS:\n' +
                          '  pip install torch transformers datasets soundfile\n\n' +
                          'For Chinese/Japanese TTS:\n' +
                          '  pip install gtts pydub\n\n' +
                          'For speed adjustment:\n' +
                          '  pip install librosa'
            })
            return
        
        text = data.get('text', '').strip()
        language = data.get('language', 'auto').strip()  # Default to auto-detect
        speed = float(data.get('speed', 1.0))
        
        if not text:
            emit('tts_error', {'message': 'Text is required'})
            return
        
        logger.info(f"[TTS] Synthesizing speech for {client_id[:8]}: text='{text[:50]}...', language={language} (auto-detect), speed={speed}")
        
        # Synthesize speech (language will be auto-detected if 'auto')
        audio_bytes, sample_rate = synthesize_speech(
            text=text,
            language=language,
            speed=speed,
            device_preference="auto"
        )
        
        # Convert to base64 for transmission
        import base64
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Detect the actual language used (in case it was auto-detected)
        from .audio.tts import detect_language
        detected_lang = detect_language(text) if language == 'auto' else language
        
        logger.info(f"[TTS] Synthesized {len(audio_bytes)} bytes of audio (sample_rate={sample_rate}Hz, language={detected_lang})")
        
        # Emit audio data to client
        emit('tts_audio', {
            'audio': audio_b64,
            'sample_rate': sample_rate,
            'text': text,
            'language': detected_lang
        })
        
    except Exception as e:
        logger.error(f"Error in synthesize_speech: {e}", exc_info=True)
        emit('tts_error', {'message': str(e)})

if __name__ == '__main__':
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5421))
    certfile = os.path.join(os.path.dirname(__file__), '../certs/cert.pem')
    keyfile = os.path.join(os.path.dirname(__file__), '../certs/key.pem')
    
    if os.path.exists(certfile) and os.path.exists(keyfile):
        logger.info(f"Starting HTTPS server on {host}:{port}")
        socketio.run(app, host=host, port=port, certfile=certfile, keyfile=keyfile)
    else:
        logger.warning("SSL certificates not found, running without HTTPS")
        socketio.run(app, host=host, port=port)
