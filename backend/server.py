"""Flask-SocketIO server for real-time STT/TTS"""
import eventlet
eventlet.monkey_patch()
try:
    from eventlet import tpool as eventlet_tpool
except Exception:
    eventlet_tpool = None

import base64
import logging
import numpy as np
import re
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
from .audio.stt_whisper_offline import WhisperOfflineSTT

# Try to import TTS (optional - server can run without it)
try:
    from .audio.tts import synthesize_speech
    _HAS_TTS = True
except (ImportError, RuntimeError, PermissionError) as e:
    # Logger not yet defined, use print for early import errors
    print(f"Warning: TTS not available: {e}. TTS feature will be disabled.")
    _HAS_TTS = False
    synthesize_speech = None

_LOG_LEVEL = os.getenv("FACTORY_LOG_LEVEL", "INFO").upper()
_LOG_LEVEL_VALUE = getattr(logging, _LOG_LEVEL, logging.INFO)
logging.basicConfig(level=_LOG_LEVEL_VALUE, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        
        # Suppress tqdm/progress-bar noise from model internals.
        if re.match(r'^\s*\d+%\|', text_str) or "frames/s]" in text_str:
            return
        
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


class FilteredStdout:
    def __init__(self, original):
        self.original = original

    def write(self, text):
        text_str = str(text)
        # Suppress tqdm/progress-bar noise from model internals.
        if re.match(r'^\s*\d+%\|', text_str) or "frames/s]" in text_str:
            return
        self.original.write(text)

    def flush(self):
        self.original.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


sys.stdout = FilteredStdout(sys.stdout)

app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend')
app.config['SECRET_KEY'] = 'factory-stt-tts-secret-key'
CORS(app)

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",
    logger=False,
    engineio_logger=False,
    ping_interval=25,
    ping_timeout=120,
)

config = AudioConfig()
connected_clients = set()
client_pipelines = {}
client_audio_buffers = {}
client_recording_state = {}
client_system_audio = {}  # Server-side system audio capture per client
client_audio_queues = {}  # Queues for passing audio from background threads to eventlet
MIN_CHUNK_SIZE = 480
MIN_AUDIO_LEVEL = 0.0005
MAX_TTS_SEGMENT_CHARS = 220
_stt_preloader = None
_STT_EMIT_TRACE = os.getenv("FACTORY_STT_EMIT_TRACE", "1").lower() not in {"0", "false", "off", "no"}


def _log_stt_emit(client_id: str, event_type: str, payload: dict):
    """Compact STT emit logging: interim text only."""
    if not _STT_EMIT_TRACE:
        return
    if event_type != "transcription_interim":
        return
    text = ((payload or {}).get("text", "") or "").strip()
    if text:
        logger.info(f"[STT INTERIM] {text}")


def _ensure_stt_preloaded():
    """Warm-load Whisper model at page connect/server startup."""
    global _stt_preloader
    if _stt_preloader is not None:
        return
    try:
        _stt_preloader = WhisperOfflineSTT(
            model=config.WHISPER_MODEL,
            sample_rate=16000,
            on_transcript=None,
        )
    except Exception as e:
        logger.warning(f"STT preload skipped: {e}")


def _sanitize_tts_text_for_segmentation(text: str) -> str:
    """Remove reference markers and normalize spacing before sentence splitting."""
    if not text:
        return ""
    cleaned = str(text)
    # Remove citation markers that should not be spoken.
    cleaned = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', cleaned)
    cleaned = re.sub(r'【\d+(?:\s*,\s*\d+)*】', '', cleaned)
    # Remove empty bracket groups such as [] that may remain after cleanup.
    cleaned = re.sub(r'\[\s*\]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _has_speakable_content(text: str) -> bool:
    """Return True when the segment contains letters/numbers/CJK content."""
    if not text:
        return False
    return re.search(r'[A-Za-z0-9\u4e00-\u9fff\u3040-\u30ff]', text) is not None


def _split_long_sentence(sentence: str, max_chars: int = MAX_TTS_SEGMENT_CHARS) -> list[str]:
    """Split a long sentence into smaller chunks while preserving punctuation where possible."""
    text = sentence.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Prefer splitting on comma-like separators first.
    parts = re.split(r'([,，、:：]+)', text)
    chunks = []
    current = ""
    for i in range(0, len(parts), 2):
        content = parts[i].strip()
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        piece = f"{content}{sep}".strip()
        if not piece:
            continue
        if not current:
            current = piece
        elif len(current) + 1 + len(piece) <= max_chars:
            current = f"{current} {piece}".strip()
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)

    # If still too long, split by words (for spaced languages) or hard-cut as final fallback.
    final_chunks = []
    for chunk in chunks if chunks else [text]:
        if len(chunk) <= max_chars:
            final_chunks.append(chunk)
            continue

        words = chunk.split()
        if len(words) > 1:
            current_words = []
            current_len = 0
            for word in words:
                next_len = len(word) if current_len == 0 else current_len + 1 + len(word)
                if next_len <= max_chars:
                    current_words.append(word)
                    current_len = next_len
                else:
                    final_chunks.append(" ".join(current_words))
                    current_words = [word]
                    current_len = len(word)
            if current_words:
                final_chunks.append(" ".join(current_words))
        else:
            for idx in range(0, len(chunk), max_chars):
                final_chunks.append(chunk[idx:idx + max_chars].strip())

    return [c for c in final_chunks if c]


def split_text_for_tts(text: str, max_chars: int = MAX_TTS_SEGMENT_CHARS) -> list[str]:
    """
    Split text into sentence-sized chunks for safer TTS processing.
    Supports English/Chinese/Japanese punctuation.
    """
    normalized = _sanitize_tts_text_for_segmentation(text)
    if not normalized:
        return []

    raw_segments = []
    start = 0
    # Sentence endings for mixed-language text.
    for match in re.finditer(r'[.!?。！？]+(?:["\'”’)\]]+)?', normalized):
        end = match.end()
        segment = normalized[start:end].strip()
        if segment:
            raw_segments.append(segment)
        start = end

    tail = normalized[start:].strip()
    if tail:
        raw_segments.append(tail)

    if not raw_segments:
        raw_segments = [normalized]

    final_segments = []
    for segment in raw_segments:
        final_segments.extend(_split_long_sentence(segment, max_chars=max_chars))
    return [s for s in final_segments if _has_speakable_content(s)]


def _stream_tts_segments(
    client_id: str,
    request_id,
    detected_lang: str,
    speed: float,
    text_segments: list[str]
):
    """Synthesize and emit TTS audio one segment at a time for low-latency playback."""
    safe_segments = [seg for seg in text_segments if _has_speakable_content(seg)]
    total_segments = len(safe_segments)
    total_bytes = 0

    if total_segments == 0:
        error_payload = {'message': 'No valid text segments to synthesize'}
        if request_id:
            error_payload['request_id'] = request_id
        socketio.emit('tts_error', error_payload, room=client_id)
        return

    try:
        for idx, segment_text in enumerate(safe_segments):
            # Offload heavy synthesis/model warmup to a native thread so
            # eventlet heartbeat traffic can continue while TTS is running.
            if eventlet_tpool is not None:
                audio_bytes, sample_rate = eventlet_tpool.execute(
                    synthesize_speech,
                    segment_text,
                    detected_lang,
                    speed,
                    "auto",
                )
            else:
                # Fallback for environments where eventlet.tpool is unavailable.
                audio_bytes, sample_rate = synthesize_speech(
                    text=segment_text,
                    language=detected_lang,
                    speed=speed,
                    device_preference="auto",
                )
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            total_bytes += len(audio_bytes)

            socketio.emit('tts_audio', {
                'request_id': request_id,
                'audio': audio_b64,
                'sample_rate': sample_rate,
                'text': segment_text,
                'language': detected_lang,
                'segment_index': idx,
                'segment_count': total_segments,
                'is_last': idx == total_segments - 1
            }, room=client_id)

            logger.info(f"[TTS] Emitted segment {idx + 1}/{total_segments} to {client_id[:8]}")
            # Yield to eventlet so packets are flushed immediately between segments.
            socketio.sleep(0)

        logger.info(
            f"[TTS] Synthesized {total_segments} segment(s), total_audio_bytes={total_bytes}, language={detected_lang}"
        )
    except Exception as e:
        logger.error(f"Error streaming tts segments: {e}", exc_info=True)
        error_payload = {'message': str(e)}
        if request_id:
            error_payload['request_id'] = request_id
        socketio.emit('tts_error', error_payload, room=client_id)

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
    _ensure_stt_preloaded()
    connected_clients.add(client_id)
    client_audio_buffers[client_id] = np.array([], dtype=np.float32)
    client_recording_state[client_id] = False
    logger.info(f"✅ Client connected: {client_id} (total: {len(connected_clients)})")
    emit('connected', {'status': 'ready', 'message': 'WebSocket ready'})

@socketio.on('disconnect')
def handle_disconnect(reason=None):
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


# Warm up model as soon as server module initializes.
_ensure_stt_preloaded()

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
                payload = {
                    "text": data.get("text", ""),
                    "language": data.get("language"),
                    "confidence": float(data.get("confidence", 0.0)),
                    "timestamp": data.get("timestamp"),
                    "is_final": True
                }
                _log_stt_emit(client_id, "transcription", payload)
                socketio.emit("transcription", payload, room=client_id)
            elif event_type == "transcription_interim":
                payload = {
                    "text": data.get("text", ""),
                    "language": data.get("language"),
                    "confidence": float(data.get("confidence", 0.0)),
                    "timestamp": data.get("timestamp"),
                    "incremental_text": data.get("incremental_text"),
                    "is_final": False
                }
                _log_stt_emit(client_id, "transcription_interim", payload)
                socketio.emit("transcription_interim", payload, room=client_id)
            else:
                socketio.emit("speech_event", {
                    "type": "speech_event",
                    "event": event_type,
                    "data": {k: v for k, v in data.items() if k != "audio_segment"}
                }, room=client_id)
        except Exception as e:
            logger.error(f"Error emitting event to {client_id}: {e}")
    
    client_pipelines[client_id] = AudioPipeline(
        config,
        event_callback=client_speech_callback,
        input_mode=input_mode,
    )
    client_pipelines[client_id].reset()
    
    # Both system audio and microphone: use server-side capture
    try:
        # Use a queue to pass audio from background thread to eventlet context.
        # Queue must be drained fast: capture produces ~33 chunks/sec (30ms each); processing must keep up.
        import queue
        audio_event_queue = queue.Queue(maxsize=500)  # Absorb bursts; drain loop below keeps up with capture
        client_audio_queues[client_id] = audio_event_queue
        
        def on_audio_chunk(audio_data):
            """Callback for server-side audio - queues for eventlet processing"""
            if not client_recording_state.get(client_id, False):
                return
            try:
                audio_event_queue.put_nowait((client_id, audio_data))
            except queue.Full:
                pass  # Drop only if queue full (should be rare with drain loop)
        
        # Process queued audio: drain queue when data available so we keep up with capture (~33 chunks/sec).
        # Single get(timeout=0.1) would process at most 10/sec and cause backlog + dropped chunks.
        def process_audio_queue():
            while client_recording_state.get(client_id, False) and client_id in client_audio_queues:
                try:
                    queued_client_id, audio_data = audio_event_queue.get(timeout=0.02)  # Short wait when empty
                    if queued_client_id in client_pipelines and client_pipelines[queued_client_id]:
                        _process_audio_chunk(queued_client_id, audio_data)
                    # Drain available chunks (cap per iteration so we yield to event loop)
                    drain_count = 0
                    max_drain = 50
                    while drain_count < max_drain:
                        try:
                            queued_client_id, audio_data = audio_event_queue.get_nowait()
                            if queued_client_id in client_pipelines and client_pipelines[queued_client_id]:
                                _process_audio_chunk(queued_client_id, audio_data)
                            drain_count += 1
                        except queue.Empty:
                            break
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error processing queued audio: {e}")
        
        socketio.start_background_task(process_audio_queue)
        
        # Create audio capture with appropriate input type
        audio_capture = SystemAudioCapture(
            sample_rate=config.SAMPLE_RATE,
            chunk_size=MIN_CHUNK_SIZE,
            on_audio=on_audio_chunk,
            input_type=input_mode  # 'system' or 'microphone'
        )
        
        if audio_capture.start():
            client_system_audio[client_id] = audio_capture
            input_name = 'system audio' if input_mode == 'system' else 'microphone'
            logger.info(f"🎙️ {input_name.capitalize()} started: {client_id}")
            emit('recording_status', {'is_recording': True, 'status': f'Recording {input_name}...'})
        else:
            input_name = 'system audio' if input_mode == 'system' else 'microphone'
            logger.error(f"⚠️ Failed to start {input_name} for {client_id}")
            emit('recording_status', {'is_recording': False, 'status': f'Failed to start {input_name}. Check server logs.'})
    except Exception as e:
        input_name = 'system audio' if input_mode == 'system' else 'microphone'
        logger.error(f"Failed to start {input_name}: {e}")
        emit('recording_status', {'is_recording': False, 'status': f'{input_name.capitalize()} error: {str(e)}'})

@socketio.on('stop_recording')
def handle_stop_recording(data=None):
    client_id = request.sid
    client_recording_state[client_id] = False
    client_audio_buffers[client_id] = np.array([], dtype=np.float32)

    # Stop server-side capture first to prevent new chunks during finalization.
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

    # Now stop pipeline STT and emit final transcription.
    pipeline = None
    if client_id in client_pipelines:
        pipeline = client_pipelines[client_id]
        if pipeline:
            try:
                if hasattr(pipeline, 'streaming_stt') and pipeline.streaming_stt:
                    if pipeline.streaming_stt.is_streaming:
                        logger.info(f"🛑 Stopping STT stream for {client_id} to generate final transcription")
                        pipeline.streaming_stt.stop_stream()
            except Exception as e:
                logger.error(f"Error stopping pipeline STT: {e}", exc_info=True)
    
    # Now remove the pipeline after transcription should be complete
    # The pipeline callback will still work until this point, so transcriptions can be emitted
    if client_id in client_pipelines:
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
                        payload = {
                            "text": data.get("text", ""),
                            "language": data.get("language"),
                            "confidence": float(data.get("confidence", 0.0)),
                            "timestamp": data.get("timestamp"),
                            "is_final": True
                        }
                        _log_stt_emit(client_id, "transcription", payload)
                        socketio.emit("transcription", payload, room=client_id)
                    elif event_type == "transcription_interim":
                        payload = {
                            "text": data.get("text", ""),
                            "language": data.get("language"),
                            "confidence": float(data.get("confidence", 0.0)),
                            "timestamp": data.get("timestamp"),
                            "incremental_text": data.get("incremental_text"),
                            "is_final": False
                        }
                        _log_stt_emit(client_id, "transcription_interim", payload)
                        socketio.emit("transcription_interim", payload, room=client_id)
                    else:
                        socketio.emit("speech_event", {
                            "type": "speech_event",
                            "event": event_type,
                            "data": {k: v for k, v in data.items() if k != "audio_segment"}
                        }, room=client_id)
                except Exception as e:
                    logger.error(f"Error emitting event to {client_id}: {e}")
            
            fallback_input_mode = "microphone"
            if client_id in client_system_audio and client_system_audio[client_id] is not None:
                fallback_input_mode = getattr(client_system_audio[client_id], "input_type", "microphone")
            client_pipelines[client_id] = AudioPipeline(
                config,
                event_callback=client_speech_callback,
                input_mode=fallback_input_mode,
            )
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
        logger.debug(f"[Audio Debug {_process_audio_chunk._debug_count[client_id]}] Client {client_id}: level={audio_level:.6f}, dB={input_db:.2f}, samples={len(audio_to_process)}")
    
    # CRITICAL: Process ALL audio through pipeline (VAD, noise reduction, STT)
    # Do NOT skip based on audio level - let VAD determine if it's speech
    # This ensures both microphone and system audio modes work identically
    processed = pipeline.process_chunk(audio_to_process)
    input_db = 20 * np.log10(np.abs(audio_to_process).max() + 1e-10)
    # CRITICAL: Use per-chunk VAD result for has_speech (frontend needs this for history)
    # But use pipeline.is_speaking for speech_state (aggregated state with chunk counting)
    # This allows frontend to build history while respecting backend's chunk counting logic
    is_speech = pipeline.last_chunk_is_speech  # Per-chunk VAD result for frontend history
    speech_state = "speech" if pipeline.is_speaking else "silence"  # Aggregated state
    # Get VAD probability from denoised audio for display purposes
    audio_for_vad = processed if processed is not None else audio_to_process
    vad_prob = pipeline.vad.get_probability(audio_for_vad) if audio_for_vad is not None else 0.0
    
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
        logger.debug(f"[Emit Debug {_process_audio_chunk._emit_count[client_id]}] Emitting processed_audio to {client_id[:8]}... (has_speech={is_speech}, {len(audio_b64)} bytes)")
    
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
    # Skip browser audio - both modes now use server-side capture
    # This handler is kept for backward compatibility but should not be used
    if client_id in client_system_audio:
        return  # Server-side capture is active
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
                'audio': '', 'has_speech': False, 'speech_state': 'silence',
                'audio_level_db': float(round(input_db, 2)), 'vad_probability': 0.0
            })
    except Exception as e:
        logger.error(f"Error processing audio chunk: {e}", exc_info=True)

@socketio.on('synthesize_speech')
def handle_synthesize_speech(data):
    """Handle TTS synthesis request"""
    client_id = request.sid
    request_id = None
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
        request_id = data.get('request_id') if isinstance(data, dict) else None
        
        if not text:
            emit('tts_error', {'message': 'Text is required'})
            return
        
        logger.info(f"[TTS] Synthesizing speech for {client_id[:8]}: text='{text[:50]}...', language={language} (auto-detect), speed={speed}")

        # Detect the actual language used (in case it was auto-detected)
        from .audio.tts import detect_language
        detected_lang = detect_language(text) if language == 'auto' else language

        # Split long text into sentence-level segments and synthesize in order.
        text_segments = split_text_for_tts(text)
        if not text_segments:
            emit('tts_error', {'message': 'No valid text segments to synthesize'})
            return

        logger.info(f"[TTS] Split input into {len(text_segments)} segment(s) for sequential synthesis")

        socketio.start_background_task(
            _stream_tts_segments,
            client_id,
            request_id,
            detected_lang,
            speed,
            text_segments
        )
        
    except Exception as e:
        logger.error(f"Error in synthesize_speech: {e}", exc_info=True)
        error_payload = {'message': str(e)}
        try:
            if request_id:
                error_payload['request_id'] = request_id
        except Exception:
            pass
        emit('tts_error', error_payload)

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
