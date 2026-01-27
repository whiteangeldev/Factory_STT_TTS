"""Streaming Speech-to-Text using OpenAI Realtime API"""
import json
import base64
import numpy as np
import logging
import threading
import queue
import time
import os
import ssl
import websocket
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class OpenAIRealtimeSTT:
    """OpenAI Realtime API Streaming STT"""
    
    def __init__(self, api_key=None, model="gpt-4o-realtime-preview-2024-10-01", sample_rate=24000, on_transcript: Optional[Callable] = None):
        """
        Initialize OpenAI Realtime Streaming STT
        
        Args:
            api_key: OpenAI API key (if None, will try to get from OPENAI_API_KEY env var)
            model: Model to use (default: gpt-4o-realtime-preview-2024-10-01)
            sample_rate: Audio sample rate (default: 24000 for OpenAI Realtime API)
            on_transcript: Callback function called when transcription is received
                           Signature: on_transcript(text: str, is_final: bool, language: str, confidence: float)
        """
        self.model = model
        self.sample_rate = sample_rate
        self.on_transcript = on_transcript
        
        # Get API key from environment or parameter
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("=" * 60)
            logger.error("❌ OPENAI_API_KEY NOT FOUND!")
            logger.error("   OpenAI requires an API key to work.")
            logger.error("   Get your API key at: https://platform.openai.com/api-keys")
            logger.error("   Then set it: export OPENAI_API_KEY='your-api-key-here'")
            logger.error("   Or add it to your .env file: OPENAI_API_KEY=your-api-key-here")
            logger.error("=" * 60)
            self.api_key = None
            self.ws = None
            self.is_connected = False
            return
        
        self.api_key = api_key
        
        # WebSocket connection
        self.ws = None
        self.is_connected = False
        self.is_streaming = False
        self.audio_queue = queue.Queue()
        self.stream_thread = None
        self.ws_thread = None
        self.connection_lock = threading.Lock()
        self.session_id = None
        
        logger.info(f"OpenAI Realtime STT initialized (model={model}, sample_rate={sample_rate})")
        
        # Note: Network connectivity will be tested when connection is attempted
        # We don't test here to avoid blocking initialization and spam
    
    def start_stream(self):
        """Start the streaming transcription session"""
        if not self.api_key:
            logger.error("[OpenAIRealtime] Cannot start stream: API key not configured")
            return False
        
        with self.connection_lock:
            if self.is_connected:
                logger.warning("[OpenAIRealtime] Stream already started")
                return True
            
            try:
                # OpenAI Realtime API WebSocket URL
                ws_url = f"wss://api.openai.com/v1/realtime?model={self.model}"
                
                # Create WebSocket connection with Authorization header
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                }
                
                logger.info("[OpenAIRealtime] Connecting to OpenAI Realtime API...")
                
                # Create WebSocket connection
                self.ws = websocket.WebSocketApp(
                    ws_url,
                    header=headers,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )
                
                # Start WebSocket in a separate thread
                self.ws_thread = threading.Thread(target=self._ws_run, daemon=True)
                self.ws_thread.start()
                
                # Wait for connection to be established (with timeout)
                connection_timeout = 10.0  # 10 seconds timeout (increased for SSL handshake)
                wait_interval = 0.1  # Check every 100ms
                elapsed = 0.0
                
                logger.debug("[OpenAIRealtime] Waiting for WebSocket connection...")
                while not self.is_connected and elapsed < connection_timeout:
                    time.sleep(wait_interval)
                    elapsed += wait_interval
                
                if not self.is_connected:
                    logger.error(f"[OpenAIRealtime] ❌ Connection timeout - WebSocket did not open within {connection_timeout}s")
                    logger.error("   Possible causes:")
                    logger.error("   1. SSL/TLS handshake taking too long")
                    logger.error("   2. Network latency or firewall blocking WebSocket upgrade")
                    logger.error("   3. OpenAI API temporarily unavailable")
                    self.is_streaming = False
                    if self.ws:
                        try:
                            self.ws.close()
                        except:
                            pass
                        self.ws = None
                    return False
                
                logger.debug(f"[OpenAIRealtime] Connection established in {elapsed:.2f}s")
                
                # Connection established, start audio streaming thread
                self.is_streaming = True
                self.stream_thread = threading.Thread(target=self._stream_audio_worker, daemon=True)
                self.stream_thread.start()
                
                logger.info("[OpenAIRealtime] ✅ Stream started successfully (connection established)")
                return True
                
            except Exception as e:
                error_str = str(e)
                if "Lookup timed out" in error_str or "timed out" in error_str.lower():
                    logger.error(f"[OpenAIRealtime] ❌ Network timeout when starting stream: {e}")
                    logger.error("   Cannot connect to OpenAI API - check network connectivity")
                else:
                    logger.error(f"[OpenAIRealtime] ❌ Failed to start stream: {e}", exc_info=True)
                self.is_connected = False
                self.is_streaming = False
                return False
    
    def stop_stream(self):
        """Stop the streaming transcription session"""
        with self.connection_lock:
            if not self.is_connected:
                return
            
            try:
                self.is_streaming = False
                
                # Send session end event
                if self.ws and self.is_connected:
                    try:
                        end_event = {
                            "type": "session.update",
                            "session": {
                                "instructions": "",
                                "modalities": [],
                                "voice": "alloy",
                                "input_audio_format": "pcm16",
                                "output_audio_format": "pcm16",
                                "input_audio_transcription": {
                                    "model": "whisper-1"
                                },
                                "turn_detection": {
                                    "type": "server_vad",
                                    "threshold": 0.5,
                                    "prefix_padding_ms": 300,
                                    "silence_duration_ms": 500
                                }
                            }
                        }
                        self.ws.send(json.dumps(end_event))
                    except Exception as e:
                        logger.warning(f"[OpenAIRealtime] Error sending end event: {e}")
                
                # Clear audio queue
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Close WebSocket connection
                if self.ws:
                    try:
                        self.ws.close()
                    except Exception as e:
                        logger.warning(f"[OpenAIRealtime] Error closing connection: {e}")
                    self.ws = None
                
                self.is_connected = False
                logger.info("[OpenAIRealtime] Stream stopped")
                
            except Exception as e:
                logger.error(f"[OpenAIRealtime] Error stopping stream: {e}", exc_info=True)
    
    def send_audio(self, audio: np.ndarray):
        """
        Send audio chunk to streaming API
        
        Args:
            audio: Audio data as numpy array (float32, range [-1, 1])
        """
        if not self.is_connected or not self.is_streaming:
            return
        
        try:
            # Ensure audio is float32 and in correct range
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Clip to valid range
            audio = np.clip(audio, -1.0, 1.0)
            
            # Convert to int16 PCM (OpenAI expects PCM16)
            audio_int16 = (audio * 32767.0).astype(np.int16)
            
            # Add to queue for streaming
            self.audio_queue.put(audio_int16.tobytes())
            
        except Exception as e:
            logger.error(f"[OpenAIRealtime] Error sending audio: {e}", exc_info=True)
    
    def _ws_run(self):
        """Run WebSocket in a thread"""
        try:
            # Set SSL options to prevent hanging
            # Use default SSL context (verify certificates)
            sslopt = {
                "cert_reqs": ssl.CERT_REQUIRED,
                "check_hostname": True
            }
            
            # Note: websocket-client's run_forever() doesn't support timeout parameter
            # Connection timeout is handled in start_stream() instead
            self.ws.run_forever(
                ping_interval=20,  # Send ping every 20 seconds
                ping_timeout=10,   # Wait 10 seconds for pong
                ping_payload="ping",
                sslopt=sslopt
            )
        except websocket.WebSocketTimeoutException as e:
            logger.error(f"[OpenAIRealtime] WebSocket timeout: {e}")
            self.is_connected = False
        except Exception as e:
            logger.error(f"[OpenAIRealtime] WebSocket run error: {e}", exc_info=True)
            self.is_connected = False
    
    def _stream_audio_worker(self):
        """Worker thread that streams audio from queue to OpenAI"""
        logger.info("[OpenAIRealtime] Audio streaming worker started")
        
        while self.is_streaming:
            try:
                # Get audio chunk from queue (with timeout)
                try:
                    audio_bytes = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Send to OpenAI Realtime API
                if self.ws and self.is_connected:
                    try:
                        # Encode audio as base64
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        
                        # Create input audio event
                        event = {
                            "type": "input_audio_buffer.append",
                            "audio": audio_base64
                        }
                        
                        self.ws.send(json.dumps(event))
                    except Exception as e:
                        logger.error(f"[OpenAIRealtime] Error streaming audio chunk: {e}")
                        break
                
            except Exception as e:
                logger.error(f"[OpenAIRealtime] Error in audio streaming worker: {e}", exc_info=True)
                break
        
        logger.info("[OpenAIRealtime] Audio streaming worker stopped")
    
    def _on_open(self, ws):
        """Callback when WebSocket connection is opened"""
        logger.info("[OpenAIRealtime] ✅ WebSocket connection opened")
        
        try:
            # Mark as connected first (before sending config)
            self.is_connected = True
            
            # Configure session for transcription
            session_config = {
                "type": "session.update",
                "session": {
                    "instructions": "",
                    "modalities": ["text"],
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500
                    }
                }
            }
            
            ws.send(json.dumps(session_config))
            logger.debug("[OpenAIRealtime] Session configuration sent")
            
        except Exception as e:
            logger.error(f"[OpenAIRealtime] Error configuring session: {e}", exc_info=True)
            self.is_connected = False
    
    def _on_message(self, ws, message):
        """Callback when message is received from OpenAI"""
        try:
            data = json.loads(message)
            event_type = data.get("type", "")
            
            # Log all events for debugging (first 20 events)
            if not hasattr(self, '_message_count'):
                self._message_count = 0
            self._message_count += 1
            if self._message_count <= 20:
                logger.debug(f"[OpenAIRealtime] Received event #{self._message_count}: {event_type}")
            
            if event_type == "session.created":
                self.session_id = data.get("session", {}).get("id")
                logger.info(f"[OpenAIRealtime] ✅ Session created: {self.session_id}")
            
            elif event_type == "session.updated":
                logger.debug("[OpenAIRealtime] Session updated")
            
            elif event_type == "input_audio_buffer.speech_started":
                logger.debug("[OpenAIRealtime] 🎤 Speech started (server VAD)")
            
            elif event_type == "input_audio_buffer.speech_stopped":
                logger.debug("[OpenAIRealtime] 🎤 Speech stopped (server VAD)")
            
            elif event_type == "input_audio_buffer.committed":
                logger.debug("[OpenAIRealtime] Audio buffer committed")
            
            elif event_type == "conversation.item.created":
                item = data.get("item", {})
                item_type = item.get("type", "")
                logger.debug(f"[OpenAIRealtime] Conversation item created: {item_type}")
            
            elif event_type == "conversation.item.input_audio_transcription.completed":
                # Final transcription completed
                item = data.get("item", {})
                transcription = item.get("input_audio_transcription", {})
                text = transcription.get("transcript", "").strip()
                
                if text and self.on_transcript:
                    logger.info(f"[OpenAIRealtime] [FINAL] '{text[:60]}...'")
                    self.on_transcript(text, True, "en", 1.0)
                elif not text:
                    logger.warning("[OpenAIRealtime] Transcription completed but text is empty")
            
            elif event_type == "conversation.item.input_audio_transcription.failed":
                error = data.get("error", {})
                logger.error(f"[OpenAIRealtime] ❌ Transcription failed: {error}")
            
            elif event_type == "response.audio_transcription.delta":
                # Interim transcription (if supported)
                delta = data.get("delta", "")
                if delta and self.on_transcript:
                    logger.debug(f"[OpenAIRealtime] [INTERIM] '{delta[:60]}...'")
                    # Note: OpenAI Realtime may not support interim transcriptions
                    # This is here in case they add it in the future
            
            elif event_type == "error":
                error = data.get("error", {})
                error_message = error.get("message", str(error))
                error_code = error.get("code", "unknown")
                logger.error(f"[OpenAIRealtime] ❌ API error [{error_code}]: {error_message}")
            
            elif event_type == "ping":
                # Respond to ping
                try:
                    ws.send(json.dumps({"type": "pong"}))
                except Exception as e:
                    logger.warning(f"[OpenAIRealtime] Error sending pong: {e}")
            
            else:
                # Log unknown event types (first 10)
                if self._message_count <= 10:
                    logger.debug(f"[OpenAIRealtime] Unknown event type: {event_type}")
            
        except json.JSONDecodeError as e:
            logger.error(f"[OpenAIRealtime] ❌ Failed to parse JSON message: {e}")
            logger.debug(f"[OpenAIRealtime] Message: {message[:200]}")
        except Exception as e:
            logger.error(f"[OpenAIRealtime] ❌ Error processing message: {e}", exc_info=True)
    
    def _on_error(self, ws, error):
        """Callback when error occurs"""
        error_str = str(error)
        if "Lookup timed out" in error_str or "timed out" in error_str.lower():
            logger.error(f"[OpenAIRealtime] ❌ Network timeout error: {error}")
            logger.error("   This usually means:")
            logger.error("   1. DNS resolution failed (cannot resolve api.openai.com)")
            logger.error("   2. Network firewall blocking OpenAI API")
            logger.error("   3. Server has no internet connectivity")
            logger.error("   Check network connectivity: ping api.openai.com")
        elif "Connection refused" in error_str:
            logger.error(f"[OpenAIRealtime] ❌ Connection refused: {error}")
            logger.error("   OpenAI API may be down or unreachable")
        else:
            logger.error(f"[OpenAIRealtime] ❌ WebSocket error: {error}")
        self.is_connected = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Callback when connection is closed"""
        logger.info(f"[OpenAIRealtime] Connection closed (code: {close_status_code})")
        self.is_connected = False
