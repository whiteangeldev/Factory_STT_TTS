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
    def __init__(self, api_key=None, model="gpt-4o-realtime-preview-2024-10-01", sample_rate=24000, on_transcript: Optional[Callable] = None):
        self.model = model
        self.sample_rate = sample_rate
        self.on_transcript = on_transcript
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not found")
            self.api_key = None
            self.ws = None
            self.is_connected = False
            return
        
        self.api_key = api_key
        self.ws = None
        self.is_connected = False
        self.is_streaming = False
        self.audio_queue = queue.Queue(maxsize=100)  # Limit queue size to prevent memory issues
        self.stream_thread = None
        self.ws_thread = None
        self.connection_lock = threading.Lock()
        self.session_id = None
        self.audio_chunks_sent = 0
        self.transcriptions_received = 0
        
        logger.info(f"OpenAI Realtime STT initialized (model={model}, sample_rate={sample_rate})")
    
    def start_stream(self):
        if not self.api_key:
            return False
        
        with self.connection_lock:
            if self.is_connected:
                return True
            
            if self.ws_thread and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=0.5)
            
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass
                self.ws = None
            
            try:
                ws_url = f"wss://api.openai.com/v1/realtime?model={self.model}"
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "OpenAI-Beta": "realtime=v1"
                }
                
                logger.info("Connecting to OpenAI Realtime API...")
                self.ws = websocket.WebSocketApp(
                    ws_url, header=headers,
                    on_open=self._on_open, on_message=self._on_message,
                    on_error=self._on_error, on_close=self._on_close
                )
                
                self.ws_thread = threading.Thread(target=self._ws_run, daemon=True)
                self.ws_thread.start()
                
                timeout = 10.0
                elapsed = 0.0
                while not self.is_connected and elapsed < timeout:
                    time.sleep(0.1)
                    elapsed += 0.1
                
                if not self.is_connected:
                    logger.error(f"Connection timeout after {timeout}s")
                    if self.ws:
                        try:
                            self.ws.close()
                        except:
                            pass
                        self.ws = None
                    return False
                
                self.is_streaming = True
                self.stream_thread = threading.Thread(target=self._stream_audio_worker, daemon=True)
                self.stream_thread.start()
                
                logger.info("Stream started successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start stream: {e}")
                self.is_connected = False
                self.is_streaming = False
                return False
    
    def stop_stream(self):
        with self.connection_lock:
            if not self.is_connected and not self.is_streaming:
                return
            
            try:
                self.is_streaming = False
                
                if self.stream_thread and self.stream_thread.is_alive():
                    self.stream_thread.join(timeout=0.5)
                
                if self.ws and self.is_connected:
                    try:
                        end_event = {
                            "type": "session.update",
                            "session": {
                                "instructions": "", "modalities": [], "voice": "alloy",
                                "input_audio_format": "pcm16", "output_audio_format": "pcm16",
                                "input_audio_transcription": {"model": "whisper-1"},
                                "turn_detection": {
                                    "type": "server_vad", "threshold": 0.5,
                                    "prefix_padding_ms": 300, "silence_duration_ms": 500
                                }
                            }
                        }
                        self.ws.send(json.dumps(end_event))
                    except:
                        pass
                
                while not self.audio_queue.empty():
                    try:
                        self.audio_queue.get_nowait()
                    except queue.Empty:
                        break
                
                if self.ws:
                    try:
                        self.is_connected = False
                        self.ws.close()
                    except:
                        pass
                    finally:
                        self.ws = None
                
                if self.ws_thread and self.ws_thread.is_alive():
                    self.ws_thread.join(timeout=1.0)
                
                self.is_connected = False
                logger.info("Stream stopped")
                
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
                self.is_connected = False
                self.is_streaming = False
    
    def send_audio(self, audio: np.ndarray):
        if not self.is_connected or not self.is_streaming:
            return
        
        try:
            # Validate and prepare audio
            if len(audio) == 0:
                return
            
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Clip to valid range
            audio = np.clip(audio, -1.0, 1.0)
            
            # Check for silence (skip if all zeros to reduce bandwidth)
            max_level = np.abs(audio).max()
            if max_level < 0.001:
                return  # Skip silence chunks
            
            # Convert to int16 PCM
            audio_int16 = (audio * 32767.0).astype(np.int16)
            
            # Put in queue (non-blocking, drop if queue is full)
            try:
                self.audio_queue.put_nowait(audio_int16.tobytes())
            except queue.Full:
                logger.warning("Audio queue full, dropping chunk")
        
        except Exception as e:
            logger.error(f"Error sending audio to STT: {e}", exc_info=True)
    
    def _ws_run(self):
        try:
            sslopt = {"cert_reqs": ssl.CERT_REQUIRED, "check_hostname": True}
            self.ws.run_forever(ping_interval=20, ping_timeout=10, ping_payload="ping", sslopt=sslopt)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.is_connected = False
    
    def _stream_audio_worker(self):
        logger.info("Audio streaming worker started")
        while self.is_streaming:
            try:
                try:
                    audio_bytes = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                if self.ws and self.is_connected:
                    try:
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        event = {"type": "input_audio_buffer.append", "audio": audio_base64}
                        self.ws.send(json.dumps(event))
                        self.audio_chunks_sent += 1
                        
                        # Log first few chunks and periodically
                        if self.audio_chunks_sent <= 3:
                            logger.debug(f"[STT Audio] Sent chunk {self.audio_chunks_sent} ({len(audio_bytes)} bytes)")
                        elif self.audio_chunks_sent % 100 == 0:
                            logger.debug(f"[STT Audio] Sent {self.audio_chunks_sent} chunks, received {self.transcriptions_received} transcriptions")
                    
                    except Exception as e:
                        logger.error(f"Error streaming audio to STT: {e}")
                        if "Connection" in str(e) or "closed" in str(e).lower():
                            break
            except Exception as e:
                logger.error(f"Error in audio worker: {e}")
                break
        logger.info(f"Audio streaming worker stopped (sent {self.audio_chunks_sent} chunks)")
    
    def _on_open(self, ws):
        logger.info("WebSocket connection opened")
        try:
            self.is_connected = True
            
            # Configure session for transcription
            # Important: modalities should include "audio" to enable transcription
            session_config = {
                "type": "session.update",
                "session": {
                    "instructions": "You are a transcription assistant. Transcribe the audio accurately.",
                    "modalities": ["audio", "text"],  # Include "audio" to enable transcription
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
            
            logger.debug("Sending session configuration...")
            ws.send(json.dumps(session_config))
            logger.debug("Session configuration sent")
            
        except Exception as e:
            logger.error(f"Error configuring STT session: {e}", exc_info=True)
            self.is_connected = False
    
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            event_type = data.get("type", "")
            
            # Log all received events (except ping/pong) to debug transcription issues
            if event_type not in ["ping", "pong"]:
                logger.debug(f"[STT Event] {event_type}: {json.dumps(data)[:200]}")
            
            if event_type == "session.created":
                self.session_id = data.get("session", {}).get("id")
                logger.info(f"Session created: {self.session_id}")
            
            # Handle transcription events - multiple possible formats
            elif event_type == "conversation.item.input_audio_transcription.completed":
                item = data.get("item", {})
                transcription = item.get("input_audio_transcription", {})
                if isinstance(transcription, dict):
                    text = transcription.get("transcript", "").strip()
                else:
                    text = str(transcription).strip()
                
                if text and self.on_transcript:
                    self.transcriptions_received += 1
                    logger.info(f"[STT FINAL #{self.transcriptions_received}] '{text}'")
                    self.on_transcript(text, True, "en", 1.0)
            
            elif event_type == "conversation.item.input_audio_transcription.delta":
                # Interim transcriptions (partial results)
                delta = data.get("delta", "")
                if delta and isinstance(delta, str) and delta.strip():
                    logger.debug(f"[STT INTERIM] '{delta[:60]}...'")
                    if self.on_transcript:
                        self.on_transcript(delta.strip(), False, "en", 0.8)
            
            elif event_type == "conversation.item.created":
                # Check if this is a transcription item
                item = data.get("item", {})
                item_type = item.get("type", "")
                
                # Handle different item types that might contain transcriptions
                text = None
                if item_type == "input_audio_transcription":
                    text = item.get("transcript", "").strip()
                elif item_type == "message":
                    transcription = item.get("input_audio_transcription", {})
                    if isinstance(transcription, dict):
                        text = transcription.get("transcript", "").strip()
                    elif isinstance(transcription, str):
                        text = transcription.strip()
                
                if text and self.on_transcript:
                    self.transcriptions_received += 1
                    logger.info(f"[STT CREATED #{self.transcriptions_received}] '{text}'")
                    self.on_transcript(text, True, "en", 1.0)
            
            elif event_type == "conversation.item.update":
                # Check for transcription updates
                item = data.get("item", {})
                if "input_audio_transcription" in item:
                    transcription = item.get("input_audio_transcription", {})
                    if isinstance(transcription, dict):
                        text = transcription.get("transcript", "").strip()
                    elif isinstance(transcription, str):
                        text = transcription.strip()
                    else:
                        text = str(transcription).strip()
                    
                    if text and self.on_transcript:
                        self.transcriptions_received += 1
                        logger.info(f"[STT UPDATE #{self.transcriptions_received}] '{text}'")
                        self.on_transcript(text, True, "en", 1.0)
            
            elif event_type == "error":
                error = data.get("error", {})
                error_msg = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                logger.error(f"STT API error: {error_msg}")
                # Don't disconnect on error - let it recover
            
            elif event_type == "ping":
                try:
                    ws.send(json.dumps({"type": "pong"}))
                except:
                    pass
            
            elif event_type == "input_audio_buffer.speech_started":
                logger.debug("Speech started in STT buffer")
            
            elif event_type == "input_audio_buffer.speech_stopped":
                logger.debug("Speech stopped in STT buffer")
            
            else:
                # Log unknown events for debugging
                if event_type not in ["session.updated", "response.audio_transcription.delta", "response.audio_transcription.done"]:
                    logger.debug(f"[STT Unknown Event] {event_type}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse STT message as JSON: {e}")
            logger.debug(f"Message was: {message[:500]}")
        except Exception as e:
            logger.error(f"Error processing STT message: {e}", exc_info=True)
            logger.debug(f"Message was: {message[:500]}")
    def _on_error(self, ws, error):
        error_str = str(error)
        if "timed out" in error_str.lower():
            logger.error(f"Network timeout: {error}")
        elif "Connection refused" in error_str:
            logger.error(f"Connection refused: {error}")
        else:
            logger.error(f"WebSocket error: {error}")
        self.is_connected = False
    
    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"Connection closed (code: {close_status_code})")
        self.is_connected = False
