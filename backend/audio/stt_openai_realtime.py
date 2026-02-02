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
        self.audio_queue = queue.Queue(maxsize=50)  # Smaller queue for lower latency
        self.stream_thread = None
        self.ws_thread = None
        self.connection_lock = threading.Lock()
        self.session_id = None
        self.audio_chunks_sent = 0
        self.transcriptions_received = 0
        self.first_audio_time = None  # Track when first audio is sent
        self.first_transcription_time = None  # Track when first transcription arrives
        self.pending_transcription = False  # Track if we're waiting for transcription
        self.last_item_id = None  # Track the last item ID we're waiting for transcription
        
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
                        # Check if buffer was already committed automatically by API
                        # If not, commit it manually
                        if self.pending_transcription or self.last_item_id:
                            logger.info(f"[STT] Buffer already committed (item_id={self.last_item_id}) - waiting for transcription...")
                        else:
                            # Commit any remaining audio in the buffer
                            commit_event = {"type": "input_audio_buffer.commit"}
                            self.ws.send(json.dumps(commit_event))
                            logger.info("[STT] Committed final audio buffer - waiting for transcription...")
                            self.pending_transcription = True
                        
                        # Wait longer for transcription events to arrive (up to 5 seconds)
                        # Transcription can take time after buffer commit, especially for longer speech
                        max_wait = 5.0
                        wait_interval = 0.2
                        waited = 0.0
                        while waited < max_wait and self.pending_transcription:
                            time.sleep(wait_interval)
                            waited += wait_interval
                            if not self.pending_transcription:
                                logger.info(f"[STT] Transcription received after {waited:.2f}s")
                                break
                        
                        if self.pending_transcription:
                            logger.warning(f"[STT] Still waiting for transcription after {waited:.2f}s - proceeding anyway")
                        else:
                            logger.info(f"[STT] Transcription completed, closing stream")
                        
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
                    except Exception as e:
                        logger.error(f"Error in stop_stream cleanup: {e}")
                
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
            
            # Track first audio timestamp for latency measurement
            if self.first_audio_time is None:
                self.first_audio_time = time.time()
            
            # Check for silence (skip if all zeros to reduce bandwidth)
            # Use a lower threshold to avoid filtering out quiet speech
            max_level = np.abs(audio).max()
            if max_level < 0.0001:  # Lowered from 0.001 to avoid filtering quiet speech
                return  # Skip only truly silent chunks
            
            # Convert to int16 PCM
            audio_int16 = (audio * 32767.0).astype(np.int16)
            
            # Put in queue (non-blocking, drop if queue is full)
            try:
                self.audio_queue.put_nowait(audio_int16.tobytes())
                # Log first few chunks to verify they're being queued
                if self.audio_chunks_sent == 0:
                    logger.debug(f"[STT] First audio chunk queued (len={len(audio_int16)}, max={max_level:.6f})")
            except queue.Full:
                logger.warning(f"Audio queue full, dropping chunk (queue size: {self.audio_queue.qsize()})")
        
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
                    audio_bytes = self.audio_queue.get(timeout=0.05)  # Reduced timeout for lower latency
                except queue.Empty:
                    continue
                
                if self.ws and self.is_connected:
                    try:
                        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                        event = {"type": "input_audio_buffer.append", "audio": audio_base64}
                        self.ws.send(json.dumps(event))
                        self.audio_chunks_sent += 1
                        
                        # Log first few chunks to verify they're being sent
                        if self.audio_chunks_sent <= 5:
                            logger.info(f"[STT Audio] Sent chunk {self.audio_chunks_sent} ({len(audio_bytes)} bytes, {len(audio_base64)} base64 chars)")
                        
                        # Don't commit buffer automatically - let speech end trigger final commit
                        # This prevents early finalization of transcription
                        # Buffer commits will happen when speech ends (in stop_stream)
                        
                        # Log periodically
                        if self.audio_chunks_sent % 100 == 0:
                            logger.info(f"[STT Audio] Sent {self.audio_chunks_sent} chunks, received {self.transcriptions_received} transcriptions")
                    
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
            
            # Configure session for low-latency, high-accuracy transcription
            # Optimized settings for real-time transcription with minimal delay
            session_config = {
                "type": "session.update",
                "session": {
                    "instructions": "You are a transcription assistant. Transcribe the audio accurately and in real-time.",
                    "modalities": ["audio", "text"],  # Required: ["audio", "text"] for transcription
                    "voice": "alloy",
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": "whisper-1"
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 200,  # Reduced from 300ms for faster response
                        "silence_duration_ms": 300  # Reduced from 500ms for lower latency
                    }
                }
            }
            
            logger.debug("Sending session configuration...")
            ws.send(json.dumps(session_config))
            logger.debug("Session configuration sent")
            
            # Request transcription explicitly after session is configured
            # This ensures transcription is enabled
            transcription_request = {
                "type": "input_audio_buffer.commit"
            }
            logger.debug("Requesting transcription...")
            # Note: We'll send this after we start receiving audio, not here
            
        except Exception as e:
            logger.error(f"Error configuring STT session: {e}", exc_info=True)
            self.is_connected = False
    
    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            event_type = data.get("type", "")
            
            # Log all received events (except ping/pong) to debug transcription issues
            if event_type not in ["ping", "pong", "session.updated"]:
                logger.info(f"[STT Event] {event_type}: {json.dumps(data)[:300]}")
            
            if event_type == "session.created":
                self.session_id = data.get("session", {}).get("id")
                logger.info(f"Session created: {self.session_id}")
            
            # Handle transcription events - multiple possible formats
            elif event_type == "conversation.item.input_audio_transcription.completed":
                # Get transcript from the event data directly
                transcript = data.get("transcript", "")
                if not transcript:
                    # Fallback: try to get from item structure
                    item = data.get("item", {})
                    if item:
                        transcription = item.get("input_audio_transcription", {})
                        if isinstance(transcription, dict):
                            transcript = transcription.get("transcript", "").strip()
                        else:
                            transcript = str(transcription).strip()
                else:
                    transcript = transcript.strip()
                
                if transcript and self.on_transcript:
                    self.transcriptions_received += 1
                    self.pending_transcription = False  # Mark transcription as received
                    # Track latency
                    if self.first_transcription_time is None and self.first_audio_time:
                        latency = time.time() - self.first_audio_time
                        logger.info(f"[STT Latency] First transcription: {latency:.3f}s")
                        self.first_transcription_time = time.time()
                    logger.info(f"[STT FINAL #{self.transcriptions_received}] '{transcript}'")
                    self.on_transcript(transcript, True, "en", 1.0)
                elif not transcript:
                    logger.warning(f"[STT] Received completed event but transcript is empty: {json.dumps(data)[:200]}")
            
            elif event_type == "conversation.item.input_audio_transcription.delta":
                # Interim transcriptions (partial results) - send immediately for low latency
                delta = data.get("delta", "")
                if delta and isinstance(delta, str) and delta.strip():
                    # Track first interim result latency
                    if self.first_transcription_time is None and self.first_audio_time:
                        latency = time.time() - self.first_audio_time
                        logger.info(f"[STT Latency] First interim: {latency:.3f}s")
                        self.first_transcription_time = time.time()
                    logger.info(f"[STT INTERIM] '{delta}'")
                    if self.on_transcript:
                        self.on_transcript(delta.strip(), False, "en", 0.8)
                    # Don't mark pending as False for interim - wait for completed event
            
            elif event_type == "conversation.item.created":
                # Check if this is a transcription item
                item = data.get("item", {})
                item_type = item.get("type", "")
                item_id = item.get("id", "")
                
                # If this is the item we're waiting for, mark that we're waiting
                if item_id == self.last_item_id:
                    self.pending_transcription = True
                    logger.info(f"[STT] Item created for committed buffer (id={item_id}) - waiting for transcription")
                
                # Handle different item types that might contain transcriptions
                text = None
                if item_type == "input_audio_transcription":
                    text = item.get("transcript", "").strip()
                elif item_type == "message":
                    # Check content array for transcription (transcript might be in content[0].transcript)
                    content = item.get("content", [])
                    for content_item in content:
                        if content_item.get("type") == "input_audio":
                            # Transcript might be in the content item
                            transcript = content_item.get("transcript")
                            if transcript:
                                text = transcript.strip()
                                break
                    
                    # Also check for input_audio_transcription field
                    if not text:
                        transcription = item.get("input_audio_transcription", {})
                        if isinstance(transcription, dict):
                            text = transcription.get("transcript", "").strip()
                        elif isinstance(transcription, str):
                            text = transcription.strip()
                
                if text and self.on_transcript:
                    self.transcriptions_received += 1
                    self.pending_transcription = False
                    logger.info(f"[STT CREATED #{self.transcriptions_received}] '{text}'")
                    self.on_transcript(text, True, "en", 1.0)
                elif item_type == "message":
                    # Log when we get a message item but no transcript yet (transcription might come later)
                    logger.info(f"[STT] Message item created (id={item_id}) but transcript is null - waiting for transcription events")
            
            elif event_type == "conversation.item.update":
                # Check for transcription updates
                item = data.get("item", {})
                
                # Check content array for transcription updates
                content = item.get("content", [])
                text = None
                for content_item in content:
                    if content_item.get("type") == "input_audio":
                        transcript = content_item.get("transcript")
                        if transcript:
                            text = transcript.strip()
                            break
                
                # Also check for input_audio_transcription field
                if not text and "input_audio_transcription" in item:
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
            
            elif event_type == "input_audio_buffer.committed":
                # Buffer committed - transcription should be coming soon
                item_id = data.get("item_id")
                self.last_item_id = item_id
                self.pending_transcription = True
                logger.info(f"[STT] Audio buffer committed for item: {item_id} - waiting for transcription")
                # Don't extract transcription here - wait for conversation.item events
            
            else:
                # Log unknown events for debugging - might contain transcription data
                if event_type not in ["session.updated", "response.audio_transcription.delta", "response.audio_transcription.done"]:
                    logger.info(f"[STT Unknown Event] {event_type}: {json.dumps(data)[:300]}")
                    # Check if this unknown event contains transcription data
                    event_str = json.dumps(data).lower()
                    if "transcription" in event_str or "transcript" in event_str:
                        logger.warning(f"[STT] Unknown event with transcription data: {event_type}")
                        # Try to extract transcription from unknown event
                        text = None
                        if "transcript" in data:
                            text = data.get("transcript", "").strip()
                        elif "transcription" in data:
                            trans = data.get("transcription", {})
                            if isinstance(trans, dict):
                                text = trans.get("transcript", "").strip()
                            elif isinstance(trans, str):
                                text = trans.strip()
                        
                        if text and self.on_transcript:
                            logger.info(f"[STT EXTRACTED] '{text}'")
                            self.on_transcript(text, True, "en", 0.9)
        
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
