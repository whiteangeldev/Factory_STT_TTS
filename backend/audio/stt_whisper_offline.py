"""Offline Speech-to-Text using OpenAI Whisper"""
import numpy as np
import logging
import threading
import time
from typing import Optional, Callable

# Try to import Whisper
try:
    import whisper
    _HAS_WHISPER = True
except ImportError:
    _HAS_WHISPER = False
    whisper = None

logger = logging.getLogger(__name__)

class WhisperOfflineSTT:
    def __init__(self, model="base", sample_rate=16000, on_transcript: Optional[Callable] = None):
        """
        Initialize offline Whisper STT.
        
        Args:
            model: Whisper model size ("tiny", "base", "small", "medium", "large")
            sample_rate: Audio sample rate (Whisper uses 16kHz)
            on_transcript: Callback function(text, is_final, language, confidence)
        """
        self.model_name = model
        self.sample_rate = sample_rate
        self.on_transcript = on_transcript
        
        if not _HAS_WHISPER:
            logger.error("Whisper not available. Install with: pip install openai-whisper")
            self.model = None
            self.is_available = False
            self.api_key = None  # Compatibility property
            return
        
        self.is_available = True
        self.model = None
        self.is_streaming = False
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        self.processing_thread = None
        self.min_buffer_duration = 2.0  # Minimum 2 seconds of audio before processing (increased for better accuracy)
        self.max_buffer_duration = 6.0  # Maximum 6 seconds before forcing transcription
        self.interim_interval = 2.0  # Process interim results every 2 seconds
        self.last_transcription_time = None
        self.audio_chunks_received = 0
        self.first_audio_time = None
        self.first_transcription_time = None
        self.transcriptions_received = 0
        self.current_interim_text = ""  # Track interim text
        self.last_final_text = ""  # Track last final text
        
        # Compatibility property for pipeline
        self.api_key = "offline"  # Non-None value to indicate availability
        
        # Load model in background thread to avoid blocking
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model (can be slow, so do it in background)"""
        def load():
            try:
                logger.info(f"Loading Whisper model: {self.model_name} (this may take a moment...)")
                self.model = whisper.load_model(self.model_name)
                logger.info(f"Whisper model {self.model_name} loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Whisper model: {e}")
                self.is_available = False
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def start_stream(self):
        """Start STT stream"""
        if not self.is_available:
            logger.warning("Whisper not available")
            return False
        
        # Wait for model to load if still loading
        max_wait = 30.0
        elapsed = 0.0
        while self.model is None and elapsed < max_wait:
            time.sleep(0.5)
            elapsed += 0.5
        
        if self.model is None:
            logger.error("Whisper model failed to load")
            return False
        
        with self.buffer_lock:
            if self.is_streaming:
                return True
            
            self.is_streaming = True
            self.audio_buffer = []
            self.audio_chunks_received = 0
            self.last_transcription_time = None
            self.first_audio_time = None
            self.first_transcription_time = None
            self.current_interim_text = ""
            self.last_final_text = ""
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_audio_worker, daemon=True)
            self.processing_thread.start()
            
            logger.info("Whisper STT stream started")
            return True
    
    def stop_stream(self):
        """Stop STT stream and process remaining audio"""
        # First, give a moment for any pending audio chunks to be added to buffer
        # Increased wait to ensure all chunks from extended stopping period are added
        time.sleep(0.8)  # Increased from 0.5 to ensure all chunks are added
        
        with self.buffer_lock:
            if not self.is_streaming:
                return
            
            self.is_streaming = False
            
            # Process any remaining audio in buffer
            if len(self.audio_buffer) > 0:
                buffer_duration = sum(len(chunk) for chunk in self.audio_buffer) / self.sample_rate
                logger.info(f"Processing final buffer with {len(self.audio_buffer)} chunks ({buffer_duration:.2f}s)")
                # Process final buffer WITHOUT clearing it first - process all accumulated audio
                self._process_buffer(final=True)
            else:
                logger.warning("Final buffer is empty - no audio to process")
            
            # Clear buffer AFTER processing (done in _process_buffer for final=True)
        
        # Wait for processing thread to finish and for final transcription to be sent
        # This ensures we capture the complete transcription including the end
        max_wait = 8.0  # Increased wait time for final transcription
        waited = 0.0
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=max_wait)
            waited = max_wait if self.processing_thread.is_alive() else 0.0
        
        # Additional wait to ensure final transcription callback is called
        # (transcription processing happens in the thread, callback might take a moment)
        if waited < max_wait:
            remaining_wait = max_wait - waited
            time.sleep(min(remaining_wait, 2.0))  # Wait up to 2 more seconds
        
        logger.info("Whisper STT stream stopped")
    
    def send_audio(self, audio: np.ndarray):
        """Add audio chunk to buffer"""
        if self.model is None:
            return
        
        # Check if streaming - if not, don't accept new audio
        if not self.is_streaming:
            return
        
        if len(audio) == 0:
            return
        
        # Ensure float32 format
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Clip to valid range
        audio = np.clip(audio, -1.0, 1.0)
        
        # Track first audio timestamp
        if self.first_audio_time is None:
            self.first_audio_time = time.time()
        
        with self.buffer_lock:
            self.audio_buffer.append(audio)
            self.audio_chunks_received += 1
            
            # Calculate buffer duration
            buffer_samples = sum(len(chunk) for chunk in self.audio_buffer)
            buffer_duration = buffer_samples / self.sample_rate
            
            # Force processing if buffer is too large (but don't clear buffer - keep all for final)
            if buffer_duration >= self.max_buffer_duration:
                self._process_buffer(final=False)
    
    def _process_audio_worker(self):
        """Background worker that processes audio buffer periodically"""
        while self.is_streaming:
            time.sleep(1.0)  # Check every 1 second (less frequent for better accuracy)
            
            with self.buffer_lock:
                if not self.is_streaming:
                    break
                
                if len(self.audio_buffer) == 0:
                    continue
                
                buffer_samples = sum(len(chunk) for chunk in self.audio_buffer)
                buffer_duration = buffer_samples / self.sample_rate
                
                # Process if we have enough audio and enough time has passed since last transcription
                time_since_last = time.time() - self.last_transcription_time if self.last_transcription_time else float('inf')
                
                # Increased interval for interim results
                # Only process interim if buffer is getting large (to avoid too frequent processing)
                # This ensures we keep all audio for the final transcription
                if buffer_duration >= self.min_buffer_duration and time_since_last >= self.interim_interval:
                    # Only process interim if buffer is getting large (>= 3 seconds)
                    # This prevents clearing buffer too early and losing end of speech
                    if buffer_duration >= 3.0:
                        self._process_buffer(final=False)
    
    def _process_buffer(self, final=False):
        """Process audio buffer and generate transcription"""
        if not self.model or len(self.audio_buffer) == 0:
            return
        
        try:
            # Concatenate all audio chunks
            audio_data = np.concatenate(self.audio_buffer)
            
            # Increased minimum audio length for better accuracy
            if len(audio_data) < self.sample_rate * 1.0:  # At least 1 second (was 0.5)
                return
            
            # Validate audio quality before processing
            audio_level = np.abs(audio_data).max()
            if audio_level < 0.001:  # Too quiet, skip
                logger.debug(f"Skipping transcription - audio too quiet (max={audio_level:.6f})")
                return
            
            # Resample if needed (Whisper expects 16kHz)
            if self.sample_rate != 16000:
                try:
                    from scipy import signal
                    num_samples = int(len(audio_data) * 16000 / self.sample_rate)
                    audio_data = signal.resample(audio_data, num_samples).astype(np.float32)
                except ImportError:
                    # Simple linear interpolation fallback
                    ratio = 16000 / self.sample_rate
                    indices = np.linspace(0, len(audio_data) - 1, int(len(audio_data) * ratio))
                    audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data).astype(np.float32)
            
            # For final transcription, add small silence padding at the end
            # This helps Whisper capture the last word more accurately
            if final:
                # Add 0.3 seconds of silence at the end (helps Whisper detect end of speech)
                silence_padding = int(0.3 * 16000)  # 0.3 seconds at 16kHz
                padding = np.zeros(silence_padding, dtype=np.float32)
                audio_data = np.concatenate([audio_data, padding])
                logger.debug(f"Added {silence_padding/16000:.2f}s silence padding for final transcription")
            
            # Transcribe using Whisper with better settings
            logger.debug(f"Processing {len(audio_data)/16000:.2f}s of audio (final={final})")
            
            # For final transcription, use better settings to capture complete text
            transcribe_kwargs = {
                "language": None,  # Auto-detect language
                "task": "transcribe",
                "fp16": False,  # Use fp32 for compatibility
                "verbose": False,
            }
            
            if final:
                # For final transcription, use best effort settings
                transcribe_kwargs["condition_on_previous_text"] = True  # Use previous text for better continuity
                transcribe_kwargs["initial_prompt"] = self.last_final_text if self.last_final_text else None
                # Use beam_size for better accuracy on final transcription
                transcribe_kwargs["beam_size"] = 5
                # Use best_of for better final results
                transcribe_kwargs["best_of"] = 5
                # Use temperature=0 for more deterministic results
                transcribe_kwargs["temperature"] = 0
            else:
                # For interim, use faster settings
                transcribe_kwargs["condition_on_previous_text"] = True
                transcribe_kwargs["initial_prompt"] = self.last_final_text if self.last_final_text else None
            
            result = self.model.transcribe(audio_data, **transcribe_kwargs)
            
            text = result.get("text", "").strip()
            language = result.get("language", "en")
            segments = result.get("segments", [])
            
            # Filter out very short or low-confidence transcriptions
            if len(text) < 2:  # Too short, likely noise
                logger.debug(f"Skipping transcription - text too short: '{text}'")
                return
            
            if text and self.on_transcript:
                # Calculate confidence from segments
                confidence = 1.0
                if segments:
                    # Use average no_speech_prob to estimate confidence
                    avg_no_speech_prob = np.mean([s.get("no_speech_prob", 0.0) for s in segments])
                    confidence = max(0.0, 1.0 - avg_no_speech_prob)
                    
                    # Filter low-confidence interim results
                    if not final and confidence < 0.3:
                        logger.debug(f"Skipping low-confidence interim transcription: {confidence:.2f}")
                        return
                
                # Map language code to standard format
                lang_map = {
                    "en": "en",
                    "zh": "zh",
                    "ja": "ja",
                    "japanese": "ja",
                    "chinese": "zh"
                }
                detected_lang = lang_map.get(language.lower(), language.lower())
                
                # Track latency
                if self.first_transcription_time is None and self.first_audio_time:
                    latency = time.time() - self.first_audio_time
                    logger.info(f"[Whisper Latency] First transcription: {latency:.3f}s")
                    self.first_transcription_time = time.time()
                
                self.transcriptions_received += 1
                
                if final:
                    # Final transcription - send as final
                    logger.info(f"[Whisper STT #{self.transcriptions_received} FINAL] '{text[:80]}...' (lang={detected_lang})")
                    self.on_transcript(text, True, detected_lang, confidence)
                    self.last_final_text = text
                    self.current_interim_text = ""
                else:
                    # Interim transcription - send as interim
                    logger.info(f"[Whisper STT #{self.transcriptions_received} INTERIM] '{text[:80]}...' (lang={detected_lang})")
                    self.on_transcript(text, False, detected_lang, confidence)
                    self.current_interim_text = text
            
            # Clear buffer after processing (but keep streaming for more audio)
            with self.buffer_lock:
                if not final:
                    # For interim, DON'T clear buffer - keep ALL audio for final transcription
                    # This ensures we don't lose the end of speech
                    # Only update the last transcription time
                    pass
                else:
                    # For final, clear buffer after processing
                    self.audio_buffer = []
                
                self.last_transcription_time = time.time()
                
        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}", exc_info=True)
            with self.buffer_lock:
                self.audio_buffer = []
