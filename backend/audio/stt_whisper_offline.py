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
        # Check buffer size before and after wait to ensure we have all audio
        with self.buffer_lock:
            if not self.is_streaming:
                return
            
            initial_buffer_size = len(self.audio_buffer)
            initial_duration = sum(len(chunk) for chunk in self.audio_buffer) / self.sample_rate if initial_buffer_size > 0 else 0
        
        # Wait to ensure all chunks from extended stopping period are added
        time.sleep(1.0)  # Increased from 0.8
        time.sleep(0.3)  # Additional wait
        
        with self.buffer_lock:
            self.is_streaming = False
            
            # Process any remaining audio in buffer
            if len(self.audio_buffer) > 0:
                buffer_duration = sum(len(chunk) for chunk in self.audio_buffer) / self.sample_rate
                final_buffer_size = len(self.audio_buffer)
                logger.info(f"Processing final buffer: {initial_buffer_size} -> {final_buffer_size} chunks, {initial_duration:.2f}s -> {buffer_duration:.2f}s")
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
            
            # Don't force interim processing - keep all audio for final transcription
            # This ensures we capture complete speech from beginning to end
            # If buffer gets too large, it will be processed when speech ends
    
    def _process_audio_worker(self):
        """Background worker that processes audio buffer periodically"""
        # DISABLED: Don't process interim transcriptions to avoid interference
        # Only process final transcription when speech ends
        # This ensures we always get complete transcriptions with beginning and end
        while self.is_streaming:
            time.sleep(2.0)  # Just wait, don't process interim
            # Interim processing disabled - only final transcription will be processed
            # This ensures complete transcriptions with all audio
    
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
            
            # Normalize audio to improve accuracy (prevent clipping and ensure good dynamic range)
            audio_max = np.abs(audio_data).max()
            if audio_max > 0:
                # Normalize to 0.95 peak to avoid clipping while maximizing dynamic range
                audio_data = audio_data / audio_max * 0.95
            
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
            
            # For final transcription, add silence padding at the end
            # This helps Whisper capture the last word more accurately
            if final:
                # Add 0.5 seconds of silence at the end (increased from 0.3)
                # This gives Whisper more time to process the end of speech
                silence_padding = int(0.5 * 16000)  # 0.5 seconds at 16kHz
                padding = np.zeros(silence_padding, dtype=np.float32)
                audio_data = np.concatenate([audio_data, padding])
                logger.info(f"Added {silence_padding/16000:.2f}s silence padding for final transcription (total audio: {len(audio_data)/16000:.2f}s)")
            
            # Transcribe using Whisper with better settings
            logger.debug(f"Processing {len(audio_data)/16000:.2f}s of audio (final={final})")
            
            # CRITICAL: Restrict to only English, Chinese, and Japanese
            supported_languages = ["en", "zh", "ja"]
            
            # For interim transcriptions, use English (fastest, interim is just for feedback)
            # For final transcriptions, try all 3 and pick the best (more accurate)
            if not final:
                # Use English for interim transcriptions (fast and reliable)
                try:
                    transcribe_kwargs = {
                        "language": "en",  # Use English for interim (fast)
                        "task": "transcribe",
                        "fp16": False,
                        "verbose": False,
                        "condition_on_previous_text": True,
                        "initial_prompt": self.last_final_text if self.last_final_text else None,
                    }
                    result = self.model.transcribe(audio_data, **transcribe_kwargs)
                    language = "en"  # Interim uses English, final will determine actual language
                except Exception as e:
                    logger.error(f"Interim transcription failed: {e}")
                    return  # Skip this interim transcription
            else:
                # For final transcription: use Whisper's auto-detection first, then verify with scoring
                # First, try transcribing with language=None to let Whisper auto-detect
                detected_language = None
                
                try:
                    # Try auto-detection by transcribing without specifying language
                    # Whisper will detect the language automatically
                    auto_result = self.model.transcribe(
                        audio_data,
                        language=None,  # Auto-detect
                        task="transcribe",
                        fp16=False,
                        verbose=False,
                        condition_on_previous_text=True,
                        initial_prompt=self.last_final_text if self.last_final_text else None,
                    )
                    
                    # Get detected language from result
                    detected_language = auto_result.get("language", "en")
                    logger.info(f"Whisper auto-detected language: '{detected_language}'")
                    
                    # Map to our supported languages
                    language_map = {
                        'zh': 'zh', 'chinese': 'zh',
                        'en': 'en', 'english': 'en',
                        'ja': 'ja', 'japanese': 'ja'
                    }
                    
                    detected_language_lower = detected_language.lower() if detected_language else "en"
                    if detected_language_lower in language_map:
                        detected_language = language_map[detected_language_lower]
                    elif detected_language not in supported_languages:
                        detected_language = None  # Not supported, will try all languages
                        logger.warning(f"Auto-detected language '{detected_language}' not in supported list")
                except Exception as e:
                    logger.warning(f"Auto-detection failed: {e}, will try all languages")
                    detected_language = None
                
                # If auto-detection succeeded and language is supported, use it with better settings
                if detected_language and detected_language in supported_languages:
                    try:
                        transcribe_kwargs = {
                            "language": detected_language,
                            "task": "transcribe",
                            "fp16": False,
                            "verbose": False,
                            "condition_on_previous_text": True,
                            "initial_prompt": self.last_final_text if self.last_final_text else None,
                            "beam_size": 5,  # Increased for better accuracy
                            "best_of": 5,
                            "temperature": 0,
                            "compression_ratio_threshold": 2.4,
                            "logprob_threshold": -1.0,
                            "no_speech_threshold": 0.6,
                            "word_timestamps": True,
                        }
                        try:
                            import torch
                            if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                                transcribe_kwargs["fp16"] = True
                        except:
                            pass
                        
                        result = self.model.transcribe(audio_data, **transcribe_kwargs)
                        language = detected_language
                        logger.info(f"Transcribed with auto-detected language '{language}'")
                    except Exception as e:
                        logger.warning(f"Transcription with detected language '{detected_language}' failed: {e}, trying fallback")
                        detected_language = None  # Fall through to fallback
                
                # Fallback: if auto-detection failed or detected unsupported language, try all 3 and pick best
                if not detected_language or detected_language not in supported_languages:
                    best_result = None
                    best_language = "en"
                    best_score = -float('inf')
                    
                    for lang in supported_languages:
                        try:
                            transcribe_kwargs = {
                                "language": lang,
                                "task": "transcribe",
                                "fp16": False,
                                "verbose": False,
                                "condition_on_previous_text": True,
                                "initial_prompt": self.last_final_text if self.last_final_text else None,
                                "beam_size": 5,  # Increased for better accuracy
                                "best_of": 5,
                                "temperature": 0,
                                "compression_ratio_threshold": 2.4,
                                "logprob_threshold": -1.0,
                                "no_speech_threshold": 0.6,
                                "word_timestamps": True,
                            }
                            try:
                                import torch
                                if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                                    transcribe_kwargs["fp16"] = True
                            except:
                                pass
                            
                            test_result = self.model.transcribe(audio_data, **transcribe_kwargs)
                            test_text = test_result.get("text", "").strip()
                            test_segments = test_result.get("segments", [])
                            
                            if not test_text:
                                continue
                            
                            # Improved scoring: prioritize avg_logprob (confidence) over text length
                            # Higher logprob = better match for that language
                            if test_segments:
                                avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in test_segments])
                                # Use logprob as primary score (it's already a good indicator)
                                # Add small bonus for longer text only if logprob is similar
                                score = avg_logprob + (len(test_text) * 0.0001)  # Reduced text length weight
                            else:
                                score = -1.0  # No segments = low score
                            
                            if score > best_score:
                                best_score = score
                                best_result = test_result
                                best_language = lang
                        except Exception as e:
                            logger.warning(f"Error transcribing with language {lang}: {e}")
                            continue
                    
                    # Use the best result
                    if best_result:
                        result = best_result
                        language = best_language
                        logger.info(f"Selected language '{language}' (score: {best_score:.3f})")
                    else:
                        # Final fallback: use English
                        logger.warning("All language attempts failed, using English as fallback")
                        language = "en"
                        transcribe_kwargs = {
                            "language": "en",
                            "task": "transcribe",
                            "fp16": False,
                            "verbose": False,
                            "condition_on_previous_text": True,
                            "initial_prompt": self.last_final_text if self.last_final_text else None,
                            "beam_size": 3,
                            "best_of": 3,
                            "temperature": 0,
                        }
                        result = self.model.transcribe(audio_data, **transcribe_kwargs)
            
            text = result.get("text", "").strip()
            segments = result.get("segments", [])
            
            # Final verification: ensure language is one of our 3
            if language not in supported_languages:
                language = "en"
            
            # Filter out very short or low-confidence transcriptions
            if len(text) < 2:  # Too short, likely noise
                logger.debug(f"Skipping transcription - text too short: '{text}'")
                return
            
            if text and self.on_transcript:
                # Calculate confidence from segments
                confidence = 1.0
                if segments:
                    if final:
                        # For final transcriptions, use average logprob for better confidence estimation
                        # and cap at 100% since it's the final result
                        avg_logprob = np.mean([s.get("avg_logprob", 0.0) for s in segments])
                        # Convert logprob to confidence (logprob is typically negative, higher is better)
                        # Typical range: -1.0 (good) to -0.5 (excellent)
                        confidence = min(1.0, max(0.8, 1.0 + avg_logprob))  # Map to 0.8-1.0 range
                        # For final transcription, show at least 95% to indicate completion
                        confidence = max(0.95, confidence)
                    else:
                        # For interim, use average no_speech_prob to estimate confidence
                        avg_no_speech_prob = np.mean([s.get("no_speech_prob", 0.0) for s in segments])
                        confidence = max(0.0, 1.0 - avg_no_speech_prob)
                    
                    # Filter low-confidence interim results
                    if not final and confidence < 0.3:
                        logger.debug(f"Skipping low-confidence interim transcription: {confidence:.2f}")
                        return
                
                # Language is already normalized to en/zh/ja above
                detected_lang = language
                
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
