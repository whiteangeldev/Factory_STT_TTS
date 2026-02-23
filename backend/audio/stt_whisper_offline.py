"""Offline Speech-to-Text using OpenAI Whisper"""
import numpy as np
import logging
import threading
import time
import re
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
        self.min_buffer_duration = 2.0  # Minimum 2 seconds before first interim
        self.interim_interval = 2.5  # Process interim results every 2.5 seconds (reduced for lower latency)
        self.last_transcription_time = None
        self.last_interim_time = None  # Track last interim processing time
        self.last_processed_chunk_count = 0  # Track how many chunks we've processed (avoid reprocessing)
        self.is_processing_interim = False  # Prevent concurrent interim processing
        self.audio_chunks_received = 0
        self.first_audio_time = None
        self.first_transcription_time = None
        self.transcriptions_received = 0
        self.current_interim_text = ""  # Track interim text
        self.last_sent_interim_text = ""  # Track last sent interim text for incremental updates (current sentence only)
        self.last_full_text = ""  # Track last full text to detect what's new
        self.last_final_text = ""  # Track last final text
        self.detected_language_interim = None  # Cache detected language for interim
        
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
            self.last_interim_time = None  # Reset for new stream
            self.detected_language_interim = None  # Reset for new stream
            self.last_sent_interim_text = ""  # Reset for new stream
            self.last_full_text = ""  # Reset for new stream
            self.last_processed_chunk_count = 0  # Reset for new stream
            self.is_processing_interim = False  # Reset for new stream
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
            
            # Trigger interim processing if enough audio accumulated
            # The worker thread will handle the actual processing
            current_time = time.time()
            if (buffer_duration >= self.min_buffer_duration and 
                (self.last_interim_time is None or 
                 current_time - self.last_interim_time >= self.interim_interval)):
                # Worker thread will process - just mark that we should process
                pass
    
    def _process_audio_worker(self):
        """Background worker that processes audio buffer periodically for real-time transcription"""
        while self.is_streaming:
            try:
                with self.buffer_lock:
                    if len(self.audio_buffer) == 0:
                        time.sleep(0.5)
                        continue
                    
                    buffer_samples = sum(len(chunk) for chunk in self.audio_buffer)
                    buffer_duration = buffer_samples / self.sample_rate
                    
                    # Process interim if we have enough audio and enough time has passed
                    current_time = time.time()
                    should_process_interim = (
                        buffer_duration >= self.min_buffer_duration and
                        (self.last_interim_time is None or 
                         current_time - self.last_interim_time >= self.interim_interval)
                    )
                    
                    if should_process_interim and not self.is_processing_interim:
                        # Check if we have new audio to process OR enough time passed for refinement
                        current_chunk_count = len(self.audio_buffer)
                        time_since_last = current_time - self.last_interim_time if self.last_interim_time else float('inf')
                        
                        # Process if:
                        # 1. We have new audio chunks, OR
                        # 2. Enough time passed (allows Whisper to refine same audio)
                        has_new_audio = current_chunk_count > self.last_processed_chunk_count
                        should_refine = time_since_last >= self.interim_interval and current_chunk_count > 0
                        
                        if has_new_audio or should_refine:
                            # Process the FULL buffer (not sliding window)
                            # This ensures we get complete context and proper incremental updates
                            processing_buffer = self.audio_buffer.copy()
                            self.last_interim_time = current_time
                            self.is_processing_interim = True
                            
                            # Process in separate thread to avoid blocking audio capture
                            def process_interim():
                                try:
                                    # Process the full buffer for interim transcription
                                    self._process_audio_buffer(processing_buffer, final=False)
                                    # Update processed chunk count after successful processing
                                    with self.buffer_lock:
                                        if has_new_audio:
                                            self.last_processed_chunk_count = len(processing_buffer)
                                except Exception as e:
                                    logger.error(f"Error processing interim transcription: {e}", exc_info=True)
                                finally:
                                    # CRITICAL: Always clear the flag, even on error, to allow continued processing
                                    with self.buffer_lock:
                                        self.is_processing_interim = False
                            
                            threading.Thread(target=process_interim, daemon=True).start()
                
                time.sleep(0.5)  # Check every 0.5 seconds
            except Exception as e:
                logger.error(f"Error in audio worker: {e}")
                time.sleep(1.0)
    
    def _detect_sentence_end(self, old_text: str, new_text: str) -> bool:
        """
        Detect if a sentence has ended in the new text.
        Looks for sentence-ending punctuation (. ! ?) that wasn't in old text.
        Only detects if there's substantial content and clear sentence structure.
        """
        if not old_text or not new_text:
            return False
        
        # Normalize for comparison
        old_normalized = old_text.strip()
        new_normalized = new_text.strip()
        
        # Find sentence endings in new text
        sentence_endings = list(re.finditer(r'[.!?]+', new_normalized))
        
        if not sentence_endings:
            return False
        
        # Check each sentence ending
        for match in sentence_endings:
            end_pos = match.end()
            
            # Check if this ending is new (after the old text length)
            # Allow some tolerance for whitespace differences
            if end_pos > len(old_normalized) + 5:  # New content after old text
                # Extract the sentence up to this ending
                sentence = new_normalized[:end_pos].strip()
                
                # Validate: must have substantial content (at least 15 chars, 3 words)
                words = sentence.split()
                if len(sentence) >= 15 and len(words) >= 3:
                    # Check if this is a real sentence (not just punctuation)
                    # Must have at least one letter/word before the ending
                    text_before_ending = sentence[:-len(match.group())].strip()
                    if len(text_before_ending) >= 10 and any(c.isalnum() for c in text_before_ending):
                        return True
        
        return False
    
    def _extract_completed_sentence(self, old_text: str, new_text: str) -> str:
        """
        Extract the completed sentence (up to the first sentence ending that's new).
        """
        if not new_text:
            return ""
        
        old_len = len(old_text) if old_text else 0
        
        # Find first sentence ending that's new (after old text)
        match = re.search(r'[.!?]+', new_text)
        if match:
            end_pos = match.end()
            # Only return if this ending is new (after old text)
            if end_pos > old_len + 5:  # New sentence ending
                sentence = new_text[:end_pos].strip()
                # Validate it's a real sentence
                if len(sentence) >= 15:
                    words = sentence.split()
                    if len(words) >= 3:
                        return sentence
        
        return ""
    
    def _extract_remaining_text_after_sentence(self, text: str) -> str:
        """
        Extract text that comes after the first completed sentence.
        """
        if not text:
            return ""
        
        # Find first sentence ending
        match = re.search(r'[.!?]+', text)
        if match:
            end_pos = match.end()
            remaining = text[end_pos:].strip()
            # Remove leading whitespace/punctuation
            remaining = re.sub(r'^[\s,;:]+', '', remaining)
            return remaining
        
        return ""
    
    def _get_incremental_text(self, old_text: str, new_text: str) -> str:
        """
        Extract only the new words from new_text compared to old_text.
        This creates a streaming effect where only new words are added.
        Uses fuzzy matching to handle Whisper's refinements.
        """
        if not old_text:
            return new_text  # First update, return all text
        
        old_text = old_text.strip()
        new_text = new_text.strip()
        
        if not new_text:
            return ""
        
        # Normalize texts for comparison (remove punctuation differences)
        old_normalized = re.sub(r'[^\w\s]', '', old_text.lower())
        new_normalized = re.sub(r'[^\w\s]', '', new_text.lower())
        
        # Check if new text contains old text (Whisper might refine but keep content)
        if old_normalized in new_normalized:
            # Find where old text ends in new text
            old_words = old_normalized.split()
            new_words = new_normalized.split()
            
            # Find longest matching suffix of old_text in new_text
            # This handles cases where Whisper refines the beginning
            best_match_start = 0
            best_match_len = 0
            
            for start_idx in range(len(new_words)):
                match_len = 0
                for i in range(min(len(old_words), len(new_words) - start_idx)):
                    if old_words[i] == new_words[start_idx + i]:
                        match_len += 1
                    else:
                        break
                if match_len > best_match_len:
                    best_match_len = match_len
                    best_match_start = start_idx
            
            # If we found a good match, return words after the match
            if best_match_len >= len(old_words) * 0.7:  # At least 70% match
                remaining_words = new_text.split()[best_match_start + best_match_len:]
                if remaining_words:
                    return " ".join(remaining_words)
        
        # Fallback: simple word-by-word comparison from start
        old_words = old_text.split()
        new_words = new_text.split()
        
        # Find how many words match from the start
        common_prefix_len = 0
        min_len = min(len(old_words), len(new_words))
        for i in range(min_len):
            # Normalize words for comparison (ignore punctuation/case)
            old_word = re.sub(r'[^\w]', '', old_words[i].lower())
            new_word = re.sub(r'[^\w]', '', new_words[i].lower())
            if old_word == new_word:
                common_prefix_len += 1
            else:
                break
        
        # CRITICAL: If beginning words don't match, Whisper may have refined/removed them
        # In this case, show full new text to preserve the beginning
        if common_prefix_len == 0 and len(new_words) > 0:
            # Beginning changed - show full text to preserve it
            return new_text
        
        # If new text is significantly longer, return the difference
        if len(new_words) > common_prefix_len:
            new_words_only = new_words[common_prefix_len:]
            return " ".join(new_words_only)
        
        # If texts are similar length but different, check if beginning was preserved
        if abs(len(new_words) - len(old_words)) <= 2:
            # If first few words match, it's likely a refinement - show incremental
            if common_prefix_len >= 2:
                # Beginning preserved, show incremental
                if len(new_words) > common_prefix_len:
                    return " ".join(new_words[common_prefix_len:])
            else:
                # Beginning changed, show full text
                return new_text
        
        return ""
    
    def _process_buffer(self, final=False):
        """Process the main audio buffer and generate transcription"""
        with self.buffer_lock:
            if len(self.audio_buffer) == 0:
                return
            buffer_to_process = self.audio_buffer.copy()
        
        # Process the buffer (without holding the lock during processing)
        self._process_audio_buffer(buffer_to_process, final=final)
        
        # Clear buffer after processing if final
        if final:
            with self.buffer_lock:
                self.audio_buffer = []
    
    def _process_audio_buffer(self, audio_buffer_list, final=False):
        """Process a specific audio buffer list and generate transcription"""
        if not self.model or len(audio_buffer_list) == 0:
            return
        
        try:
            # Concatenate all audio chunks
            audio_data = np.concatenate(audio_buffer_list)
            
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
            
            # For interim transcriptions: use faster settings with quick language detection
            # For final transcriptions: use full language detection and better settings
            if not final:
                # Quick language detection on first interim (use first 2 seconds)
                if self.detected_language_interim is None and len(audio_data) >= self.sample_rate * 2:
                    try:
                        detection_audio = audio_data[:int(2.0 * self.sample_rate)]
                        # Quick detection with minimal processing
                        detection_result = self.model.transcribe(
                            detection_audio,
                            language=None,  # Auto-detect
                            task="transcribe",
                            fp16=False,
                            verbose=False,
                            beam_size=1,
                            best_of=1,
                            temperature=0,
                        )
                        detected_lang = detection_result.get("language", "en")
                        # Map to supported languages
                        if detected_lang.lower() in ['zh', 'chinese', 'cmn', 'zho']:
                            self.detected_language_interim = 'zh'
                        elif detected_lang.lower() in ['ja', 'japanese']:
                            self.detected_language_interim = 'ja'
                        else:
                            self.detected_language_interim = 'en'
                        logger.info(f"Quick language detection for interim: '{self.detected_language_interim}' (from '{detected_lang}')")
                    except Exception as e:
                        logger.warning(f"Quick language detection failed: {e}, defaulting to English")
                        self.detected_language_interim = 'en'
                
                # Use detected language or default to English
                interim_lang = self.detected_language_interim or 'en'
                
                try:
                    transcribe_kwargs = {
                        "language": interim_lang,
                        "task": "transcribe",
                        "fp16": False,
                        "verbose": False,
                        "condition_on_previous_text": True,
                        "initial_prompt": self.last_final_text if self.last_final_text else None,
                        "beam_size": 2,  # Faster for interim (was 1, but 2 gives better quality)
                        "best_of": 2,  # Faster for interim
                        "temperature": 0,
                    }
                    try:
                        import torch
                        if torch.cuda.is_available() or (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                            transcribe_kwargs["fp16"] = True
                    except:
                        pass
                    
                    result = self.model.transcribe(audio_data, **transcribe_kwargs)
                    language = interim_lang
                except Exception as e:
                    logger.error(f"Interim transcription failed: {e}")
                    return  # Skip this interim transcription
            else:
                # For final transcription: detect language FIRST (fast), then transcribe with detected language
                # CRITICAL: This prevents translation - we transcribe in the actual spoken language
                # OPTIMIZED: Use minimal audio (1 second) for fast detection
                detected_language = None
                
                # Step 1: Fast language detection using sufficient audio sample
                # CRITICAL: Use more audio (2-3 seconds) for reliable detection, especially for microphone mode
                # Microphone audio often has noise at the start, so we need more audio for accurate detection
                try:
                    # Use 2-3 seconds for detection (more reliable than 1 second)
                    # If audio is short (< 5 seconds), use entire audio for better accuracy
                    audio_duration = len(audio_data) / 16000.0
                    if audio_duration < 5.0:
                        # Short audio - use entire audio for detection (more accurate)
                        detection_audio = audio_data
                        logger.info(f"Using entire audio ({audio_duration:.2f}s) for language detection")
                    else:
                        # Longer audio - use first 3 seconds (balance between speed and accuracy)
                        detection_audio = audio_data[:min(len(audio_data), 16000 * 3)]
                        logger.info(f"Using first 3 seconds for language detection")
                    
                    # CRITICAL: Restrict language detection to ONLY the 3 supported languages (en, zh, ja)
                    # This is the most essential initial step - we should never detect other languages
                    language_probs = None
                    detected_language = None
                    top_prob = 0.0
                    
                    # Language mapping for variants
                    language_map = {
                        'zh': 'zh', 'chinese': 'zh', 'cmn': 'zh', 'zho': 'zh',
                        'en': 'en', 'english': 'en',
                        'ja': 'ja', 'japanese': 'ja'
                    }
                    
                    try:
                        # Method 1: Try detect_language if available, but filter to only our 3 languages
                        try:
                            _, all_language_probs = self.model.detect_language(detection_audio)
                            if all_language_probs:
                                # Filter to only supported languages and their variants
                                filtered_probs = {}
                                for lang_code, prob in all_language_probs.items():
                                    lang_lower = lang_code.lower()
                                    # Check if this language maps to one of our supported languages
                                    if lang_lower in language_map:
                                        mapped_lang = language_map[lang_lower]
                                        # Sum probabilities for variants (e.g., 'cmn' and 'zh' both map to 'zh')
                                        if mapped_lang not in filtered_probs:
                                            filtered_probs[mapped_lang] = 0.0
                                        filtered_probs[mapped_lang] += prob
                                
                                # Pick the language with highest probability among our 3 supported ones
                                if filtered_probs:
                                    detected_language = max(filtered_probs, key=filtered_probs.get)
                                    top_prob = filtered_probs[detected_language]
                                    logger.info(f"Fast language detection (restricted to en/zh/ja): '{detected_language}' (prob={top_prob:.3f})")
                                else:
                                    # No supported language detected - will try all 3 explicitly
                                    detected_language = None
                                    logger.info(f"Language detection found no supported languages (en/zh/ja) in top results, will try all 3 explicitly")
                        except (AttributeError, TypeError):
                            # detect_language not available - try each of the 3 languages explicitly
                            logger.info("detect_language() not available, trying each of 3 supported languages explicitly")
                            detected_language = None
                            raise  # Fall through to explicit testing
                        
                        # Method 2: If detect_language failed or not available, test each of the 3 languages explicitly
                        if detected_language is None:
                            best_detection_lang = None
                            best_detection_score = -float('inf')
                            
                            # CRITICAL: Ensure detection_audio is properly formatted numpy array
                            # Whisper expects numpy array, not torch tensor
                            if not isinstance(detection_audio, np.ndarray):
                                detection_audio = np.array(detection_audio, dtype=np.float32)
                            else:
                                detection_audio = detection_audio.astype(np.float32)
                            if not detection_audio.flags['C_CONTIGUOUS']:
                                detection_audio = np.ascontiguousarray(detection_audio)
                            
                            # Ensure audio is normalized (Whisper expects values between -1 and 1)
                            audio_max = np.abs(detection_audio).max()
                            if audio_max > 1.0:
                                detection_audio = detection_audio / audio_max
                            
                            for test_lang in supported_languages:
                                try:
                                    # Quick transcription test with minimal settings for fast detection
                                    test_result = self.model.transcribe(
                                        detection_audio,  # Pass numpy array directly, Whisper handles conversion
                                        language=test_lang,
                                        task="transcribe",  # CRITICAL: transcribe, not translate
                                        fp16=False,
                                        verbose=False,
                                        beam_size=1,  # Minimal for speed
                                        best_of=1,  # Minimal for speed
                                        temperature=0,
                                    )
                                    
                                    test_text = test_result.get("text", "").strip()
                                    result_detected_lang = test_result.get("language", test_lang)
                                    result_detected_lang_lower = result_detected_lang.lower() if result_detected_lang else ""
                                    
                                    if not test_text:
                                        logger.debug(f"Language '{test_lang}': No text produced, skipping")
                                        continue
                                    
                                    # CRITICAL: Verify text is actually in the requested language (not translated)
                                    is_correct_language = False
                                    if test_lang == 'zh':
                                        # Chinese: must contain Chinese characters
                                        has_chinese = any('\u4e00' <= char <= '\u9fff' for char in test_text)
                                        is_correct_language = has_chinese
                                    elif test_lang == 'ja':
                                        # Japanese: must contain Hiragana, Katakana, or Kanji
                                        has_japanese = any(
                                            ('\u3040' <= char <= '\u309F') or  # Hiragana
                                            ('\u30A0' <= char <= '\u30FF') or  # Katakana
                                            ('\u4e00' <= char <= '\u9fff')     # Kanji
                                            for char in test_text
                                        )
                                        is_correct_language = has_japanese
                                    elif test_lang == 'en':
                                        # English: should NOT contain CJK characters
                                        has_cjk = any(
                                            ('\u4e00' <= char <= '\u9fff') or  # Chinese/Kanji
                                            ('\u3040' <= char <= '\u309F') or  # Hiragana
                                            ('\u30A0' <= char <= '\u30FF')      # Katakana
                                            for char in test_text
                                        )
                                        is_correct_language = not has_cjk
                                    
                                    # Only accept if: 1) detected language matches, 2) text is in correct language
                                    if not is_correct_language:
                                        logger.debug(f"Language '{test_lang}': Text language mismatch (translation detected), skipping")
                                        continue
                                    
                                    if result_detected_lang_lower not in language_map.get(test_lang, [test_lang]):
                                        logger.debug(f"Language '{test_lang}': Whisper detected '{result_detected_lang}' (mismatch), skipping")
                                        continue
                                    
                                    # Calculate confidence score
                                    segments = test_result.get("segments", [])
                                    if segments:
                                        avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in segments])
                                        score = avg_logprob
                                        
                                        # For Chinese/Japanese, require higher confidence (stricter)
                                        if test_lang in ['zh', 'ja'] and avg_logprob < -0.7:
                                            logger.debug(f"Language '{test_lang}': Low logprob ({avg_logprob:.3f}), skipping")
                                            continue
                                        
                                        if score > best_detection_score:
                                            best_detection_score = score
                                            best_detection_lang = test_lang
                                            top_prob = max(0.0, min(1.0, 1.0 + avg_logprob))
                                            logger.debug(f"Language '{test_lang}': Valid detection (score={score:.3f}, text='{test_text[:30]}...')")
                                    else:
                                        # No segments - very low confidence
                                        if best_detection_score == -float('inf'):
                                            best_detection_lang = test_lang
                                            top_prob = 0.2  # Very low confidence
                                            
                                except Exception as e:
                                    logger.warning(f"Error testing language '{test_lang}' for detection: {e}")
                                    continue
                            
                            if best_detection_lang:
                                detected_language = best_detection_lang
                                logger.info(f"✅ Language detected: '{detected_language}' (confidence={top_prob:.3f}, score={best_detection_score:.3f})")
                            else:
                                logger.warning("❌ Could not detect language from 3 supported languages, will try all 3 in full transcription")
                                detected_language = None
                                
                    except Exception as e:
                        logger.error(f"❌ Language detection failed: {e}", exc_info=True)
                        detected_language = None
                    
                    # CRITICAL: Check confidence - if low confidence, don't trust detection
                    # This is especially important for microphone mode where initial audio may be noisy
                    if detected_language and detected_language in supported_languages:
                        if top_prob < 0.3:  # Very low confidence threshold (stricter)
                            logger.warning(f"⚠️ Low confidence detection ({top_prob:.3f}) for '{detected_language}' - will try all 3 languages for accuracy")
                            detected_language = None  # Force trying all 3 languages
                        elif top_prob < 0.5:  # Medium confidence - log warning but still use it
                            logger.info(f"⚠️ Medium confidence detection ({top_prob:.3f}) for '{detected_language}' - will verify with full transcription")
                except Exception as e:
                    logger.error(f"❌ Language detection failed: {e}", exc_info=True)
                    detected_language = None
                    top_prob = 0.0
                
                # Step 2: If detected language is supported and confident, transcribe with it
                # CRITICAL: Only use detected language if confidence is high enough
                if detected_language and detected_language in supported_languages and top_prob >= 0.3:
                    try:
                        transcribe_kwargs = {
                            "language": detected_language,
                            "task": "transcribe",  # CRITICAL: transcribe, not translate
                            "fp16": False,
                            "verbose": False,
                            "condition_on_previous_text": True,
                            "initial_prompt": self.last_final_text if self.last_final_text else None,
                            "beam_size": 5,
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
                        result_lang = result.get("language", detected_language)
                        result_text = result.get("text", "").strip()
                        result_segments = result.get("segments", [])
                        
                        # CRITICAL: Verify the result is actually in the detected language (not translated)
                        is_correct_language = False
                        if detected_language == 'zh':
                            is_correct_language = any('\u4e00' <= char <= '\u9fff' for char in result_text)
                        elif detected_language == 'ja':
                            is_correct_language = any(
                                ('\u3040' <= char <= '\u309F') or
                                ('\u30A0' <= char <= '\u30FF') or
                                ('\u4e00' <= char <= '\u9fff')
                                for char in result_text
                            )
                        elif detected_language == 'en':
                            has_cjk = any(
                                ('\u4e00' <= char <= '\u9fff') or
                                ('\u3040' <= char <= '\u309F') or
                                ('\u30A0' <= char <= '\u30FF')
                                for char in result_text
                            )
                            is_correct_language = not has_cjk
                        
                        # Check if result language matches detected language
                        result_lang_lower = result_lang.lower() if result_lang else ""
                        lang_variants_check = {
                            'zh': ['zh', 'chinese', 'cmn', 'zho'],
                            'en': ['en', 'english'],
                            'ja': ['ja', 'japanese']
                        }
                        
                        if is_correct_language and result_lang_lower in lang_variants_check.get(detected_language, [detected_language]):
                            # Match confirmed - use this result
                            language = detected_language
                            if result_segments:
                                avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in result_segments])
                                confidence_estimate = max(0.0, min(1.0, 1.0 + avg_logprob))
                                logger.info(f"✅ Transcribed with detected language '{language}' (task=transcribe, detected='{result_lang}', confidence={confidence_estimate:.3f})")
                            else:
                                logger.info(f"✅ Transcribed with detected language '{language}' (task=transcribe, detected='{result_lang}')")
                        else:
                            # Mismatch - initial detection was wrong, try all languages
                            logger.warning(f"❌ Language verification failed: detected '{detected_language}' but result shows '{result_lang}' or wrong text language - trying all languages")
                            detected_language = None  # Fall through to trying all languages
                    except Exception as e:
                        logger.warning(f"Transcription with detected language '{detected_language}' failed: {e}, trying all languages")
                        detected_language = None
                
                # Step 3: If detection failed or language not supported, try all 3 and pick best
                # CRITICAL: Only accept results where detected language matches requested language
                # This prevents translation - ensures transcription in actual spoken language
                if not detected_language or detected_language not in supported_languages:
                    best_result = None
                    best_language = None
                    best_score = -float('inf')
                    
                    # Language variants for matching
                    lang_variants = {
                        'zh': ['zh', 'chinese', 'cmn', 'zho'],
                        'en': ['en', 'english'],
                        'ja': ['ja', 'japanese']
                    }
                    
                    for lang in supported_languages:
                        try:
                            transcribe_kwargs = {
                                "language": lang,
                                "task": "transcribe",  # CRITICAL: transcribe, not translate
                                "fp16": False,
                                "verbose": False,
                                "condition_on_previous_text": True,
                                "initial_prompt": self.last_final_text if self.last_final_text else None,
                                "beam_size": 5,
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
                            result_detected_lang = test_result.get("language", lang)
                            
                            if not test_text:
                                continue
                            
                            # CRITICAL: Verify text is actually in the requested language (not translated)
                            is_correct_language = False
                            if lang == 'zh':
                                is_correct_language = any('\u4e00' <= char <= '\u9fff' for char in test_text)
                            elif lang == 'ja':
                                is_correct_language = any(
                                    ('\u3040' <= char <= '\u309F') or
                                    ('\u30A0' <= char <= '\u30FF') or
                                    ('\u4e00' <= char <= '\u9fff')
                                    for char in test_text
                                )
                            elif lang == 'en':
                                has_cjk = any(
                                    ('\u4e00' <= char <= '\u9fff') or
                                    ('\u3040' <= char <= '\u309F') or
                                    ('\u30A0' <= char <= '\u30FF')
                                    for char in test_text
                                )
                                is_correct_language = not has_cjk
                            
                            if not is_correct_language:
                                logger.info(f"Rejecting '{lang}': Text language mismatch (translation detected), text='{test_text[:50]}...'")
                                continue
                            
                            # CRITICAL: Only accept if Whisper's detected language matches requested language
                            result_lang_lower = result_detected_lang.lower() if result_detected_lang else ""
                            if result_lang_lower not in lang_variants.get(lang, [lang]):
                                logger.info(f"Rejecting '{lang}': Whisper detected '{result_detected_lang}' (mismatch - would be translation)")
                                continue
                            
                            # Improved scoring: prioritize avg_logprob (confidence) over text length
                            # Higher logprob = better match for that language
                            if test_segments:
                                avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in test_segments])
                                # Use logprob as primary score (it's already a good indicator)
                                # Add small bonus for longer text only if logprob is similar
                                score = avg_logprob + (len(test_text) * 0.0001)
                            else:
                                score = -1.0
                            
                            logger.info(f"Language '{lang}': detected='{result_detected_lang}' (match), score={score:.3f}, text='{test_text[:50]}...'")
                            
                            if score > best_score:
                                best_score = score
                                best_result = test_result
                                best_language = lang
                        except Exception as e:
                            logger.warning(f"Error transcribing with language {lang}: {e}")
                            continue
                    
                    # Use the best result (only if we found a valid match)
                    if best_result and best_language:
                        result = best_result
                        language = best_language
                        final_detected = result.get("language", language)
                        result_text = result.get("text", "").strip()
                        logger.info(f"Selected language '{language}' (score: {best_score:.3f}, detected: '{final_detected}', text: '{result_text[:50]}...')")
                    else:
                        # Final fallback: use English, but log warning
                        logger.warning("All language attempts failed or were rejected (no language matches), using English as fallback")
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
                    self.last_sent_interim_text = ""  # Reset for next recording
                    self.last_full_text = ""  # Reset for next recording
                else:
                    # Interim transcription - simplified approach: work with full text, track current sentence
                    # CRITICAL: Always continue processing - never return early
                    
                    # If this is the first transcription, send full text immediately
                    if not self.last_sent_interim_text:
                        # First update - send full text to preserve beginning
                        # CRITICAL: Always send full text on first update, even if it seems incomplete
                        # Whisper might refine it later, but we need to show what we have
                        logger.info(f"[Whisper STT #{self.transcriptions_received} INTERIM] First update: '{text[:80]}...' (len={len(text)})")
                        try:
                            self.on_transcript(text, False, detected_lang, confidence, incremental_update=text)
                        except TypeError:
                            self.on_transcript(text, False, detected_lang, confidence)
                        except Exception as e:
                            logger.error(f"Error sending first transcription: {e}", exc_info=True)
                        self.last_sent_interim_text = text
                        self.current_interim_text = text
                        self.last_full_text = text
                        return
                    
                    # For subsequent updates, check if sentence ended
                    sentence_ended = self._detect_sentence_end(self.last_sent_interim_text, text)
                    
                    if sentence_ended:
                        # Sentence ended - extract and finalize
                        final_sentence = self._extract_completed_sentence(self.last_sent_interim_text, text)
                        if final_sentence and len(final_sentence.strip()) > 10:
                            # Send completed sentence as final
                            logger.info(f"[Whisper STT #{self.transcriptions_received} SENTENCE_END] '{final_sentence[:80]}...'")
                            try:
                                self.on_transcript(final_sentence, True, detected_lang, confidence)
                            except Exception as e:
                                logger.error(f"Error sending final sentence: {e}", exc_info=True)
                            self.last_final_text = final_sentence
                            
                            # Extract remaining text after sentence
                            remaining_text = self._extract_remaining_text_after_sentence(text)
                            
                            # Reset tracking for new sentence
                            self.last_sent_interim_text = ""
                            
                            # Start new sentence with remaining text (if any)
                            if remaining_text and len(remaining_text.strip()) > 3:  # Lower threshold to continue processing
                                logger.info(f"[Whisper STT #{self.transcriptions_received} INTERIM] New sentence: '{remaining_text[:50]}...'")
                                try:
                                    self.on_transcript(remaining_text, False, detected_lang, confidence, incremental_update=remaining_text)
                                except Exception as e:
                                    logger.error(f"Error sending new sentence: {e}", exc_info=True)
                                self.last_sent_interim_text = remaining_text
                                self.current_interim_text = remaining_text
                            else:
                                # No remaining text - reset and wait for next update
                                # CRITICAL: Don't return - continue processing
                                self.current_interim_text = ""
                                self.last_sent_interim_text = ""
                                logger.debug("Sentence ended, no remaining text - waiting for next update")
                        else:
                            # Couldn't extract sentence - fall through to incremental update
                            logger.debug("Sentence end detected but couldn't extract, using incremental")
                            sentence_ended = False
                    
                    if not sentence_ended:
                        # No sentence end - send incremental update
                        incremental_text = self._get_incremental_text(self.last_sent_interim_text, text)
                        
                        if incremental_text:
                            logger.info(f"[Whisper STT #{self.transcriptions_received} INTERIM] New: '{incremental_text[:50]}...' (Full: '{text[:80]}...')")
                            try:
                                self.on_transcript(text, False, detected_lang, confidence, incremental_update=incremental_text)
                            except TypeError:
                                self.on_transcript(text, False, detected_lang, confidence)
                            except Exception as e:
                                logger.error(f"Error sending interim transcription: {e}", exc_info=True)
                            # CRITICAL: Always update tracking, even if sending failed
                            self.last_sent_interim_text = text
                            self.current_interim_text = text
                        else:
                            # No new content - might be refinement
                            if text != self.last_sent_interim_text:
                                # Text changed - update tracking but don't send (avoid duplicates)
                                self.last_sent_interim_text = text
                                self.current_interim_text = text
                            logger.debug(f"[Whisper STT #{self.transcriptions_received} INTERIM] No new content, skipping")
                    
                    # Always update last_full_text for next comparison
                    self.last_full_text = text
            
            # Update last transcription time
            with self.buffer_lock:
                self.last_transcription_time = time.time()
                
        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}", exc_info=True)
            with self.buffer_lock:
                self.audio_buffer = []
