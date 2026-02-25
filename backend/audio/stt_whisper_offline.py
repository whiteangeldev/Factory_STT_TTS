"""Offline Speech-to-Text using OpenAI Whisper"""
import numpy as np
import logging
import threading
import time
import re
import subprocess
import os
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
    # Shared cache so multiple pipeline instances reuse one loaded model.
    _MODEL_CACHE = {}
    _MODEL_LOADING_EVENTS = {}
    _MODEL_CACHE_LOCK = threading.Lock()

    def __init__(self, model="small", sample_rate=16000, on_transcript: Optional[Callable] = None):
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
        # Compact log mode by default; enable detailed trace only when explicitly requested.
        self.trace_enabled = os.getenv("FACTORY_STT_TRACE", "0").lower() not in {"0", "false", "off", "no"}
        
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
        self.full_session_buffer = []  # Never trimmed; used for final transcription to preserve full context
        self.max_full_session_duration = 600.0  # Cap at 10 min to bound memory
        self.buffer_lock = threading.Lock()
        self.processing_thread = None
        self.min_buffer_duration = 2.0  # Minimum 2 seconds before first interim
        self.interim_interval = 1.0  # Process interim results every 1.0s for faster UI updates
        # Keep memory bounded for long-running streams and process interim on a recent window only.
        self.max_buffer_duration = 45.0
        self.interim_window_duration = 10.0
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
        self.pending_sentence_candidate = ""  # Candidate sentence waiting for stability confirmation
        self.pending_sentence_hits = 0
        self.pending_sentence_text = ""
        self.detected_language_interim = None  # Cache detected language for interim
        self.compute_device = "cpu"
        self.use_fp16 = False
        
        # Compatibility property for pipeline
        self.api_key = "offline"  # Non-None value to indicate availability
        
        # Select compute backend once and reuse it across all transcribe calls.
        self._configure_inference_device()
        
        # Load model in background thread to avoid blocking
        self._load_model()

    def _trace(self, message: str):
        """Detailed STT trace logs for debugging transcription workflow."""
        if self.trace_enabled:
            logger.debug(f"[STT TRACE] {message}")

    def _trim_audio_buffer_locked(self):
        """Trim old chunks to keep audio buffer within max duration."""
        if self.max_buffer_duration <= 0:
            return
        max_samples = int(self.sample_rate * self.max_buffer_duration)
        total_samples = sum(len(chunk) for chunk in self.audio_buffer)
        while total_samples > max_samples and len(self.audio_buffer) > 1:
            removed = self.audio_buffer.pop(0)
            total_samples -= len(removed)

        # If old chunks were dropped, keep counters in a valid range.
        # Interims use full_session_buffer, so clamp to that length.
        if self.last_processed_chunk_count > len(self.full_session_buffer):
            self.last_processed_chunk_count = len(self.full_session_buffer)

    def _get_recent_buffer_locked(self, window_seconds: float) -> list[np.ndarray]:
        """Return the most recent audio chunks up to the requested duration."""
        target_samples = int(max(0.1, window_seconds) * self.sample_rate)
        collected = 0
        selected = []
        for chunk in reversed(self.audio_buffer):
            selected.append(chunk)
            collected += len(chunk)
            if collected >= target_samples:
                break
        selected.reverse()
        return selected

    def _nvidia_gpu_present(self) -> bool:
        """Best-effort check for an NVIDIA GPU on Windows/Linux."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            return result.returncode == 0 and "GPU" in (result.stdout or "")
        except Exception:
            return False

    def _configure_inference_device(self):
        """Choose the fastest available inference device."""
        try:
            import torch
        except Exception:
            self.compute_device = "cpu"
            self.use_fp16 = False
            logger.warning("PyTorch unavailable, Whisper STT will run on CPU.")
            return

        if torch.cuda.is_available():
            self.compute_device = "cuda"
            self.use_fp16 = True
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            logger.info("Whisper STT configured for NVIDIA GPU (CUDA + fp16).")
            return

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.compute_device = "mps"
            self.use_fp16 = False
            logger.info("Whisper STT configured for Apple Metal (MPS).")
            return

        self.compute_device = "cpu"
        self.use_fp16 = False
        if self._nvidia_gpu_present():
            logger.warning(
                "NVIDIA GPU detected, but PyTorch CUDA is unavailable. "
                "Install a CUDA-enabled torch build to avoid CPU-only STT."
            )
        logger.info("Whisper STT configured for CPU.")
    
    def _load_model(self):
        """Load Whisper model in background with cross-instance cache."""
        model_key = (self.model_name, self.compute_device)

        with self.__class__._MODEL_CACHE_LOCK:
            cached = self.__class__._MODEL_CACHE.get(model_key)
            if cached is not None:
                self.model = cached
                logger.info(f"Whisper model {self.model_name} reused from cache on {self.compute_device}")
                return

            loading_event = self.__class__._MODEL_LOADING_EVENTS.get(model_key)
            should_start_loader = loading_event is None
            if should_start_loader:
                loading_event = threading.Event()
                self.__class__._MODEL_LOADING_EVENTS[model_key] = loading_event

        def load_or_wait():
            if should_start_loader:
                loaded = None
                try:
                    logger.info(
                        f"Loading Whisper model: {self.model_name} on {self.compute_device} (this may take a moment...)"
                    )
                    loaded = whisper.load_model(self.model_name, device=self.compute_device)
                    logger.info(f"Whisper model {self.model_name} loaded successfully on {self.compute_device}")
                except Exception as e:
                    logger.error(f"Failed to load Whisper model: {e}")
                finally:
                    with self.__class__._MODEL_CACHE_LOCK:
                        if loaded is not None:
                            self.__class__._MODEL_CACHE[model_key] = loaded
                        self.__class__._MODEL_LOADING_EVENTS.pop(model_key, None)
                        loading_event.set()
            else:
                loading_event.wait(timeout=60.0)

            with self.__class__._MODEL_CACHE_LOCK:
                self.model = self.__class__._MODEL_CACHE.get(model_key)
            if self.model is None:
                self.is_available = False

        thread = threading.Thread(target=load_or_wait, daemon=True)
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
            time.sleep(0.2)
            elapsed += 0.5
        
        if self.model is None:
            logger.error("Whisper model failed to load")
            return False
        
        with self.buffer_lock:
            if self.is_streaming:
                return True
            
            self.is_streaming = True
            self.audio_buffer = []
            self.full_session_buffer = []
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
            self.pending_sentence_candidate = ""
            self.pending_sentence_hits = 0
            self.pending_sentence_text = ""
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_audio_worker, daemon=True)
            self.processing_thread.start()
            
            logger.info("Whisper STT stream started")
            return True
    
    def stop_stream(self):
        """Stop STT stream and process remaining audio. Emits interim-as-final ASAP for fast UI."""
        # Hold lock only long enough to read refs and interim text (no heavy copy here).
        with self.buffer_lock:
            if not self.is_streaming:
                return
            latest_interim = (self.current_interim_text or self.last_sent_interim_text or "").strip()
            latest_lang = (self.detected_language_interim or "en")
            refs = list(self.full_session_buffer) if self.full_session_buffer else []
        time.sleep(0.05)  # Minimal wait for in-flight chunks (system audio already stopped on user stop)

        with self.buffer_lock:
            self.is_streaming = False
            # Emit interim as final immediately so UI shows "done" without waiting for full-session run
            if latest_interim and len(latest_interim) > 2:
                try:
                    self.on_transcript(latest_interim, True, latest_lang, 0.98)
                    self.last_final_text = latest_interim
                except Exception as e0:
                    logger.error(f"Error sending immediate final (interim): {e0}", exc_info=True)
            if refs:
                full_duration = sum(len(c) for c in refs) / self.sample_rate
                if full_duration >= 1.0:
                    logger.info(f"Finalizing from full session buffer ({full_duration:.2f}s) in background to preserve entire transcript")
                    def run_final_in_background():
                        try:
                            copy_buf = [np.array(c, copy=True) for c in refs]
                            self._process_audio_buffer(copy_buf, final=True, language_hint=latest_lang)
                        except Exception as e:
                            logger.error(f"Error in full-session final transcription: {e}", exc_info=True)
                    threading.Thread(target=run_final_in_background, daemon=True).start()
            self.full_session_buffer = []
            self.audio_buffer = []

        # Short wait for worker to see is_streaming=False and exit; final already emitted above.
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=0.5)
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
            self._trim_audio_buffer_locked()
            # Keep full session for final transcription (cap duration to bound memory)
            self.full_session_buffer.append(np.array(audio, dtype=np.float32, copy=True))
            max_samples = int(self.sample_rate * self.max_full_session_duration)
            total_full = sum(len(c) for c in self.full_session_buffer)
            while total_full > max_samples and len(self.full_session_buffer) > 1:
                removed = self.full_session_buffer.pop(0)
                total_full -= len(removed)
            
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
                        time.sleep(0.2)
                        continue
                    buffer_samples = sum(len(chunk) for chunk in self.audio_buffer)
                    buffer_duration = buffer_samples / self.sample_rate
                    current_time = time.time()
                    should_process_interim = (
                        buffer_duration >= self.min_buffer_duration and
                        (self.last_interim_time is None or
                         current_time - self.last_interim_time >= self.interim_interval)
                    )
                    refs = []
                    current_chunk_count = 0
                    full_duration = 0.0
                    has_new_audio = False
                    should_refine = False
                    has_enough_session = False
                    if should_process_interim and not self.is_processing_interim:
                        full_samples = sum(len(c) for c in self.full_session_buffer)
                        full_duration = full_samples / self.sample_rate
                        current_chunk_count = len(self.full_session_buffer)
                        time_since_last = current_time - self.last_interim_time if self.last_interim_time else float('inf')
                        has_new_audio = current_chunk_count > self.last_processed_chunk_count
                        should_refine = time_since_last >= self.interim_interval and current_chunk_count > 0
                        has_enough_session = full_duration >= self.min_buffer_duration
                        if (has_new_audio or should_refine) and has_enough_session:
                            refs = list(self.full_session_buffer)
                            self.last_interim_time = current_time
                            self.is_processing_interim = True
                # Copy outside lock so stop_stream() can acquire lock quickly for fast final.
                if refs:
                    processing_buffer = [np.array(c, copy=True) for c in refs]
                    self._trace(
                        f"worker trigger: full_session chunks={current_chunk_count}, duration={full_duration:.2f}s, "
                        f"has_new_audio={has_new_audio}, should_refine={should_refine}, "
                        f"last_processed_chunks={self.last_processed_chunk_count}"
                    )

                    def process_interim():
                        try:
                            self._process_audio_buffer(processing_buffer, final=False)
                            with self.buffer_lock:
                                if has_new_audio:
                                    self.last_processed_chunk_count = current_chunk_count
                        except Exception as e:
                            logger.error(f"Error processing interim transcription: {e}", exc_info=True)
                        finally:
                            with self.buffer_lock:
                                self.is_processing_interim = False

                    threading.Thread(target=process_interim, daemon=True).start()

                time.sleep(0.2)  # Check frequently for faster interim updates
            except Exception as e:
                logger.error(f"Error in audio worker: {e}")
                time.sleep(1.0)
    
    def _detect_sentence_end(self, old_text: str, new_text: str) -> bool:
        """
        Detect if a sentence has ended in the new text.
        Works with rolling-window hypotheses where new_text may not strictly grow.
        """
        return bool(self._extract_completed_sentence(old_text, new_text))

    def _is_reliable_sentence_chunk(self, sentence: str) -> bool:
        """Reject tiny/fragment clauses that should not be finalized."""
        if not sentence:
            return False
        candidate = sentence.strip()
        if not candidate:
            return False
        if not any(c.isalnum() for c in candidate):
            return False

        # CJK text may not contain spaces; use character-length threshold.
        if " " not in candidate:
            return len(candidate) >= 8

        words = candidate.split()
        if len(words) < 5 or len(candidate) < 24:
            return False

        first_word = re.sub(r"[^a-z]", "", words[0].lower())
        fragment_starters = {
            "and", "or", "but", "so", "for", "to", "of", "in",
            "on", "at", "with", "by", "from", "as"
        }
        # Most bad splits in logs are short tail clauses beginning with conjunctions/prepositions.
        if first_word in fragment_starters and len(words) < 7:
            return False

        return True

    def _requires_sentence_confirmation(self, sentence: str) -> bool:
        """
        Determine whether a candidate sentence is likely unstable and should
        be seen twice before finalizing.
        """
        if not sentence:
            return True
        candidate = sentence.strip()
        words = candidate.split()
        if not words:
            return True

        first_word_raw = words[0]
        first_word = re.sub(r"[^a-z]", "", first_word_raw.lower())
        fragment_starters = {
            "and", "or", "but", "so", "for", "to", "of", "in",
            "on", "at", "with", "by", "from", "as", "including",
            "processing", "using"
        }

        # If first alphabetic char is lowercase, it is often a continuation fragment.
        first_alpha = next((ch for ch in first_word_raw if ch.isalpha()), "")
        starts_lowercase = bool(first_alpha) and first_alpha.islower()
        is_short = len(words) < 10
        starts_fragment = first_word in fragment_starters

        return starts_lowercase or starts_fragment or is_short

    def _extract_sentence_until_punctuation(self, text: str) -> str:
        """
        Return the first complete sentence in text, if present.
        """
        if not text:
            return ""

        sentence_pattern = r"[^.!?]+[.!?]+"
        matches = list(re.finditer(sentence_pattern, text.strip()))
        if not matches:
            return ""

        for match in matches:
            candidate = match.group().strip()
            if self._is_reliable_sentence_chunk(candidate):
                return candidate
        return ""
    
    def _extract_completed_sentence(self, old_text: str, new_text: str) -> str:
        """
        Extract the completed sentence (up to the first sentence ending that's new).
        """
        if not new_text:
            return ""

        old_normalized = (old_text or "").strip()
        new_normalized = new_text.strip()
        sentence_pattern = r"[^.!?]+[.!?]+"

        old_count = len(re.findall(sentence_pattern, old_normalized))
        new_matches = list(re.finditer(sentence_pattern, new_normalized))
        if len(new_matches) <= old_count:
            return ""

        # Only consider newly appeared complete chunks.
        for idx, match in enumerate(new_matches):
            if idx < old_count:
                continue
            candidate = match.group().strip()
            if self._is_reliable_sentence_chunk(candidate):
                return candidate
        return ""
    
    def _extract_remaining_text_after_sentence(self, text: str, completed_sentence: str = "") -> str:
        """
        Extract text that comes after the first completed sentence.
        """
        if not text:
            return ""

        if completed_sentence:
            start_idx = text.find(completed_sentence)
            if start_idx >= 0:
                end_idx = start_idx + len(completed_sentence)
                remaining = text[end_idx:].strip()
                remaining = re.sub(r'^[\s,;:]+', '', remaining)
                return remaining

        # Find first sentence ending at a boundary.
        match = re.search(r"[.!?]+(?=\s|$)", text)
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

    def _collapse_adjacent_duplicate_phrases(self, text: str, max_phrase_words: int = 12) -> str:
        """Collapse immediate repeated phrase loops in unstable hypotheses."""
        words = text.split()
        if len(words) < 8:
            return text.strip()

        def _norm(seq):
            return [re.sub(r"[^\w]", "", w.lower()) for w in seq]

        out = []
        i = 0
        n_words = len(words)
        while i < n_words:
            duplicated = False
            max_n = min(max_phrase_words, (n_words - i) // 2)
            for n in range(max_n, 2, -1):
                a = _norm(words[i:i + n])
                b = _norm(words[i + n:i + (2 * n)])
                if a and a == b:
                    out.extend(words[i:i + n])
                    i += 2 * n
                    while i + n <= n_words and _norm(words[i:i + n]) == a:
                        i += n
                    duplicated = True
                    break
            if not duplicated:
                out.append(words[i])
                i += 1
        return " ".join(out).strip()

    def _sanitize_transcript_text(self, text: str) -> str:
        """Normalize whitespace and suppress common repetition artifacts."""
        if not text:
            return ""
        cleaned = re.sub(r"\s+", " ", text).strip()
        cleaned = self._collapse_adjacent_duplicate_phrases(cleaned)

        # Collapse adjacent duplicate sentence-level chunks.
        sentence_parts = re.split(r"(?<=[.!?])\s+", cleaned)
        deduped = []
        prev_norm = ""
        for part in sentence_parts:
            p = part.strip()
            if not p:
                continue
            norm = re.sub(r"[^\w\s]", "", p.lower()).strip()
            if norm and norm == prev_norm:
                continue
            deduped.append(p)
            prev_norm = norm
        return " ".join(deduped).strip()

    def _merge_interim_monotonic(self, old_text: str, new_text: str) -> str:
        """
        Keep interim text monotonic so already-shown words do not disappear.
        This prioritizes display stability over aggressive hypothesis rewrites.
        """
        old_text = (old_text or "").strip()
        new_text = (new_text or "").strip()
        if not old_text:
            return new_text
        if not new_text:
            return old_text

        if new_text.startswith(old_text):
            return new_text
        if old_text.startswith(new_text):
            return old_text

        old_words = old_text.split()
        new_words = new_text.split()
        if not old_words or not new_words:
            return new_text or old_text

        def _norm(word: str) -> str:
            return re.sub(r"[^\w]", "", word.lower())

        max_overlap = min(len(old_words), len(new_words), 20)
        overlap = 0
        for k in range(max_overlap, 0, -1):
            tail = [_norm(w) for w in old_words[-k:]]
            head = [_norm(w) for w in new_words[:k]]
            if tail == head:
                overlap = k
                break

        if overlap > 0:
            merged_words = old_words + new_words[overlap:]
        else:
            merged_words = old_words + new_words
        return " ".join(merged_words).strip()

    def _strip_overlap_with_last_final(self, text: str) -> str:
        """
        Remove leading overlap between new interim text and the previous finalized sentence.
        This keeps each new section focused on new content.
        """
        if not text:
            return ""
        if not self.last_final_text:
            return text.strip()

        new_words = text.strip().split()
        last_words = self.last_final_text.strip().split()
        if not new_words or not last_words:
            return text.strip()

        norm_new = [re.sub(r"[^\w]", "", w.lower()) for w in new_words]
        norm_last = [re.sub(r"[^\w]", "", w.lower()) for w in last_words]
        norm_new = [w for w in norm_new if w]
        norm_last = [w for w in norm_last if w]
        if not norm_new or not norm_last:
            return text.strip()

        max_overlap = min(len(norm_new), len(norm_last), 12)
        min_overlap = 1 if len(norm_new) <= 3 else 2
        overlap = 0
        for k in range(max_overlap, min_overlap - 1, -1):
            if norm_last[-k:] == norm_new[:k]:
                overlap = k
                break

        if overlap <= 0:
            return text.strip()

        stripped_words = new_words[overlap:]
        return " ".join(stripped_words).strip()
    
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
    
    def _process_audio_buffer(self, audio_buffer_list, final=False, language_hint=None):
        """Process a specific audio buffer list and generate transcription.
        When final=True and language_hint is set (e.g. from stream), skip language detection
        to avoid numpy/torch dtype errors and use the known language for full-session transcript."""
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
            self._trace(
                f"transcribe start: final={final}, chunks={len(audio_buffer_list)}, "
                f"audio={len(audio_data)/16000:.2f}s"
            )
            
            # CRITICAL: Restrict to only English, Chinese, and Japanese
            supported_languages = ["en", "zh", "ja"]
            
            # For interim transcriptions: use faster settings with quick language detection
            # For final transcriptions: use full language detection and better settings
            if not final:
                # Quick language detection on first interim: skip noise head so pre-buffer doesn't lock to 'en'
                if self.detected_language_interim is None and len(audio_data) >= self.sample_rate * 2:
                    try:
                        n = len(audio_data)
                        if n >= self.sample_rate * 3:
                            start = int(1.0 * self.sample_rate)
                            end = start + int(2.0 * self.sample_rate)
                        else:
                            start = int(0.5 * self.sample_rate)
                            end = n
                        detection_audio = audio_data[start:end]
                        # Quick detection on speech-rich window (skip first 0.5–1s)
                        detection_result = self.model.transcribe(
                            detection_audio,
                            language=None,  # Auto-detect
                            task="transcribe",
                            fp16=self.use_fp16,
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
                        "fp16": self.use_fp16,
                        "verbose": False,
                        "condition_on_previous_text": False,
                        "beam_size": 2,
                        "best_of": 2,
                        "temperature": 0,
                        "compression_ratio_threshold": 2.4,
                        "logprob_threshold": -1.0,
                        "no_speech_threshold": 0.6,
                        "word_timestamps": False,
                    }
                    
                    result = self.model.transcribe(audio_data, **transcribe_kwargs)
                    language = interim_lang
                except Exception as e:
                    logger.error(f"Interim transcription failed: {e}")
                    return  # Skip this interim transcription
            else:
                # For final transcription: use language_hint if provided (full-session path), else detect language
                # CRITICAL: When finalizing from full_session_buffer we pass language_hint to skip detect_language,
                # avoiding numpy/torch dtype errors and ensuring the entire transcript (including beginning) is kept
                detected_language = language_hint if (language_hint and language_hint in ("en", "zh", "ja")) else None
                top_prob = 0.99 if detected_language else 0.0

                # Step 1: Fast language detection (skip if we have language_hint)
                if detected_language is None:
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
                                            fp16=self.use_fp16,
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
                        
                        except Exception:
                            raise  # Let outer except handle and set top_prob
                                    
                    except Exception as e:
                        logger.error(f"❌ Language detection failed: {e}", exc_info=True)
                        detected_language = None
                        top_prob = 0.0
                    
                    # CRITICAL: Check confidence - if low confidence, don't trust detection
                    if detected_language and detected_language in supported_languages:
                        if top_prob < 0.3:  # Very low confidence threshold (stricter)
                            logger.warning(f"⚠️ Low confidence detection ({top_prob:.3f}) for '{detected_language}' - will try all 3 languages for accuracy")
                            detected_language = None  # Force trying all 3 languages
                        elif top_prob < 0.5:  # Medium confidence - log warning but still use it
                            logger.info(f"⚠️ Medium confidence detection ({top_prob:.3f}) for '{detected_language}' - will verify with full transcription")
                
                # Step 2: If detected language is supported and confident, transcribe with it
                # CRITICAL: Only use detected language if confidence is high enough
                if detected_language and detected_language in supported_languages and top_prob >= 0.3:
                    try:
                        transcribe_kwargs = {
                            "language": detected_language,
                            "task": "transcribe",  # CRITICAL: transcribe, not translate
                            "fp16": self.use_fp16,
                            "verbose": False,
                            "condition_on_previous_text": False,
                            "beam_size": 2,
                            "best_of": 2,
                            "temperature": 0,
                            "compression_ratio_threshold": 2.4,
                            "logprob_threshold": -1.0,
                            "no_speech_threshold": 0.6,
                            "word_timestamps": False,
                        }
                        
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
                                "fp16": self.use_fp16,
                                "verbose": False,
                                "condition_on_previous_text": False,
                                "beam_size": 2,
                                "best_of": 2,
                                "temperature": 0,
                                "compression_ratio_threshold": 2.4,
                                "logprob_threshold": -1.0,
                                "no_speech_threshold": 0.6,
                                "word_timestamps": False,
                            }
                            
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
                            "fp16": self.use_fp16,
                            "verbose": False,
                            "condition_on_previous_text": False,
                            "beam_size": 2,
                            "best_of": 2,
                            "temperature": 0,
                        }
                        result = self.model.transcribe(audio_data, **transcribe_kwargs)
            
            text = result.get("text", "").strip()
            segments = result.get("segments", [])
            text = self._sanitize_transcript_text(text)
            self._trace(
                f"transcribe result: final={final}, lang={language}, text_len={len(text)}, "
                f"text='{text[:140]}'"
            )
            
            # Final verification: ensure language is one of our 3
            if language not in supported_languages:
                language = "en"
            
            # Filter out very short or low-confidence transcriptions
            if len(text) < 2:  # Too short, likely noise
                logger.debug(f"Skipping transcription - text too short: '{text}'")
                self._trace("skip: text too short")
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
                        # For interim, combine logprob with no_speech probability.
                        # no_speech alone can still look "confident" for hallucinated text.
                        avg_logprob = np.mean([s.get("avg_logprob", -1.0) for s in segments])
                        avg_no_speech_prob = np.mean([s.get("no_speech_prob", 0.0) for s in segments])
                        logprob_conf = max(0.0, min(1.0, 1.0 + avg_logprob))
                        nospeech_conf = max(0.0, 1.0 - avg_no_speech_prob)
                        confidence = min(logprob_conf, nospeech_conf)
                    
                    # Filter low-confidence interim results
                    if not final and confidence < 0.18:
                        logger.debug(f"Skipping low-confidence interim transcription: {confidence:.2f}")
                        self._trace(f"skip interim: low confidence={confidence:.3f}")
                        return
                
                # Language is already normalized to en/zh/ja above
                detected_lang = language
                
                # Track latency
                if self.first_transcription_time is None and self.first_audio_time:
                    latency = time.time() - self.first_audio_time
                    logger.debug(f"[Whisper Latency] First transcription: {latency:.3f}s")
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
                    raw_text = self._strip_overlap_with_last_final(text)
                    if not raw_text:
                        return
                    text = raw_text
                    if not text:
                        return
                    
                    # If this is the first transcription, send full text immediately
                    if not self.last_sent_interim_text:
                        # First update - send full text to preserve beginning
                        # CRITICAL: Always send full text on first update, even if it seems incomplete
                        # Whisper might refine it later, but we need to show what we have
                        logger.debug(f"[Whisper STT #{self.transcriptions_received} INTERIM] First update: '{text[:80]}...' (len={len(text)})")
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
                    
                    # Single-section workflow: never finalize by sentence in streaming mode.
                    # Keep one continuously updated interim transcript and finalize only on stop_stream().
                    if text != self.last_sent_interim_text:
                        incremental_text = self._get_incremental_text(self.last_sent_interim_text, text)
                        self._trace(
                            f"emit interim update: old_len={len(self.last_sent_interim_text)}, "
                            f"new_len={len(text)}, inc_len={len(incremental_text) if incremental_text else 0}"
                        )
                        try:
                            self.on_transcript(text, False, detected_lang, confidence, incremental_update=incremental_text or text)
                        except TypeError:
                            self.on_transcript(text, False, detected_lang, confidence)
                        except Exception as e:
                            logger.error(f"Error sending interim transcription: {e}", exc_info=True)
                        self.last_sent_interim_text = text
                        self.current_interim_text = text
                    
                    # Always update last_full_text for next comparison
                    self.last_full_text = text
            
            # Update last transcription time
            with self.buffer_lock:
                self.last_transcription_time = time.time()
                
        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}", exc_info=True)
            with self.buffer_lock:
                self.audio_buffer = []
