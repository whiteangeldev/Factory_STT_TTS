"""Audio processing pipeline with VAD, RNNoise, and offline Whisper STT"""
import numpy as np
import logging
from typing import Optional, Callable, Dict, Any
import time

from .vad import VAD
from .rnnoise import RNNoise
from .stt_whisper_offline import WhisperOfflineSTT
from ..config import AudioConfig

logger = logging.getLogger(__name__)

class AudioPipeline:
    def __init__(self, config: AudioConfig, event_callback: Optional[Callable] = None):
        self.config = config
        self.event_callback = event_callback
        
        self.vad = VAD(config.VAD_AGGRESSIVENESS, config.VAD_FRAME_MS, config.SAMPLE_RATE)
        self.rnnoise = RNNoise(sample_rate=config.SAMPLE_RATE)
        self.streaming_stt = WhisperOfflineSTT(
            model="small",  # Whisper model: "tiny", "base", "small", "medium", "large" (small for better accuracy)
            sample_rate=16000,  # Whisper uses 16kHz
            on_transcript=self._on_transcript
        )
        
        self.stt_available = self.streaming_stt.is_available
        if not self.stt_available:
            logger.warning("Whisper STT not available (install with: pip install openai-whisper)")
        else:
            logger.info("Whisper offline STT available")
        
        self.is_speaking = False
        self.speech_start_time = None
        self.stt_start_failed = False
        self.stt_stop_time = None
        self.stt_cooldown_seconds = 0.5
        
        self.min_speech_chunks = max(1, int(config.MIN_SPEECH_DURATION_MS / config.VAD_FRAME_MS))
        self.hangover_chunks = int(config.SPEECH_HANGOVER_MS / config.VAD_FRAME_MS)
        self.speech_chunk_count = 0
        self.silence_chunk_count = 0
        self.speech_pre_buffer = []  # Buffer audio chunks before STT starts to capture beginning
        self._stopping_stt = False
        self._stopping_chunks = 0
        self._always_buffer = True  # Always buffer audio from recording start, not just when VAD detects speech
        self.last_chunk_is_speech = False  # Store per-chunk VAD result for frontend display
        # Track audio levels for noise detection
        self.recent_audio_levels = []  # Store recent audio levels to detect constant noise
        self.max_level_history = 20
        self._constant_noise_detected = False  # Persistent flag for constant noise detection  # Track last 20 chunks (~600ms at 30ms chunks)
        self.requires_silence_before_speech = False  # Disabled - was too strict and prevented real speech detection
        self.silence_chunks_required = 0  # Not used when requires_silence_before_speech is False
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        if self.event_callback:
            try:
                self.event_callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    def _on_transcript(self, text: str, is_final: bool, language: str, confidence: float):
        if not text or not text.strip():
            logger.debug(f"[Pipeline STT] Empty transcription received (is_final={is_final})")
            return
        
        # Log transcription
        status = "FINAL" if is_final else "INTERIM"
        logger.info(f"[Pipeline STT {status}] '{text[:80]}...'")
        
        # Emit to frontend with timestamp for latency tracking
        event_type = "transcription" if is_final else "transcription_interim"
        self._emit_event(event_type, {
            "text": text.strip(),
            "language": language or "en",
            "confidence": float(confidence) if confidence else (1.0 if is_final else 0.8),
            "timestamp": time.time(),
            "is_final": is_final
        })
    
    def _resample_to_stt_rate(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio to STT target rate (16kHz for Whisper)"""
        target_rate = 16000  # Whisper uses 16kHz
        
        if self.config.SAMPLE_RATE == target_rate:
            return audio
        
        ratio = target_rate / self.config.SAMPLE_RATE
        new_length = int(len(audio) * ratio)
        try:
            from scipy import signal
            return signal.resample(audio, new_length).astype(np.float32)
        except ImportError:
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    
    def process_chunk(self, audio: np.ndarray, reference_audio: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if len(audio) == 0:
            return None
        
        denoised_audio = self.rnnoise.reduce_noise(audio)
        
        # Check audio level - filter out very quiet audio to reduce false positives
        audio_level = np.abs(denoised_audio).max()
        # Use a moderate threshold - too low causes false positives, too high misses quiet speech
        MIN_AUDIO_LEVEL_FOR_VAD = 0.02  # Require at least 2% amplitude for VAD processing
        
        # Track recent audio levels to detect constant background noise (only for extreme cases)
        self.recent_audio_levels.append(audio_level)
        if len(self.recent_audio_levels) > self.max_level_history:
            self.recent_audio_levels.pop(0)
        
        # CRITICAL: Detect constant background noise at any level
        # Speech has variation even when loud, constant noise is steady
        # Key insight: Constant noise has very low variation (std < 0.02) regardless of level
        # Once detected, persist the rejection until we get real silence
        if len(self.recent_audio_levels) >= 10:
            # Calculate variation (standard deviation) of recent levels
            level_std = np.std(self.recent_audio_levels)
            level_mean = np.mean(self.recent_audio_levels)
            
            # Check if constant noise pattern is present
            is_constant_noise_now = False
            if 0.15 <= level_mean <= 0.60 and level_std < 0.02:
                is_constant_noise_now = True
            elif level_mean > 0.60 and level_std < 0.01:
                is_constant_noise_now = True
            
            # Set persistent flag if constant noise detected
            if is_constant_noise_now:
                if not self._constant_noise_detected:
                    logger.warning(f"🔇 Detected constant noise: mean={level_mean:.3f} ({level_mean*100:.1f}%), std={level_std:.3f} - rejecting as non-speech")
                self._constant_noise_detected = True
            # Clear flag if we detect variation (speech) or real silence
            elif self._constant_noise_detected:
                # Clear constant noise flag if:
                # 1. Real silence (very low levels)
                # 2. Significant variation detected (std > 0.03) - indicates speech, not constant noise
                # 3. Quiet varying audio (low mean with variation)
                if audio_level < 0.01 or level_std > 0.03 or (level_mean < 0.05 and level_std > 0.01):
                    logger.info(f"🔊 Constant noise cleared: mean={level_mean:.3f}, std={level_std:.3f} - variation detected (speech)")
                    self._constant_noise_detected = False
        
        is_likely_constant_noise = self._constant_noise_detected
        
        # CRITICAL: Always buffer audio from recording start, not just when VAD detects speech
        # This ensures we capture the first words even if VAD takes time to detect
        if self._always_buffer and not self.is_speaking:
            audio_16k = self._resample_to_stt_rate(denoised_audio)
            self.speech_pre_buffer.append(audio_16k)
            # Keep last ~3 seconds worth
            max_pre_buffer_samples = int(16000 * 3.0)
            total_samples = sum(len(chunk) for chunk in self.speech_pre_buffer)
            while total_samples > max_pre_buffer_samples and len(self.speech_pre_buffer) > 1:
                removed = self.speech_pre_buffer.pop(0)
                total_samples -= len(removed)
        
        # Only run VAD on audio that's loud enough (reduces false positives from background noise)
        is_speech = False
        
        # CRITICAL: If constant noise was detected, persist the rejection and reset speech count
        # This prevents constant noise from accumulating speech chunks and triggering STT
        if is_likely_constant_noise:
            is_speech = False
            # Reset speech chunk count to prevent false triggers from accumulated chunks
            self.speech_chunk_count = 0
        elif audio_level >= MIN_AUDIO_LEVEL_FOR_VAD:
            # Audio level is reasonable - run VAD (no upper limit, let VAD decide)
            is_speech = self.vad.is_speech(denoised_audio)
        else:
            # Very quiet - treat as silence
            is_speech = False
            # Reset constant noise flag if we get real silence (very low levels)
            if hasattr(self, '_noise_warning_logged') and audio_level < 0.01:
                # Real silence detected - allow noise detection to reset
                if len(self.recent_audio_levels) > 0:
                    recent_mean = np.mean(self.recent_audio_levels[-5:])  # Check last 5 chunks
                    if recent_mean < 0.01:
                        # Clear noise flag after sustained silence
                        if hasattr(self, '_noise_warning_logged'):
                            delattr(self, '_noise_warning_logged')
        
        # CRITICAL: Require silence before accepting new speech (reduces false positives)
        # This prevents constant noise from triggering speech detection
        if is_speech and not self.is_speaking and self.requires_silence_before_speech:
            # Check if we had enough silence chunks before this speech
            if self.silence_chunk_count < self.silence_chunks_required:
                # Not enough silence before speech - reject (likely constant noise)
                is_speech = False
        
        # Store per-chunk VAD result for frontend (needed for history-based smoothing)
        self.last_chunk_is_speech = is_speech
        
        if is_speech:
            self.speech_chunk_count += 1
            self.silence_chunk_count = 0
            
            if not self.is_speaking:
                # Audio is already being buffered above (always_buffer mode)
                # Just check if we have enough speech chunks to start STT
                if self.speech_chunk_count >= self.min_speech_chunks:
                    if self.stt_stop_time and (time.time() - self.stt_stop_time) < self.stt_cooldown_seconds:
                        return denoised_audio
                    
                    if not self.stt_available:
                        if not self.stt_start_failed:
                            logger.warning("Skipping transcription - API key not configured")
                            self.stt_start_failed = True
                        return denoised_audio
                    
                    logger.info(f"Speech detected ({self.speech_chunk_count} chunks) - starting STT")
                    self.is_speaking = True
                    self.speech_start_time = time.time()
                    self.stt_stop_time = None
                    
                    if self.streaming_stt.start_stream():
                        logger.info("STT stream started")
                        self._emit_event("speech_start", {"timestamp": self.speech_start_time})
                        self.stt_start_failed = False
                        self.stt_stop_time = None  # Clear cooldown timer on successful start
                        # REMOVED: No delay needed - STT is ready immediately
                        # Reset send counter
                        self._stt_send_count = 0
                        
                        # Send pre-buffered audio chunks first to capture the beginning
                        # This includes audio from BEFORE VAD detected speech
                        if len(self.speech_pre_buffer) > 0:
                            pre_buffer_duration = sum(len(chunk) for chunk in self.speech_pre_buffer) / 16000
                            logger.info(f"Sending {len(self.speech_pre_buffer)} pre-buffered audio chunks ({pre_buffer_duration:.2f}s) to capture speech beginning")
                            # Send all pre-buffered chunks immediately (no delay)
                            for pre_audio in self.speech_pre_buffer:
                                self.streaming_stt.send_audio(pre_audio)
                            self.speech_pre_buffer = []  # Clear after sending
                            # REMOVED: No delay needed - chunks are queued and processed asynchronously
                        
                        # CRITICAL: Also send the current chunk that triggered speech detection
                        # This ensures we don't lose the chunk that contains the beginning of speech
                        current_audio_16k = self._resample_to_stt_rate(denoised_audio)
                        self.streaming_stt.send_audio(current_audio_16k)
                        logger.debug(f"Sent current chunk after pre-buffer (len={len(current_audio_16k)})")
                        # Stop always-buffering now that STT has started
                        self._always_buffer = False
                    else:
                        logger.error("Failed to start STT stream")
                        if not self.stt_start_failed:
                            self.stt_start_failed = True
                        self.is_speaking = False
                        self.speech_pre_buffer = []  # Clear on failure
                        return denoised_audio
            else:
                # Already speaking - send audio normally
                # Resample to 16kHz for Whisper STT
                audio_16k = self._resample_to_stt_rate(denoised_audio)
                # Send audio to STT (it will buffer and process internally)
                # Log first few sends to debug
                if not hasattr(self, '_stt_send_count'):
                    self._stt_send_count = 0
                self._stt_send_count += 1
                if self._stt_send_count <= 5:
                    logger.debug(f"[Pipeline] Sending audio chunk {self._stt_send_count} to STT (len={len(audio_16k)}, max={np.abs(audio_16k).max():.6f})")
                self.streaming_stt.send_audio(audio_16k)
        else:
            self.silence_chunk_count += 1
            if not self.is_speaking:
                self.speech_chunk_count = 0
                # Don't clear pre-buffer immediately - keep it for a bit in case speech starts soon
                # Only clear if we have too much silence (more than 1 second)
                if len(self.speech_pre_buffer) > 0:
                    total_pre_buffer_duration = sum(len(chunk) for chunk in self.speech_pre_buffer) / 16000
                    if total_pre_buffer_duration > 1.0:  # More than 1 second of buffered audio
                        # Keep only the last 0.5 seconds
                        max_keep_samples = int(16000 * 0.5)
                        total_samples = sum(len(chunk) for chunk in self.speech_pre_buffer)
                        while total_samples > max_keep_samples and len(self.speech_pre_buffer) > 1:
                            removed = self.speech_pre_buffer.pop(0)
                            total_samples -= len(removed)
            
            if self.is_speaking:
                if self.silence_chunk_count >= self.hangover_chunks:
                    # Send any remaining audio before stopping (important for capturing end of speech)
                    audio_16k = self._resample_to_stt_rate(denoised_audio)
                    self.streaming_stt.send_audio(audio_16k)
                    
                    # Mark that we're about to stop, but continue sending a few more chunks
                    # to ensure we capture the very end of speech
                    if not self._stopping_stt:
                        self._stopping_stt = True
                        self._stopping_chunks = 0
                        # CRITICAL: Set is_speaking to False immediately when silence is detected
                        # This ensures frontend shows "No Speech" right away, even though we continue
                        # processing trailing audio for a bit longer to capture the end of speech
                        self.is_speaking = False
                        logger.info("Speech ended - silence detected, continuing to process trailing audio")
                        # Emit speech_end event immediately so frontend updates status
                        duration = time.time() - self.speech_start_time if self.speech_start_time else 0
                        self._emit_event("speech_end", {"timestamp": time.time(), "duration": duration})
                    
                    self._stopping_chunks += 1
                    
                    # Send a few more chunks after hangover to capture trailing speech
                    if self._stopping_chunks <= 10:  # Send 10 more chunks after hangover (increased from 8)
                        # Continue sending during this extended period - already sent above
                        pass
                    else:
                        # Now actually stop STT processing
                        # Longer delay to ensure last audio chunks are queued and processed
                        time.sleep(0.7)  # Increased from 0.5 to ensure all chunks are sent
                        
                        duration = time.time() - self.speech_start_time if self.speech_start_time else 0
                        logger.info(f"Stopping STT processing, duration={duration:.2f}s")
                        # stop_stream will handle waiting for final transcription
                        self.streaming_stt.stop_stream()
                        self.stt_stop_time = time.time()
                        self.speech_chunk_count = 0
                        self.silence_chunk_count = 0
                        self._stopping_stt = False
                        self._stopping_chunks = 0
                else:
                    # Continue sending audio during hangover period
                    audio_16k = self._resample_to_stt_rate(denoised_audio)
                    self.streaming_stt.send_audio(audio_16k)
        
        return denoised_audio
    
    def reset(self):
        if self.is_speaking:
            self.streaming_stt.stop_stream()
        self.is_speaking = False
        self.speech_start_time = None
        self.speech_chunk_count = 0
        self.silence_chunk_count = 0
        self.speech_pre_buffer = []
        self.stt_start_failed = False
        self.stt_stop_time = None
        self._stopping_stt = False
        self._stopping_chunks = 0
        self._always_buffer = True  # Re-enable always-buffering for next recording
        self.last_chunk_is_speech = False  # Reset per-chunk VAD result
        self.recent_audio_levels = []  # Reset audio level history
        self._constant_noise_detected = False  # Reset constant noise detection flag