"""Audio processing pipeline with VAD, RNNoise, and OpenAI Realtime STT"""
import numpy as np
import logging
from typing import Optional, Callable, Dict, Any
import time

from .vad import VAD
from .rnnoise import RNNoise
from .stt_openai_realtime import OpenAIRealtimeSTT
from ..config import AudioConfig

logger = logging.getLogger(__name__)

class AudioPipeline:
    """Real-time audio processing pipeline with VAD, RNNoise, and OpenAI Realtime STT"""
    
    def __init__(self, config: AudioConfig, event_callback: Optional[Callable] = None):
        self.config = config
        self.event_callback = event_callback
        
        # Initialize VAD
        self.vad = VAD(
            aggressiveness=config.VAD_AGGRESSIVENESS,
            frame_ms=config.VAD_FRAME_MS,
            sample_rate=config.SAMPLE_RATE
        )
        
        # Initialize RNNoise for noise suppression
        self.rnnoise = RNNoise(sample_rate=config.SAMPLE_RATE)
        
        # Initialize OpenAI Realtime STT with callback
        # Note: OpenAI Realtime API expects 24000 Hz, but we'll handle resampling
        self.streaming_stt = OpenAIRealtimeSTT(
            model="gpt-4o-realtime-preview-2024-10-01",
            sample_rate=24000,  # OpenAI Realtime API sample rate
            on_transcript=self._on_transcript
        )
        
        # Check if STT is available (has API key)
        self.stt_available = self.streaming_stt.api_key is not None
        if not self.stt_available:
            logger.warning("=" * 60)
            logger.warning("⚠️ OpenAI Realtime STT is not available (API key missing).")
            logger.warning("   Transcription will be disabled until API key is configured.")
            logger.warning("   Set OPENAI_API_KEY environment variable or add to .env file")
            logger.warning("=" * 60)
        else:
            logger.info("✅ OpenAI Realtime STT is available (API key configured)")
        
        # State tracking
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.stt_start_failed = False  # Track if we've already failed to start STT
        
        # Speech detection parameters
        self.min_speech_chunks = max(1, int(config.MIN_SPEECH_DURATION_MS / config.VAD_FRAME_MS))
        self.hangover_chunks = int(config.SPEECH_HANGOVER_MS / config.VAD_FRAME_MS)
        self.speech_chunk_count = 0
        self.silence_chunk_count = 0
        
        logger.info(f"AudioPipeline initialized with VAD, RNNoise, and OpenAI Realtime STT")
        logger.info(f"Speech detection: hangover={self.hangover_chunks} chunks ({config.SPEECH_HANGOVER_MS}ms), min_speech={self.min_speech_chunks} chunks ({config.MIN_SPEECH_DURATION_MS}ms)")
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event via callback"""
        if self.event_callback:
            try:
                self.event_callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    def _on_transcript(self, text: str, is_final: bool, language: str, confidence: float):
        """Callback from streaming STT when transcription is received"""
        try:
            if not text.strip():
                return
            
            logger.info(f"[OpenAIRealtime] {'[FINAL]' if is_final else '[INTERIM]'} '{text[:60]}...' (lang={language})")
            
            if is_final:
                # Final transcription
                self._emit_event("transcription", {
                    "text": text,
                    "language": language,
                    "confidence": confidence
                })
            else:
                # Interim transcription
                self._emit_event("transcription_interim", {
                    "text": text,
                    "language": language,
                    "confidence": confidence
                })
        except Exception as e:
            logger.error(f"Error in transcript callback: {e}", exc_info=True)
    
    def _resample_to_24k(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio from current sample rate to 24000 Hz for OpenAI"""
        if self.config.SAMPLE_RATE == 24000:
            return audio
        
        ratio = 24000 / self.config.SAMPLE_RATE
        new_length = int(len(audio) * ratio)
        
        try:
            from scipy import signal
            return signal.resample(audio, new_length).astype(np.float32)
        except ImportError:
            # Simple linear interpolation fallback
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    
    def process_chunk(self, audio: np.ndarray, reference_audio: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Process a single audio chunk"""
        if len(audio) == 0:
            return None
        
        # Apply RNNoise noise suppression
        denoised_audio = self.rnnoise.reduce_noise(audio)
        
        # VAD detection on denoised audio
        is_speech = self.vad.is_speech(denoised_audio)
        
        # Update state
        if is_speech:
            self.speech_chunk_count += 1
            self.silence_chunk_count = 0
            
            # Speech start detection - only trigger if we're not already speaking
            if not self.is_speaking:
                # Require minimum consecutive speech chunks to avoid false positives
                if self.speech_chunk_count >= self.min_speech_chunks:
                    # Check if STT is available before attempting to start
                    if not self.stt_available:
                        # STT not available (no API key), skip transcription
                        if not self.stt_start_failed:
                            logger.warning("⚠️ Skipping transcription - OpenAI API key not configured")
                            logger.warning(f"   Speech detected (chunks={self.speech_chunk_count}), but STT unavailable")
                            self.stt_start_failed = True  # Only log once
                        return denoised_audio
                    
                    # Log speech start attempt
                    logger.info(f"🔔 Speech detected ({self.speech_chunk_count} consecutive chunks) - starting OpenAI Realtime transcription")
                    logger.info(f"   STT available: {self.stt_available}, API key present: {self.streaming_stt.api_key is not None}")
                    
                    self.is_speaking = True
                    self.speech_start_time = time.time()
                    self.last_speech_time = time.time()
                    
                    # Start streaming STT
                    if self.streaming_stt.start_stream():
                        logger.info("✅ OpenAI Realtime STT stream started successfully")
                        self._emit_event("speech_start", {"timestamp": self.speech_start_time})
                        self.stt_start_failed = False  # Reset failure flag on success
                    else:
                        logger.error("❌ Failed to start OpenAI Realtime STT stream")
                        if not self.stt_start_failed:
                            logger.error("   Will not retry for this session")
                            self.stt_start_failed = True  # Only log once
                        self.is_speaking = False
                        return denoised_audio
                # else: still accumulating speech chunks, not enough yet
            
            # Send denoised audio to streaming STT if we're speaking
            if self.is_speaking:
                # Resample to 24kHz for OpenAI Realtime API
                audio_24k = self._resample_to_24k(denoised_audio)
                self.streaming_stt.send_audio(audio_24k)
                self.last_speech_time = time.time()
        else:
            # No speech detected
            self.silence_chunk_count += 1
            # Only reset speech_chunk_count if we're not speaking (to allow re-triggering)
            if not self.is_speaking:
                self.speech_chunk_count = 0
            
            # Speech end detection (with hangover)
            if self.is_speaking:
                if self.silence_chunk_count >= self.hangover_chunks:
                    # Speech ended
                    self.is_speaking = False
                    speech_end_time = time.time()
                    duration = speech_end_time - self.speech_start_time if self.speech_start_time else 0
                    
                    logger.info(f"🔔 Speech ended - stopping OpenAI Realtime transcription, duration={duration:.2f}s")
                    
                    # Stop streaming STT
                    self.streaming_stt.stop_stream()
                    
                    self._emit_event("speech_end", {
                        "timestamp": speech_end_time,
                        "duration": duration
                    })
                    
                    # Reset counters for next speech segment
                    self.speech_chunk_count = 0
                    self.silence_chunk_count = 0
                else:
                    # Still in hangover period, keep sending audio
                    audio_24k = self._resample_to_24k(denoised_audio)
                    self.streaming_stt.send_audio(audio_24k)
                    self.last_speech_time = time.time()
        
        # Return denoised audio
        return denoised_audio
    
    def reset(self):
        """Reset pipeline state (useful when starting a new recording)"""
        # Stop streaming if active
        if self.is_speaking:
            self.streaming_stt.stop_stream()
        
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.speech_chunk_count = 0
        self.silence_chunk_count = 0
        self.stt_start_failed = False  # Reset failure flag on pipeline reset
        
        logger.debug("Pipeline state reset")
