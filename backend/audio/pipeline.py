"""Main audio preprocessing pipeline"""
import numpy as np
from typing import Optional, Callable, List
import logging
from enum import Enum

from backend.audio.vad import VoiceActivityDetector
from backend.audio.noise_suppression import NoiseSuppressor
from backend.config import AudioConfig

logger = logging.getLogger(__name__)

class SpeechState(Enum):
    """Speech detection state machine"""
    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEECH = "speech"
    SPEECH_END = "speech_end"

class AudioPipeline:
    """Complete audio preprocessing pipeline with speech event detection"""
    
    def __init__(self, config: Optional[AudioConfig] = None, event_callback: Optional[Callable] = None):
        self.config = config or AudioConfig()
        self.event_callback = event_callback  # Callback for speech events
        
        # Initialize components
        self.vad = VoiceActivityDetector(
            aggressiveness=getattr(self.config, 'VAD_AGGRESSIVENESS', 3),
            sample_rate=self.config.SAMPLE_RATE,
            frame_ms=self.config.VAD_FRAME_MS
        )
        
        self.noise_suppressor = NoiseSuppressor(
            sample_rate=self.config.SAMPLE_RATE,
            reduction_strength=self.config.NOISE_REDUCTION_STRENGTH,
            use_rnnoise=getattr(self.config, 'USE_RNNOISE', True)
        )
        
        # Speech state tracking
        self.speech_state = SpeechState.SILENCE
        self.speech_buffer: List[np.ndarray] = []
        self.hangover_frames = 0
        self.speech_start_frames = 0
        # Calculate hangover and min speech in chunks (based on CHUNK_SIZE)
        chunk_duration_ms = (self.config.CHUNK_SIZE / self.config.SAMPLE_RATE) * 1000
        self.hangover_max_frames = int(
            getattr(self.config, 'HANGOVER_MS', 500) / chunk_duration_ms
        )  # Hangover in chunks
        self.min_speech_frames = int(
            getattr(self.config, 'MIN_SPEECH_MS', 100) / chunk_duration_ms
        )
        logger.info(f"Speech detection: hangover={self.hangover_max_frames} chunks ({getattr(self.config, 'HANGOVER_MS', 500)}ms), min_speech={self.min_speech_frames} chunks ({getattr(self.config, 'MIN_SPEECH_MS', 100)}ms)")
        
        
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        if not self.config.ENABLE_NORMALIZATION:
            return audio
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return audio
        
        # Calculate gain, but limit maximum gain to prevent excessive amplification
        target_rms = 10 ** (self.config.TARGET_DBFS / 20)
        gain = target_rms / rms if rms > 0 else 1.0
        
        # Limit gain to prevent clipping and artifacts (max 100x amplification)
        max_gain = 100.0
        gain = min(gain, max_gain)
        
        normalized = audio * gain
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
    
    def _check_noise_gate(self, audio: np.ndarray) -> bool:
        """Check if audio passes noise gate threshold"""
        if not getattr(self.config, 'ENABLE_NOISE_GATING', True):
            return True
        rms = np.sqrt(np.mean(audio ** 2))
        threshold = getattr(self.config, 'NOISE_GATE_THRESHOLD', 0.005)
        passed = rms >= threshold
        if not passed and rms > 0:
            # Log occasionally if audio is being rejected
            if hasattr(self, '_gate_reject_count'):
                self._gate_reject_count += 1
            else:
                self._gate_reject_count = 1
            if self._gate_reject_count % 100 == 0:  # Log every 100 rejections
                logger.debug(f"Noise gate rejected audio: RMS={rms:.6f}, threshold={threshold}")
        return passed
    
    def _emit_event(self, event_type: str, data: Optional[dict] = None):
        """Emit speech event to callback"""
        if self.event_callback:
            try:
                self.event_callback(event_type, data or {})
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    def process_chunk(self, audio_chunk: np.ndarray, reference_audio: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Process a single audio chunk through the pipeline with speech event detection
        
        IMPROVED PIPELINE ORDER (for 85dB+ factory noise):
        Input → Noise Suppression → Normalize → Noise Gate → VAD → Output
        
        State Machine: SILENCE → SPEECH_START → SPEECH → SPEECH_END → SILENCE
        
        Args:
            audio_chunk: Raw audio chunk (float32, normalized)
            reference_audio: Optional reference audio for echo cancellation
            
        Returns:
            Processed audio chunk or None if no speech detected
        """
        if audio_chunk is None or len(audio_chunk) == 0:
            return None
        
        try:
            # Step 0.5: Echo cancellation (if enabled and reference provided)
            if reference_audio is not None and getattr(self.config, 'ENABLE_AEC', False):
                # AEC processing would go here (interface ready)
                # For now, just pass through
                pass
            
            # Step 1: Noise suppression FIRST (critical for high noise environments)
            # This cleans the audio before VAD, improving detection accuracy
            input_rms = np.sqrt(np.mean(audio_chunk ** 2))
            input_db = 20 * np.log10(input_rms) if input_rms > 0 else -float('inf')
            
            if self.config.ENABLE_NOISE_SUPPRESSION:
                denoised = self.noise_suppressor.reduce_noise(
                    audio_chunk,
                    use_stationary=self.noise_suppressor.is_stationary
                )
                denoised_rms = np.sqrt(np.mean(denoised ** 2))
                denoised_db = 20 * np.log10(denoised_rms) if denoised_rms > 0 else -float('inf')
                ns_improvement = denoised_db - input_db
            else:
                denoised = audio_chunk
                denoised_db = input_db
                ns_improvement = 0.0
            
            # Step 2: Normalize audio level (after noise suppression)
            normalized = self.normalize_audio(denoised)
            normalized_rms = np.sqrt(np.mean(normalized ** 2))
            normalized_db = 20 * np.log10(normalized_rms) if normalized_rms > 0 else -float('inf')
            
            # Step 3: Noise gate (check AFTER normalization for better sensitivity)
            gate_passed = self._check_noise_gate(normalized)
            if not gate_passed:
                # Update state machine for silence
                self._update_speech_state(False)
                return None
            
            # Step 4: Check for voice activity (on CLEAN audio)
            # VAD now sees denoised audio, making detection more accurate
            
            # Step 4: Check for voice activity (on CLEAN audio)
            # First check audio level - reject very quiet audio (likely silence)
            # Minimum audio level threshold to consider for speech detection
            # Below -50dB is likely silence/noise, don't even run VAD
            MIN_AUDIO_LEVEL_DB = -50.0
            
            if normalized_db < MIN_AUDIO_LEVEL_DB:
                # Too quiet, definitely not speech
                is_speech = False
                vad_prob = 0.0
                vad_reason = "TOO_QUIET"
            else:
                # Audio level is sufficient, run VAD
                is_speech = self.vad.is_speech(normalized)
                vad_prob = self.vad.get_speech_probability(normalized)
                vad_reason = "VAD_DECISION"
            
            # Store VAD probability and metrics for server
            self._last_vad_prob = vad_prob
            self._last_audio_metrics = {
                'input_db': input_db,
                'denoised_db': denoised_db,
                'normalized_db': normalized_db,
                'ns_improvement': ns_improvement,
                'gate_passed': gate_passed,
                'vad_reason': vad_reason
            }
            
            # Step 5: Update speech state machine
            self._update_speech_state(is_speech)
            
            # If no speech detected and not in hangover, skip processing
            if not is_speech and self.speech_state == SpeechState.SILENCE:
                return None
            
            # Step 6: Final normalization (light touch-up)
            final_audio = self.normalize_audio(normalized)
            
            # Check for clipping
            if np.any(np.abs(final_audio) > 1.0):
                logger.warning("Audio clipping detected")
                final_audio = np.clip(final_audio, -1.0, 1.0)
            
            # Buffer speech chunks for speech_end event
            if self.speech_state in [SpeechState.SPEECH_START, SpeechState.SPEECH]:
                self.speech_buffer.append(final_audio.copy())
            
            return final_audio
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            return None
    
    def _update_speech_state(self, is_speech: bool):
        """Update speech state machine with hangover logic"""
        if is_speech:
            if self.speech_state == SpeechState.SILENCE:
                # Transition to speech start
                self.speech_state = SpeechState.SPEECH_START
                self.speech_start_frames = 0
                self.speech_buffer = []
                self._emit_event("speech_start", {"timestamp": self._get_timestamp()})
            
            self.speech_start_frames += 1
            
            # After minimum speech duration, transition to SPEECH
            if self.speech_start_frames >= self.min_speech_frames:
                if self.speech_state == SpeechState.SPEECH_START:
                    self.speech_state = SpeechState.SPEECH
            
            # Reset hangover counter
            self.hangover_frames = 0
            
        else:
            # No speech detected
            if self.speech_state == SpeechState.SPEECH:
                # Enter hangover period
                self.hangover_frames += 1
                if self.hangover_frames >= self.hangover_max_frames:
                    # Hangover expired, emit speech_end
                    logger.info(f"Speech hangover expired ({self.hangover_frames} frames), emitting speech_end")
                    self.speech_state = SpeechState.SPEECH_END
                    self._emit_event("speech_end", {
                        "timestamp": self._get_timestamp(),
                        "duration_frames": len(self.speech_buffer),
                        "audio_segment": np.concatenate(self.speech_buffer) if self.speech_buffer else None
                    })
                    self.speech_buffer = []
                    self.speech_state = SpeechState.SILENCE
                    self.hangover_frames = 0
            elif self.speech_state == SpeechState.SPEECH_START:
                # Speech was too short, cancel
                logger.debug(f"Speech too short ({self.speech_start_frames} frames), canceling")
                self.speech_state = SpeechState.SILENCE
                self.speech_buffer = []
                self.speech_start_frames = 0
    
    def _get_timestamp(self) -> float:
        """Get current timestamp (placeholder, can be enhanced)"""
        import time
        return time.time()
    
    def calibrate_noise_profile(self, noise_audio: np.ndarray):
        """Calibrate noise profile from noise-only sample"""
        self.noise_suppressor.estimate_noise_profile(noise_audio)
        logger.info("Noise profile calibrated")