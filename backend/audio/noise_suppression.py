"""Noise suppression module with RNNoise and noisereduce support"""
import noisereduce as nr
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try to import RNNoise (better for high noise)
try:
    try:
        from pyrnnoise import RNNoise
        RNNOISE_AVAILABLE = True
        RNNOISE_LIB = "pyrnnoise"
    except ImportError:
        try:
            # Alternative: rnnoise-python
            import rnnoise
            RNNOISE_AVAILABLE = True
            RNNOISE_LIB = "rnnoise-python"
        except ImportError:
            RNNOISE_AVAILABLE = False
            RNNOISE_LIB = None
except Exception as e:
    RNNOISE_AVAILABLE = False
    RNNOISE_LIB = None
    logger.warning(f"RNNoise not available: {e}. Using noisereduce fallback. Install: pip install pyrnnoise")

class NoiseSuppressor:
    """Suppresses background noise from audio"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 reduction_strength: float = 0.8,
                 use_rnnoise: bool = True):
        self.sample_rate = sample_rate
        self.reduction_strength = reduction_strength
        self.noise_profile = None
        self.is_stationary = True
        
        # Initialize RNNoise if available and requested
        self.use_rnnoise = use_rnnoise and RNNOISE_AVAILABLE
        self.rnnoise = None
        self.rnnoise_sample_rate = 48000  # RNNoise requires 48kHz
        
        if self.use_rnnoise:
            try:
                if RNNOISE_LIB == "pyrnnoise":
                    self.rnnoise = RNNoise()
                    logger.info("RNNoise (pyrnnoise) initialized (48kHz)")
                elif RNNOISE_LIB == "rnnoise-python":
                    self.rnnoise = rnnoise.RNNoise()
                    logger.info("RNNoise (rnnoise-python) initialized (48kHz)")
                else:
                    raise ImportError("No RNNoise library available")
            except Exception as e:
                logger.error(f"Failed to initialize RNNoise: {e}")
                self.use_rnnoise = False
        
        if not self.use_rnnoise:
            logger.info("Using noisereduce for noise suppression")
    
    def estimate_noise_profile(self, noise_audio: np.ndarray):
        """
        Estimate noise profile from noise-only audio
        
        Args:
            noise_audio: Audio containing only noise (no speech)
        """
        try:
            # Use first 0.5 seconds as noise profile
            noise_sample = noise_audio[:int(self.sample_rate * 0.5)]
            self.noise_profile = noise_sample
            logger.info("Noise profile estimated")
        except Exception as e:
            logger.error(f"Error estimating noise profile: {e}")
    
    def _resample_audio(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Resample audio using linear interpolation"""
        if from_rate == to_rate:
            return audio
        
        ratio = to_rate / from_rate
        new_length = int(len(audio) * ratio)
        resampled = np.zeros(new_length, dtype=np.float32)
        
        for i in range(new_length):
            src_index = i / ratio
            src_index_floor = int(src_index)
            src_index_ceil = min(src_index_floor + 1, len(audio) - 1)
            t = src_index - src_index_floor
            
            if src_index_floor < len(audio):
                resampled[i] = audio[src_index_floor] * (1 - t) + audio[src_index_ceil] * t
        
        return resampled
    
    def reduce_noise(self, audio: np.ndarray, 
                    use_stationary: bool = None) -> np.ndarray:
        """
        Reduce noise from audio using RNNoise (if available) or noisereduce
        
        Args:
            audio: Input audio array (float32, normalized)
            use_stationary: Use stationary noise reduction (for noisereduce only)
            
        Returns:
            Denoised audio array
        """
        if use_stationary is None:
            use_stationary = self.is_stationary
        
        try:
            if self.use_rnnoise and self.rnnoise:
                # Use RNNoise (better for high noise like 85dB+)
                # RNNoise requires 48kHz, so resample if needed
                if self.sample_rate != self.rnnoise_sample_rate:
                    audio_48k = self._resample_audio(audio, self.sample_rate, self.rnnoise_sample_rate)
                else:
                    audio_48k = audio
                
                # RNNoise processes in 480-sample frames (10ms at 48kHz)
                frame_size = 480
                denoised_chunks = []
                
                for i in range(0, len(audio_48k), frame_size):
                    frame = audio_48k[i:i + frame_size]
                    if len(frame) < frame_size:
                        # Pad last frame
                        frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
                    
                    # Process frame with RNNoise
                    if RNNOISE_LIB == "pyrnnoise":
                        denoised_frame = self.rnnoise.process(frame)
                    elif RNNOISE_LIB == "rnnoise-python":
                        denoised_frame = self.rnnoise.process(frame)
                    else:
                        denoised_frame = frame
                    denoised_chunks.append(denoised_frame)
                
                # Combine denoised chunks
                denoised_48k = np.concatenate(denoised_chunks) if denoised_chunks else audio_48k
                
                # Resample back to original sample rate
                if self.sample_rate != self.rnnoise_sample_rate:
                    denoised = self._resample_audio(denoised_48k, self.rnnoise_sample_rate, self.sample_rate)
                else:
                    denoised = denoised_48k
                
                # Trim to original length (in case of padding)
                if len(denoised) > len(audio):
                    denoised = denoised[:len(audio)]
                elif len(denoised) < len(audio):
                    denoised = np.pad(denoised, (0, len(audio) - len(denoised)), mode='constant')
                
                return np.clip(denoised, -1.0, 1.0).astype(np.float32)
            
            else:
                # Fallback to noisereduce
                if use_stationary:
                    reduced = nr.reduce_noise(
                        y=audio,
                        sr=self.sample_rate,
                        stationary=True,
                        prop_decrease=self.reduction_strength
                    )
                else:
                    reduced = nr.reduce_noise(
                        y=audio,
                        sr=self.sample_rate,
                        stationary=False,
                        prop_decrease=self.reduction_strength
                    )
                
                if isinstance(reduced, np.ndarray):
                    reduced = reduced.astype(np.float32)
                    reduced = np.clip(reduced, -1.0, 1.0)
                
                return reduced
                
        except Exception as e:
            logger.error(f"Error reducing noise: {e}")
            return audio
    