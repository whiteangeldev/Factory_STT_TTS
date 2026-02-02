"""RNNoise noise suppression"""
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

try:
    import pyrnnoise as rnnoise
    RNNOISE_AVAILABLE = True
except ImportError:
    RNNOISE_AVAILABLE = False
    logger.warning("rnnoise library not available. Install with: pip install pyrnnoise")
    logger.warning("Falling back to simple noise gate")

class RNNoise:
    """RNNoise-based noise suppression"""
    
    def __init__(self, sample_rate=16000):
        """
        Initialize RNNoise
        
        Args:
            sample_rate: Audio sample rate (RNNoise works with 48000 Hz internally)
        """
        self.sample_rate = sample_rate
        
        if not RNNOISE_AVAILABLE:
            logger.warning("RNNoise not available - using fallback noise gate")
            self.denoiser = None
            self.process_method = None
            return
        
        try:
            # RNNoise works internally at 48000 Hz
            # We'll handle resampling if needed
            self.denoiser = rnnoise.RNNoise(sample_rate=48000)
            # Check which method is available (pyrnnoise uses 'filter', not 'process')
            if hasattr(self.denoiser, 'filter'):
                self.process_method = 'filter'
            elif hasattr(self.denoiser, 'process'):
                self.process_method = 'process'
            elif callable(self.denoiser):
                # Some versions are callable directly
                self.process_method = '__call__'
            else:
                raise AttributeError("RNNoise object has no 'filter' or 'process' method")
            logger.info(f"RNNoise initialized successfully (using method: {self.process_method})")
        except Exception as e:
            logger.error(f"Failed to initialize RNNoise: {e}")
            self.denoiser = None
            self.process_method = None
    
    def reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """
        Reduce noise from audio using RNNoise
        
        Args:
            audio: Audio data as numpy array (float32, range [-1, 1])
        
        Returns:
            Denoised audio (float32, range [-1, 1])
        """
        if len(audio) == 0:
            return audio
        
        # If RNNoise not available, use simple noise gate
        if not self.denoiser:
            return self._simple_noise_gate(audio)
        
        try:
            # RNNoise expects 48000 Hz, 10ms frames (480 samples)
            # If input is 16000 Hz, we need to upsample
            if self.sample_rate == 16000:
                # Upsample to 48000 Hz (3x)
                audio_48k = self._upsample(audio, 3)
            elif self.sample_rate == 48000:
                audio_48k = audio
            else:
                # Resample to 48000 Hz
                ratio = 48000 / self.sample_rate
                audio_48k = self._resample(audio, ratio)
            
            # Process with RNNoise (expects int16)
            audio_int16 = (np.clip(audio_48k, -1.0, 1.0) * 32767.0).astype(np.int16)
            
            # RNNoise processes in 480-sample frames (10ms at 48kHz)
            frame_size = 480
            output_frames = []
            
            for i in range(0, len(audio_int16), frame_size):
                frame = audio_int16[i:i + frame_size]
                
                # Pad if needed
                if len(frame) < frame_size:
                    frame = np.pad(frame, (0, frame_size - len(frame)), mode='constant')
                
                # Process frame using the correct method
                if self.process_method == 'filter':
                    denoised_frame = self.denoiser.filter(frame)
                elif self.process_method == 'process':
                    denoised_frame = self.denoiser.process(frame)
                elif self.process_method == '__call__':
                    denoised_frame = self.denoiser(frame)
                else:
                    raise AttributeError(f"Unknown process method: {self.process_method}")
                output_frames.append(denoised_frame)
            
            # Combine frames
            denoised_48k = np.concatenate(output_frames)[:len(audio_int16)]
            
            # Convert back to float32
            denoised_48k = denoised_48k.astype(np.float32) / 32767.0
            
            # Downsample back to original sample rate if needed
            if self.sample_rate == 16000:
                denoised = self._downsample(denoised_48k, 3)
            elif self.sample_rate == 48000:
                denoised = denoised_48k
            else:
                ratio = self.sample_rate / 48000
                denoised = self._resample(denoised_48k, ratio)
            
            # Clip to valid range
            denoised = np.clip(denoised, -1.0, 1.0)
            
            return denoised.astype(np.float32)
            
        except AttributeError as e:
            # Only log this error once to avoid spam
            if not hasattr(self, '_method_error_logged'):
                logger.error(f"RNNoise method error: {e}. Available methods: {dir(self.denoiser) if self.denoiser else 'N/A'}")
                logger.error("RNNoise will be disabled. Please check pyrnnoise library version.")
                self._method_error_logged = True
                self.denoiser = None  # Disable to prevent repeated errors
            return self._simple_noise_gate(audio)
        except Exception as e:
            # Only log other errors occasionally to avoid spam
            if not hasattr(self, '_processing_error_count'):
                self._processing_error_count = 0
            self._processing_error_count += 1
            if self._processing_error_count <= 3:
                logger.warning(f"RNNoise processing failed: {e}, using original audio")
            elif self._processing_error_count == 4:
                logger.warning("RNNoise processing errors continuing - suppressing further warnings")
            return audio
    
    def _simple_noise_gate(self, audio: np.ndarray, threshold=0.01) -> np.ndarray:
        """Simple noise gate fallback"""
        # Simple noise gate: zero out samples below threshold
        magnitude = np.abs(audio)
        mask = magnitude > threshold
        return audio * mask.astype(np.float32)
    
    def _upsample(self, audio: np.ndarray, factor: int) -> np.ndarray:
        """Simple linear upsampling"""
        try:
            from scipy import signal
            return signal.resample(audio, len(audio) * factor)
        except ImportError:
            # Simple linear interpolation fallback
            indices = np.linspace(0, len(audio) - 1, len(audio) * factor)
            return np.interp(indices, np.arange(len(audio)), audio)
    
    def _downsample(self, audio: np.ndarray, factor: int) -> np.ndarray:
        """Simple linear downsampling"""
        try:
            from scipy import signal
            return signal.resample(audio, len(audio) // factor)
        except ImportError:
            # Simple decimation fallback
            return audio[::factor]
    
    def _resample(self, audio: np.ndarray, ratio: float) -> np.ndarray:
        """Resample audio by ratio"""
        try:
            from scipy import signal
            new_length = int(len(audio) * ratio)
            return signal.resample(audio, new_length)
        except ImportError:
            # Simple linear interpolation fallback
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio)
