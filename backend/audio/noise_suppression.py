"""Noise suppression using noisereduce"""
import numpy as np
import noisereduce as nr
import logging

logger = logging.getLogger(__name__)

class NoiseSuppression:
    """Noise suppression using noisereduce library"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        logger.info("Using noisereduce for noise suppression")
    
    def reduce_noise(self, audio: np.ndarray, stationary=True) -> np.ndarray:
        """Reduce noise from audio"""
        if len(audio) < 100:  # Too short for noise reduction
            return audio
        
        try:
            # Use stationary noise reduction for real-time processing
            reduced = nr.reduce_noise(
                y=audio,
                sr=self.sample_rate,
                stationary=stationary,
                prop_decrease=0.8
            )
            return reduced.astype(np.float32)
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}, returning original audio")
            return audio
