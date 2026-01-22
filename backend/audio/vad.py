"""Voice Activity Detection using WebRTC VAD"""
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Try to import WebRTC VAD
try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    logger.warning("webrtcvad not available. Install: pip install webrtcvad")

class VoiceActivityDetector:
    """Detects voice activity in audio streams using WebRTC VAD"""
    
    def __init__(self, 
                 aggressiveness: int = 3,
                 sample_rate: int = 16000,
                 frame_ms: int = 30):
        """
        Initialize WebRTC VAD
        
        Args:
            aggressiveness: 0-3 (0=least, 3=most aggressive). 3 recommended for high noise
            sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000)
            frame_ms: Frame duration in ms (must be 10, 20, or 30)
        """
        self.aggressiveness = aggressiveness
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        
        # WebRTC VAD requires specific frame sizes
        if frame_ms not in [10, 20, 30]:
            logger.warning(f"WebRTC VAD requires frame_ms of 10, 20, or 30. Using 30ms")
            self.frame_ms = 30
        
        # WebRTC VAD requires specific sample rates
        if sample_rate not in [8000, 16000, 32000, 48000]:
            logger.warning(f"WebRTC VAD requires sample_rate of 8000, 16000, 32000, or 48000. Using 16000")
            self.sample_rate = 16000
        
        self.frame_size = int(sample_rate * self.frame_ms / 1000)
        
        # Initialize WebRTC VAD
        if WEBRTC_AVAILABLE:
            try:
                self.vad = webrtcvad.Vad(aggressiveness)
                logger.info(f"WebRTC VAD initialized (aggressiveness={aggressiveness}, frame_ms={frame_ms}, sample_rate={sample_rate})")
            except Exception as e:
                logger.error(f"Failed to initialize WebRTC VAD: {e}")
                self.vad = None
        else:
            self.vad = None
            logger.warning("WebRTC VAD not available, using energy-based fallback")
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Detect if audio contains speech
        
        Args:
            audio: Audio array (float32, normalized to [-1, 1])
            
        Returns:
            True if speech detected, False otherwise
        """
        if self.vad is None:
            return self._energy_based_vad(audio)
        
        try:
            # Convert float32 [-1, 1] to int16 PCM
            audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            
            # WebRTC VAD requires exact frame size
            if len(audio_int16) < self.frame_size:
                # Pad if too short
                audio_int16 = np.pad(
                    audio_int16, 
                    (0, self.frame_size - len(audio_int16)), 
                    mode='constant'
                )
            elif len(audio_int16) > self.frame_size:
                # Truncate if too long (take first frame_size samples)
                audio_int16 = audio_int16[:self.frame_size]
            
            # Convert to bytes (required by WebRTC)
            audio_bytes = audio_int16.tobytes()
            
            # WebRTC VAD detection
            return self.vad.is_speech(audio_bytes, self.sample_rate)
            
        except Exception as e:
            logger.error(f"WebRTC VAD error: {e}")
            return self._energy_based_vad(audio)
    
    def get_speech_probability(self, audio: np.ndarray) -> float:
        """
        Get speech probability estimate (0-1) from VAD
        
        Note: WebRTC VAD is binary (speech/no-speech), so this is an estimate
        based on energy and detection result.
        
        Args:
            audio: Audio array (float32, normalized to [-1, 1])
            
        Returns:
            Speech probability estimate (0-1)
        """
        # WebRTC VAD is binary, so estimate probability from energy and detection
        rms = np.sqrt(np.mean(audio ** 2))
        is_speech = self.is_speech(audio)
        
        if is_speech:
            # If detected as speech, return higher probability based on energy
            return min(1.0, 0.5 + (rms * 5))  # Base 0.5 + energy contribution
        else:
            # If not detected, return lower probability
            return min(0.5, rms * 10)  # Energy-based estimate
    
    def _energy_based_vad(self, audio: np.ndarray) -> bool:
        """Fallback energy-based VAD when WebRTC is not available"""
        rms = np.sqrt(np.mean(audio ** 2))
        # Adaptive threshold based on aggressiveness
        threshold = 0.01 * (4 - self.aggressiveness) / 4  # Lower threshold for higher aggressiveness
        return rms > threshold
