"""Voice Activity Detection using WebRTC VAD"""
import webrtcvad
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VAD:
    """WebRTC-based Voice Activity Detection"""
    
    def __init__(self, aggressiveness=3, frame_ms=30, sample_rate=16000):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_ms = frame_ms
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_ms / 1000)
        logger.info(f"WebRTC VAD initialized (aggressiveness={aggressiveness}, frame_ms={frame_ms}, sample_rate={sample_rate})")
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio contains speech - processes all frames in the chunk"""
        if len(audio) < self.frame_size:
            return False
        
        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio * 32768.0).astype(np.int16)
        
        # Process all complete frames in the chunk
        # If any frame contains speech, return True
        num_frames = len(audio_int16) // self.frame_size
        if num_frames == 0:
            return False
        
        for i in range(num_frames):
            start_idx = i * self.frame_size
            end_idx = start_idx + self.frame_size
            frame_bytes = audio_int16[start_idx:end_idx].tobytes()
            if self.vad.is_speech(frame_bytes, self.sample_rate):
                return True
        
        return False
    
    def get_probability(self, audio: np.ndarray) -> float:
        """Get speech probability (0.0 to 1.0)"""
        # WebRTC VAD is binary, so return 1.0 if speech, 0.0 otherwise
        return 1.0 if self.is_speech(audio) else 0.0
