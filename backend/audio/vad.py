"""Voice Activity Detection using WebRTC VAD"""
import webrtcvad
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VAD:
    """WebRTC-based Voice Activity Detection"""
    
    def __init__(self, aggressiveness=3, frame_ms=30, sample_rate=16000, min_speech_ratio=0.0):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.frame_ms = frame_ms
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_ms / 1000)
        # Require this fraction of frames to be speech (0 = any frame; 0.25+ helps reject noise after speech)
        self.min_speech_ratio = float(min_speech_ratio)
        logger.info(f"WebRTC VAD initialized (aggressiveness={aggressiveness}, frame_ms={frame_ms}, sample_rate={sample_rate}, min_speech_ratio={min_speech_ratio})")
    
    def is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio contains speech. If min_speech_ratio > 0, require that fraction of frames to be speech."""
        if len(audio) < self.frame_size:
            return False
        
        # Convert float32 [-1, 1] to int16
        audio_int16 = (audio * 32768.0).astype(np.int16)
        
        num_frames = len(audio_int16) // self.frame_size
        if num_frames == 0:
            return False

        speech_frames = 0
        for i in range(num_frames):
            start_idx = i * self.frame_size
            end_idx = start_idx + self.frame_size
            frame_bytes = audio_int16[start_idx:end_idx].tobytes()
            if self.vad.is_speech(frame_bytes, self.sample_rate):
                speech_frames += 1
                if self.min_speech_ratio <= 0:
                    return True  # "any frame" mode

        if self.min_speech_ratio > 0:
            return (speech_frames / num_frames) >= self.min_speech_ratio
        return False
    
    def get_speech_ratio(self, audio: np.ndarray) -> float:
        """Return fraction of frames classified as speech (0.0 to 1.0). Used for stricter threshold after speech."""
        if len(audio) < self.frame_size:
            return 0.0
        audio_int16 = (audio * 32768.0).astype(np.int16)
        num_frames = len(audio_int16) // self.frame_size
        if num_frames == 0:
            return 0.0
        speech_frames = 0
        for i in range(num_frames):
            start_idx = i * self.frame_size
            end_idx = start_idx + self.frame_size
            frame_bytes = audio_int16[start_idx:end_idx].tobytes()
            if self.vad.is_speech(frame_bytes, self.sample_rate):
                speech_frames += 1
        return speech_frames / num_frames

    def get_probability(self, audio: np.ndarray) -> float:
        """Get speech probability (0.0 to 1.0)"""
        return float(self.get_speech_ratio(audio)) if self.min_speech_ratio > 0 else (1.0 if self.is_speech(audio) else 0.0)
