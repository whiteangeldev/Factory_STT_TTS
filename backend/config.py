"""Configuration for audio processing"""
from dataclasses import dataclass

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_SIZE: int = 480  # 30ms at 16kHz
    VAD_AGGRESSIVENESS: int = 3  # Higher = less sensitive, fewer false positives (0=least aggressive/most sensitive, 3=most aggressive/least sensitive)
    VAD_FRAME_MS: int = 30
    VAD_MIN_SPEECH_RATIO: float = 0.25  # Require this fraction of frames to be speech (0=any frame; 0.25=avoid noise after speech)
    WHISPER_MODEL: str = "small"  # "tiny"=fastest, "base"=balanced speed/accuracy, "small"=better accuracy
    MIN_SPEECH_DURATION_MS: int = 400  # Minimum speech duration before triggering transcription (increased to reduce false positives from background noise)
    SPEECH_HANGOVER_MS: int = 500  # How long to wait after silence before ending (increased for better speech end detection)
