"""Configuration for audio processing"""
from dataclasses import dataclass

@dataclass
class AudioConfig:
    """Audio processing configuration"""
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_SIZE: int = 480  # 30ms at 16kHz
    VAD_AGGRESSIVENESS: int = 3  # Higher = more aggressive (less sensitive, filters out more noise)
    VAD_FRAME_MS: int = 30
    WHISPER_MODEL: str = "base"  # "tiny"=fastest, "base"=balanced speed/accuracy, "small"=better accuracy
    MIN_SPEECH_DURATION_MS: int = 200  # Minimum speech duration before triggering transcription (increased to reduce false positives)
    SPEECH_HANGOVER_MS: int = 500  # How long to wait after silence before ending (increased for better speech end detection)
