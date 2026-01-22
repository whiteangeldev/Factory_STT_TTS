"""Audio processing configuration"""
# Note: dotenv is available for future environment variable support
# from dotenv import load_dotenv
# load_dotenv()

class AudioConfig:
    # Audio format settings
    SAMPLE_RATE = 16000  # 16kHz for most STT APIs
    CHANNELS = 1  # Mono
    CHUNK_SIZE = 1024  # Audio chunk size in frames
    FORMAT = 'int16'  # 16-bit PCM
    
    # VAD settings (WebRTC VAD)
    VAD_AGGRESSIVENESS = 3  # 0-3 (0=least, 3=most aggressive). 3 recommended for 85dB+ factory noise
    VAD_FRAME_MS = 30  # Frame size in milliseconds (must be 10, 20, or 30)
    # Note: VAD_THRESHOLD is not used with WebRTC VAD (it uses aggressiveness instead)
    
    # Noise suppression settings
    ENABLE_NOISE_SUPPRESSION = True
    NOISE_REDUCTION_STRENGTH = 0.8  # 0-1, higher = more aggressive
    USE_RNNOISE = True  # Use RNNoise if available (better for high noise 85dB+)
    
    # Audio normalization
    ENABLE_NORMALIZATION = True
    TARGET_DBFS = -20  # Target audio level in dBFS
    
    # Noise gating (disabled by default for quiet environments - enable for noisy factories)
    ENABLE_NOISE_GATING = False  # Set to True for 85dB+ factory noise environments
    NOISE_GATE_THRESHOLD = 0.005  # RMS threshold below which audio is rejected
    
    # Speech detection hangover
    HANGOVER_MS = 500  # Milliseconds to keep speech active after detection ends
    MIN_SPEECH_MS = 100  # Minimum speech duration to trigger speech_start
    
    # AEC (Acoustic Echo Cancellation)
    ENABLE_AEC = False  # Enable echo cancellation (requires reference audio)
    