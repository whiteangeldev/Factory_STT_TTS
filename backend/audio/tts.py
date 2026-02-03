"""Text-to-Speech module for multi-language TTS"""
import logging
import tempfile
from pathlib import Path
from typing import Optional

# Try to import required dependencies (make them optional)
try:
    import numpy as np
    import soundfile as sf
    _HAS_CORE_DEPS = True
except ImportError:
    _HAS_CORE_DEPS = False
    np = None
    sf = None

try:
    import torch
    from transformers import (
        AutoProcessor,
        SpeechT5ForTextToSpeech,
        SpeechT5HifiGan,
        SpeechT5Processor,
        VitsModel,
    )
    from datasets import load_dataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None

# Try to import gTTS for Japanese and Chinese support
try:
    from gtts import gTTS
    import subprocess
    _HAS_GTTS = True
except ImportError:
    _HAS_GTTS = False

# Try to import librosa for speed/tempo adjustment
try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

# Try to import pydub for audio conversion
try:
    from pydub import AudioSegment
    _HAS_PYDUB = True
except ImportError:
    _HAS_PYDUB = False

logger = logging.getLogger(__name__)

# Language to MMS-TTS model mapping (only for English - other languages use gTTS)
LANGUAGE_MODEL_MAP = {
    "en": "facebook/mms-tts-eng",
    "eng": "facebook/mms-tts-eng",
    "english": "facebook/mms-tts-eng",
}


def detect_language(text: str) -> str:
    """
    Detect language from text using character-based heuristics.
    
    Returns:
        Language code: "en", "zh", or "ja"
    """
    if not text:
        return "en"
    
    # Count character types
    chinese_chars = 0
    japanese_chars = 0
    total_chars = 0
    
    for char in text:
        code = ord(char)
        total_chars += 1
        
        # Chinese characters (CJK Unified Ideographs)
        if (0x4E00 <= code <= 0x9FFF) or (0x3400 <= code <= 0x4DBF) or (0x20000 <= code <= 0x2A6DF):
            chinese_chars += 1
        # Japanese characters (Hiragana, Katakana, CJK)
        elif (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF) or (0x4E00 <= code <= 0x9FFF):
            japanese_chars += 1
    
    # If no special characters, default to English
    if total_chars == 0:
        return "en"
    
    # Calculate ratios
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
    japanese_ratio = japanese_chars / total_chars if total_chars > 0 else 0
    
    # Determine language based on character presence
    # If significant Chinese characters, likely Chinese
    if chinese_ratio > 0.1:
        # Check if it's Japanese by looking for Hiragana/Katakana
        if japanese_ratio > 0.05:
            return "ja"
        return "zh"
    
    # If significant Japanese characters, likely Japanese
    if japanese_ratio > 0.1:
        return "ja"
    
    # Default to English
    return "en"

# Language to gTTS language code mapping (for Japanese and Chinese)
GTTS_LANGUAGE_MAP = {
    "ja": "ja",
    "jpn": "ja",
    "japanese": "ja",
    "zh": "zh",
    "cmn": "zh",
    "zho": "zh",
    "chinese": "zh",
    "mandarin": "zh",
    "zh-cn": "zh-cn",
    "zh-tw": "zh-tw",
}


def _apply_speed_adjustment(
    wav: np.ndarray,
    sr: int,
    speed: float = 1.0,
) -> tuple[np.ndarray, int]:
    """
    Apply speed adjustment to audio using librosa time_stretch.
    Returns adjusted audio and original sampling rate.
    """
    if abs(speed - 1.0) < 1e-6:
        return wav, sr
    
    if not _HAS_LIBROSA:
        raise RuntimeError(
            "Speed adjustment requires librosa. Install with: pip install librosa"
        )
    
    # Ensure speed is reasonable
    speed = max(0.5, min(2.0, speed))  # Clamp between 0.5x and 2.0x
    
    # Apply time stretch (rate > 1.0 = faster, < 1.0 = slower)
    adjusted = librosa.effects.time_stretch(wav.astype(np.float32), rate=speed)
    return adjusted.astype(np.float32), sr


def synthesize_speech(
    text: str,
    language: str = "auto",
    speed: float = 1.0,
    device_preference: str = "auto",
) -> tuple[bytes, int]:
    """Synthesize speech from text and return audio data as bytes.

    Args:
        text: The content to speak.
        language: Language code ("en", "ja", "zh", etc.) or "auto" for auto-detection.
        speed: Playback speed multiplier (1.0 = normal, 1.2 = 20% faster, 0.9 = 10% slower).
        device_preference: Device to use ("auto", "cpu", or "mps").

    Returns:
        Tuple of (audio_bytes, sample_rate) where audio_bytes is WAV file bytes.
    """
    if not _HAS_CORE_DEPS:
        raise RuntimeError(
            "TTS requires numpy and soundfile. Install with: pip install numpy soundfile"
        )
    
    if not text:
        raise ValueError("Text must not be empty.")

    # Auto-detect language if not specified or set to "auto"
    if language.lower().strip() in ("auto", ""):
        language = detect_language(text)
        logger.info(f"Auto-detected language: {language} for text: '{text[:50]}...'")
    
    language_lower = language.lower().strip()
    model_id = LANGUAGE_MODEL_MAP.get(language_lower)
    
    # If MMS-TTS model exists for this language, use it
    if model_id is not None and _HAS_TORCH:
        try:
            logger.info(f"Using MMS-TTS model for language: {language} (offline-capable)")
            
            # Set up device
            if device_preference == "mps":
                device = (
                    torch.device("mps")
                    if torch.backends.mps.is_available()
                    else torch.device("cpu")
                )
            elif device_preference == "cpu":
                device = torch.device("cpu")
            else:
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            
            # Load model with local_files_only=True to use cached model if available
            # This allows offline operation if model was previously downloaded
            try:
                model = VitsModel.from_pretrained(model_id, local_files_only=True).to(device)
                processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
                logger.debug("Loaded MMS-TTS model from local cache (offline mode)")
            except (OSError, ValueError) as cache_error:
                # Model not in cache, try downloading (requires internet)
                logger.info("Model not in cache, attempting to download (requires internet)...")
                try:
                    model = VitsModel.from_pretrained(model_id).to(device)
                    processor = AutoProcessor.from_pretrained(model_id)
                    logger.info("Model downloaded and cached successfully")
                except Exception as download_error:
                    # Check if it's a network error
                    error_msg = str(download_error).lower()
                    if any(keyword in error_msg for keyword in ["connection", "network", "timeout", "unreachable", "offline"]):
                        raise RuntimeError(
                            f"MMS-TTS model not found in cache and cannot download (no internet). "
                            f"Please download the model first with internet: "
                            f"python -c \"from transformers import VitsModel; VitsModel.from_pretrained('{model_id}')\""
                        ) from download_error
                    else:
                        raise download_error
            inputs = processor(text=text, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            try:
                with torch.no_grad():
                    outputs = model(**inputs)
            except RuntimeError as e:
                if str(device) == "mps":
                    # Fallback to CPU if an op is unsupported on MPS
                    logger.warning("MPS operation failed, falling back to CPU")
                    model = model.to("cpu")
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = model(**inputs)
                else:
                    raise e
            speech = outputs.waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
            sampling_rate = getattr(model.config, "sampling_rate", 16000)
            
            # Apply speed adjustment if requested
            if abs(speed - 1.0) > 1e-6:
                speech, sampling_rate = _apply_speed_adjustment(speech, sampling_rate, speed)
            
            # Convert to bytes (WAV format)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                tmp_wav_path = tmp_wav.name
            
            try:
                sf.write(tmp_wav_path, np.clip(speech, -1.0, 1.0), samplerate=sampling_rate)
                with open(tmp_wav_path, "rb") as f:
                    audio_bytes = f.read()
                Path(tmp_wav_path).unlink()
                return audio_bytes, sampling_rate
            except Exception as e:
                if Path(tmp_wav_path).exists():
                    Path(tmp_wav_path).unlink()
                raise e
        except RuntimeError as e:
            # Re-raise RuntimeError (offline errors) so they're handled properly
            error_msg = str(e).lower()
            if "not found in cache" in error_msg or "cannot download" in error_msg:
                raise e
            else:
                # Other runtime errors, try to fall through to gTTS if available
                logger.warning(f"MMS-TTS error: {e}, attempting fallback")
                pass
        except Exception as e:
            # If model loading fails, fall through to gTTS
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["not a valid model identifier", "does not exist"]):
                logger.info(f"MMS-TTS model not available, falling back to gTTS: {e}")
                pass  # Fall through to gTTS
            else:
                raise e
    
    # Fallback to gTTS for Japanese and Chinese (and other unsupported languages)
    # Note: gTTS requires internet connection
    gtts_lang = GTTS_LANGUAGE_MAP.get(language_lower)
    if gtts_lang and _HAS_GTTS:
        logger.info(f"Using gTTS for language: {language} (code: {gtts_lang}) - requires internet")
        # Use gTTS for Japanese and Chinese
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
            tmp_mp3_path = tmp_mp3.name
        
        try:
            # gTTS requires internet - will raise exception if offline
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            tts.save(tmp_mp3_path)
            
            # Convert MP3 to WAV using pydub or ffmpeg
            if _HAS_PYDUB:
                try:
                    audio = AudioSegment.from_mp3(tmp_mp3_path)
                    # Convert to numpy array for speed adjustment
                    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                    if audio.channels == 2:
                        samples = samples.reshape((-1, 2)).mean(axis=1)  # Convert stereo to mono
                    samples = samples / (1 << (8 * audio.sample_width - 1))  # Normalize
                    sr = audio.frame_rate
                    
                    # Apply speed adjustment if requested
                    if abs(speed - 1.0) > 1e-6:
                        samples, sr = _apply_speed_adjustment(samples, sr, speed)
                    
                    # Convert to bytes (WAV format)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                        tmp_wav_path = tmp_wav.name
                    
                    sf.write(tmp_wav_path, np.clip(samples, -1.0, 1.0), samplerate=int(sr))
                    with open(tmp_wav_path, "rb") as f:
                        audio_bytes = f.read()
                    Path(tmp_mp3_path).unlink()
                    Path(tmp_wav_path).unlink()
                    return audio_bytes, int(sr)
                except Exception as e:
                    logger.error(f"Error converting MP3 with pydub: {e}")
                    raise e
            else:
                # Fallback to ffmpeg if pydub not available
                try:
                    # First convert to WAV
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                        tmp_wav_path = tmp_wav.name
                    
                    subprocess.run(
                        ["ffmpeg", "-i", tmp_mp3_path, "-y", tmp_wav_path],
                        check=True,
                        capture_output=True,
                    )
                    Path(tmp_mp3_path).unlink()
                    
                    # Load and apply speed adjustment
                    wav_data, sr = sf.read(tmp_wav_path)
                    if len(wav_data.shape) > 1:
                        wav_data = wav_data.mean(axis=1)  # Convert stereo to mono
                    
                    if abs(speed - 1.0) > 1e-6:
                        wav_data, sr = _apply_speed_adjustment(wav_data, sr, speed)
                    
                    # Convert to bytes
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav2:
                        tmp_wav2_path = tmp_wav2.name
                    
                    sf.write(tmp_wav2_path, np.clip(wav_data, -1.0, 1.0), samplerate=int(sr))
                    with open(tmp_wav2_path, "rb") as f:
                        audio_bytes = f.read()
                    Path(tmp_wav_path).unlink()
                    Path(tmp_wav2_path).unlink()
                    return audio_bytes, int(sr)
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    Path(tmp_mp3_path).unlink()
                    if Path(tmp_wav_path).exists():
                        Path(tmp_wav_path).unlink()
                    raise ValueError(
                        "gTTS requires either 'pydub' or 'ffmpeg' to convert MP3 to WAV. "
                        "Install with: pip install pydub"
                    )
        except Exception as e:
            if Path(tmp_mp3_path).exists():
                Path(tmp_mp3_path).unlink()
            # Check if it's a network/connection error
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["connection", "network", "timeout", "unreachable", "offline", "failed to connect"]):
                raise RuntimeError(
                    f"gTTS requires internet connection. Cannot synthesize {language} text offline. "
                    f"For offline TTS, use English (MMS-TTS) which works without internet after initial model download."
                ) from e
            raise e
    
    # If we get here, language is not supported or dependencies are missing
    supported = []
    if _HAS_TORCH:
        supported.append("en/english (MMS-TTS - works offline after model download)")
    if _HAS_GTTS:
        supported.extend(["ja/japanese (gTTS - requires internet)", "zh/chinese (gTTS - requires internet)"])
    
    if not supported:
        raise RuntimeError(
            f"TTS is not available. Install dependencies:\n"
            f"  - For English (offline-capable): pip install torch transformers datasets soundfile\n"
            f"  - For Chinese/Japanese (requires internet): pip install gtts pydub"
        )
    
    raise ValueError(
        f"Unsupported language: {language}. "
        f"Supported languages: {', '.join(supported) if supported else 'None (install dependencies)'}"
    )
