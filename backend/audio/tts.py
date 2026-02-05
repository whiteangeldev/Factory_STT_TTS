"""Text-to-Speech module for multi-language TTS"""
import io
import json
import logging
import time
from pathlib import Path

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
    from transformers import AutoProcessor, VitsModel
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None

# Try to import Piper TTS for offline Chinese support
try:
    from piper import PiperVoice
    _HAS_PIPER_TTS = True
except ImportError:
    _HAS_PIPER_TTS = False
    PiperVoice = None

# Try to import librosa for speed/tempo adjustment
try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

# Try to import PyKokoro for Japanese TTS (alternative to MMS/OpenJTalk/Piper)
try:
    from pykokoro import build_pipeline
    _HAS_PYKOKORO = True
except ImportError:
    _HAS_PYKOKORO = False
    build_pipeline = None



logger = logging.getLogger(__name__)

# Cache for loaded PiperVoice instances to avoid reloading on each request
_piper_voice_cache = {}  # Maps voice_name -> (voice_instance, config_path, sample_rate)

# Cache for loaded MMS-TTS models to avoid reloading on each request
_mms_model_cache = {}  # Maps (model_id, device_str) -> (model, processor)

# Cache for PyKokoro TTS instance
_pykokoro_cache = None


# Language to MMS-TTS model mapping (offline-capable)
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
        # Japanese characters (Hiragana, Katakana)
        elif (0x3040 <= code <= 0x309F) or (0x30A0 <= code <= 0x30FF):
            japanese_chars += 1
    
    # If no special characters, default to English
    if total_chars == 0:
        return "en"
    
    # Calculate ratios
    chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
    japanese_ratio = japanese_chars / total_chars if total_chars > 0 else 0
    
    # Determine language based on character presence
    # If significant Japanese characters, likely Japanese
    if japanese_ratio > 0.1:
        return "ja"
    # If significant Chinese characters, likely Chinese
    if chinese_ratio > 0.1:
        return "zh"
    
    # Default to English
    return "en"

# Piper TTS voice mapping (offline-capable)
PIPER_TTS_VOICES = {
    # Chinese voices
    "zh": "zh_CN/xiaoyan/medium",
    "cmn": "zh_CN/xiaoyan/medium",
    "zho": "zh_CN/xiaoyan/medium",
    "chinese": "zh_CN/xiaoyan/medium",
    "mandarin": "zh_CN/xiaoyan/medium",
    "zh-cn": "zh_CN/xiaoyan/medium",
}


def _ensure_piper_voice(voice_name: str) -> tuple[Path, Path]:
    """Ensure Piper TTS voice is available, download if needed.
    
    Returns:
        Tuple of (model_path, config_path)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub not installed. Install with: pip install huggingface_hub"
        )
    
    # Extract voice path components
    # voice_name format: "zh_CN/xiaoyan/medium" or "ja/kokoro/medium"
    parts = voice_name.split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid voice name format: {voice_name}")
    
    lang_code_short = parts[0].split("_")[0]  # "zh" or "ja"
    voice_path = "/".join(parts[:2])  # "zh_CN/xiaoyan" or "ja/kokoro"
    quality = parts[2]  # "medium"
    
    # Model filename format: zh_CN-xiaoyan-medium.onnx or ja-kokoro-medium.onnx
    model_filename = f"{parts[0]}-{parts[1]}-{quality}.onnx"
    config_filename = f"{parts[0]}-{parts[1]}-{quality}.onnx.json"
    
    # Cache directory
    cache_dir = Path.home() / ".local" / "share" / "piper" / "voices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Voice directory - use the full voice name path structure
    voice_dir = cache_dir / voice_name
    voice_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = voice_dir / model_filename
    config_path = voice_dir / config_filename
    
    # Check if already downloaded
    if model_path.exists() and config_path.exists():
        return model_path, config_path
    
    # Download if not cached using huggingface_hub
    try:
        logger.info(f"Downloading Piper TTS voice: {voice_name}")
        repo_id = "rhasspy/piper-voices"
        # Path format depends on whether language has region code
        # For Chinese (zh_CN/xiaoyan/medium): zh/zh_CN/xiaoyan/zh_CN-xiaoyan-medium.onnx
        # For Japanese (ja/kokoro/medium): ja/kokoro/ja-kokoro-medium.onnx (no duplicate ja)
        if "_" in parts[0]:
            # Has region code (e.g., zh_CN) - use lang_code_short prefix
            model_file_path = f"{lang_code_short}/{voice_path}/{model_filename}"
            config_file_path = f"{lang_code_short}/{voice_path}/{config_filename}"
        else:
            # No region code (e.g., ja) - use voice_path directly
            model_file_path = f"{voice_path}/{model_filename}"
            config_file_path = f"{voice_path}/{config_filename}"
        
        # Download to default HF cache first, then copy to target location
        from huggingface_hub import snapshot_download
        import shutil
        
        # Download to temporary location
        temp_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"temp_piper_{voice_name.replace('/', '_')}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download model file
            downloaded_model = hf_hub_download(
                repo_id=repo_id,
                filename=model_file_path,
                cache_dir=str(temp_dir),
                local_files_only=False
            )
            
            # Download config file
            downloaded_config = hf_hub_download(
                repo_id=repo_id,
                filename=config_file_path,
                cache_dir=str(temp_dir),
                local_files_only=False
            )
            
            # Copy to target location
            import shutil
            shutil.copy2(downloaded_model, model_path)
            shutil.copy2(downloaded_config, config_path)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            logger.info("✓ Voice downloaded successfully")
            return model_path, config_path
        except Exception as download_error:
            # Clean up temp directory on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise download_error
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["connection", "network", "timeout", "unreachable", "offline", "not found", "404"]):
            raise RuntimeError(
                f"Piper TTS voice not found in cache and cannot download (no internet or voice not found). "
                f"Please download manually or ensure internet connection. "
                f"Check available voices at: https://huggingface.co/rhasspy/piper-voices"
            ) from e
        else:
            raise e


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
    
    # For Japanese: Use PyKokoro (alternative to MMS/OpenJTalk/Piper)
    if language_lower in ["ja", "jpn", "japanese"]:
        if _HAS_PYKOKORO:
            try:
                logger.info(f"Using PyKokoro for Japanese TTS (offline-capable)")
                
                # Check cache first
                global _pykokoro_cache
                if _pykokoro_cache is None:
                    logger.info("Initializing PyKokoro TTS pipeline with Japanese language support...")
                    # Configure pipeline for Japanese language
                    try:
                        from pykokoro import PipelineConfig, GenerationConfig
                        config = PipelineConfig(
                            generation=GenerationConfig(lang='ja')
                        )
                        _pykokoro_cache = build_pipeline(config=config)
                        logger.info("✓ PyKokoro pipeline initialized and cached (Japanese mode)")
                    except Exception as config_error:
                        logger.warning(f"Failed to configure Japanese language, using default: {config_error}")
                        _pykokoro_cache = build_pipeline()
                        logger.info("✓ PyKokoro pipeline initialized and cached (default mode)")
                else:
                    logger.debug("Using cached PyKokoro pipeline")
                
                # Synthesize with PyKokoro
                # pipeline.run() returns AudioResult with audio data
                # Use generation parameter with lang='ja' to ensure Japanese synthesis
                from pykokoro import GenerationConfig
                result = _pykokoro_cache.run(text, generation=GenerationConfig(lang='ja'))
                
                # Extract audio data from AudioResult
                # AudioResult has .audio (numpy array) and .sample_rate
                audio_array = result.audio
                sampling_rate = result.sample_rate
                
                # Ensure float32 format and normalize
                wav = audio_array.astype(np.float32)
                # If stereo, convert to mono
                if len(wav.shape) > 1:
                    wav = np.mean(wav, axis=1)
                # Normalize to [-1, 1] range if needed
                max_val = np.abs(wav).max()
                if max_val > 1.0:
                    wav = wav / max_val
                elif max_val > 0:
                    # If values are in int16 range, normalize
                    if max_val > 32767:
                        wav = wav / 32768.0
                
                # Apply speed adjustment if requested
                if abs(speed - 1.0) > 1e-6:
                    wav, sampling_rate = _apply_speed_adjustment(wav, sampling_rate, speed)
                
                # Convert to bytes (WAV format)
                output_buffer = io.BytesIO()
                try:
                    sf.write(output_buffer, np.clip(wav, -1.0, 1.0), samplerate=sampling_rate, format='WAV')
                    audio_bytes = output_buffer.getvalue()
                    return audio_bytes, sampling_rate
                finally:
                    output_buffer.close()
            except Exception as e:
                error_msg = str(e)
                logger.error(f"PyKokoro synthesis error: {e}")
                
                # Provide helpful error messages for common issues
                if "spacy" in error_msg.lower() or "en_core_web_sm" in error_msg.lower():
                    raise RuntimeError(
                        f"PyKokoro requires spaCy language models. "
                        f"Install with:\n"
                        f"  pip install spacy\n"
                        f"  python -m spacy download en_core_web_sm\n"
                        f"  python -m spacy download ja_core_news_sm\n"
                        f"Original error: {error_msg}"
                    ) from e
                else:
                    raise RuntimeError(
                        f"PyKokoro failed for Japanese. Error: {error_msg}. "
                        f"Install with: pip install pykokoro"
                    ) from e
        else:
            raise RuntimeError(
                "Japanese TTS requires PyKokoro. "
                "Install with: pip install pykokoro"
            )
    
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
            
            # Check cache first to avoid reloading model on each request
            cache_key = f"{model_id}_{device}"
            if cache_key in _mms_model_cache:
                model, processor = _mms_model_cache[cache_key]
                logger.debug(f"Using cached MMS-TTS model: {cache_key}")
            else:
                # Load model with local_files_only=True to use cached model if available
                # This allows offline operation if model was previously downloaded
                try:
                    logger.info(f"Loading MMS-TTS model: {model_id} (this may take a few seconds on first load)")
                    model = VitsModel.from_pretrained(model_id, local_files_only=True).to(device)
                    logger.debug("Loaded MMS-TTS model from local cache")
                    
                    # Try to load processor from cache
                    try:
                        processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
                        logger.debug("Loaded MMS-TTS processor from local cache")
                    except (OSError, ValueError) as processor_error:
                        # Processor not in cache, try downloading (requires internet)
                        logger.warning(f"Processor not in cache: {processor_error}")
                        logger.info("Attempting to download processor (requires internet)...")
                        try:
                            processor = AutoProcessor.from_pretrained(model_id)
                            logger.info("Processor downloaded and cached successfully")
                        except Exception as proc_download_error:
                            error_msg = str(proc_download_error).lower()
                            if any(keyword in error_msg for keyword in ["connection", "network", "timeout", "unreachable", "offline", "closed", "client"]):
                                raise RuntimeError(
                                    f"MMS-TTS processor not found in cache and cannot download (no internet or connection closed). "
                                    f"Please download the model and processor first with internet: "
                                    f"python -c \"from transformers import VitsModel, AutoProcessor; "
                                    f"VitsModel.from_pretrained('{model_id}'); AutoProcessor.from_pretrained('{model_id}')\""
                                ) from proc_download_error
                            else:
                                raise proc_download_error
                    
                except (OSError, ValueError) as cache_error:
                    # Model not in cache, try downloading (requires internet)
                    logger.info("Model not in cache, attempting to download (requires internet)...")
                    try:
                        model = VitsModel.from_pretrained(model_id).to(device)
                        processor = AutoProcessor.from_pretrained(model_id)
                        logger.info("Model and processor downloaded and cached successfully")
                    except Exception as download_error:
                        # Check if it's a network error
                        error_msg = str(download_error).lower()
                        if any(keyword in error_msg for keyword in ["connection", "network", "timeout", "unreachable", "offline", "closed", "client"]):
                            raise RuntimeError(
                                f"MMS-TTS model not found in cache and cannot download (no internet or connection closed). "
                                f"Please download the model first with internet: "
                                f"python -c \"from transformers import VitsModel, AutoProcessor; "
                                f"VitsModel.from_pretrained('{model_id}'); AutoProcessor.from_pretrained('{model_id}')\""
                            ) from download_error
                        else:
                            raise download_error
                
                # Cache the loaded model for future requests
                _mms_model_cache[cache_key] = (model, processor)
                logger.info(f"Cached MMS-TTS model: {cache_key}")
            
            # Standard MMS-TTS synthesis
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
            
            # Convert to bytes (WAV format) - use in-memory buffer instead of temp file
            wav_buffer = io.BytesIO()
            try:
                sf.write(wav_buffer, np.clip(speech, -1.0, 1.0), samplerate=sampling_rate, format='WAV')
                audio_bytes = wav_buffer.getvalue()
                return audio_bytes, sampling_rate
            finally:
                wav_buffer.close()
        except RuntimeError as e:
            # Re-raise RuntimeError (offline errors) so they're handled properly
            error_msg = str(e).lower()
            if "not found in cache" in error_msg or "cannot download" in error_msg:
                raise e
            else:
                # Other runtime errors - no fallback available
                logger.error(f"MMS-TTS error: {e}")
                raise e
        except Exception as e:
            # If model loading fails, raise error
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in ["not a valid model identifier", "does not exist"]):
                raise RuntimeError(
                    f"MMS-TTS model not available for language: {language}. "
                    f"Supported languages: English (en)"
                ) from e
            else:
                raise e
    
    # For Chinese: Try Piper TTS first (offline-capable)
    piper_voice = PIPER_TTS_VOICES.get(language_lower)
    if piper_voice and _HAS_PIPER_TTS:
        try:
            logger.info(f"Using Piper TTS for language: {language} (offline-capable)")
            
            # Check cache first to avoid reloading model on each request
            if piper_voice in _piper_voice_cache:
                voice, config_path, sample_rate = _piper_voice_cache[piper_voice]
                logger.debug(f"Using cached Piper voice: {piper_voice}")
            else:
                # Ensure voice exists (downloads if needed)
                try:
                    voice_path, config_path = _ensure_piper_voice(piper_voice)
                    logger.info(f"Loading Piper voice model: {piper_voice} (this may take a few seconds on first load)")
                    voice = PiperVoice.load(voice_path, config_path)
                    
                    # Get sample rate from voice config
                    try:
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                            sample_rate = config.get('audio', {}).get('sample_rate', 22050)
                    except:
                        sample_rate = 22050  # Default sample rate for Piper
                    
                    # Cache the loaded voice for future requests
                    _piper_voice_cache[piper_voice] = (voice, config_path, sample_rate)
                    logger.info(f"Cached Piper voice: {piper_voice} (sample_rate={sample_rate}Hz)")
                    
                    # Pre-warm g2pw by doing a dummy synthesis to initialize bert-base-chinese
                    try:
                        logger.debug("Pre-warming g2pw (Chinese phonemization)...")
                        _ = list(voice.synthesize("测试"))  # Dummy Chinese text
                        logger.debug("g2pw pre-warmed successfully")
                    except Exception as warmup_error:
                        # Ignore warmup errors, just log them
                        logger.debug(f"g2pw warmup skipped: {warmup_error}")
                except Exception as voice_error:
                    error_msg = str(voice_error).lower()
                    if any(keyword in error_msg for keyword in ["connection", "network", "timeout", "unreachable", "offline", "not found", "404"]):
                        raise RuntimeError(
                            f"Piper TTS voice not found in cache and cannot download (no internet). "
                            f"Please download the voice first with internet: "
                            f"python download_tts_model.py --lang {language_lower}"
                        ) from voice_error
                    else:
                        raise voice_error
            
            # Generate speech
            try:
                # Time the synthesis
                synth_start = time.time()
                
                # Piper TTS generates audio as int16 PCM chunks
                audio_stream = voice.synthesize(text)
                # Collect all audio chunks - use bytearray for better performance
                audio_data = bytearray()
                for chunk in audio_stream:
                    audio_data.extend(chunk.audio_int16_bytes)
                audio_data = bytes(audio_data)
                
                # Convert int16 bytes to numpy array
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Ensure mono (Piper typically outputs mono, but check anyway)
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)
                
                # Apply speed adjustment if requested
                if abs(speed - 1.0) > 1e-6:
                    audio_array, sample_rate = _apply_speed_adjustment(audio_array, sample_rate, speed)
                
                # Convert to WAV bytes - use in-memory buffer instead of temp file
                wav_buffer = io.BytesIO()
                try:
                    sf.write(wav_buffer, np.clip(audio_array, -1.0, 1.0), samplerate=int(sample_rate), format='WAV')
                    audio_bytes = wav_buffer.getvalue()
                    
                    synth_time = time.time() - synth_start
                    logger.info(f"Piper TTS synthesis completed in {synth_time:.2f}s (text length: {len(text)} chars, audio: {len(audio_bytes)} bytes)")
                    
                    return audio_bytes, int(sample_rate)
                finally:
                    wav_buffer.close()
            except Exception as e:
                raise e
        except RuntimeError as e:
            # Re-raise RuntimeError (offline errors) so they're handled properly
            error_msg = str(e).lower()
            if "not found" in error_msg or "cannot download" in error_msg or "offline" in error_msg or "404" in error_msg:
                raise e
            else:
                error_msg = str(e).lower()
                logger.error(f"Piper TTS error: {e}")
                
                # Check for missing dependencies (required for Chinese)
                if "g2pw" in error_msg or "no module named 'g2pw'" in error_msg:
                    raise RuntimeError(
                        f"Piper TTS requires g2pw for Chinese phonemization. "
                        f"Install with: pip install g2pw"
                    ) from e
                elif "unicode_rbnf" in error_msg or "unicode-rbnf" in error_msg or "no module named 'unicode_rbnf'" in error_msg:
                    raise RuntimeError(
                        f"Piper TTS requires unicode-rbnf for Chinese number formatting. "
                        f"Install with: pip install unicode-rbnf"
                    ) from e
                elif "sentence_stream" in error_msg or "sentence-stream" in error_msg or "no module named 'sentence_stream'" in error_msg:
                    raise RuntimeError(
                        f"Piper TTS requires sentence-stream for Chinese sentence processing. "
                        f"Install with: pip install sentence-stream"
                    ) from e
                
                # Build dependency list based on language
                deps = "piper-tts huggingface_hub"
                if language_lower in ["zh", "cmn", "zho", "chinese", "mandarin", "zh-cn"]:
                    deps += " g2pw unicode-rbnf sentence-stream"
                
                raise RuntimeError(
                    f"Piper TTS failed for {language}. Error: {e}. "
                    f"For offline operation, ensure the voice is downloaded and dependencies are installed: "
                    f"pip install {deps}"
                ) from e
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Piper TTS failed: {e}")
            
            # Check for missing dependencies (required for Chinese)
            if "g2pw" in error_msg or "no module named 'g2pw'" in error_msg:
                raise RuntimeError(
                    f"Piper TTS requires g2pw for Chinese phonemization. "
                    f"Install with: pip install g2pw"
                ) from e
            elif "unicode_rbnf" in error_msg or "unicode-rbnf" in error_msg or "no module named 'unicode_rbnf'" in error_msg:
                raise RuntimeError(
                    f"Piper TTS requires unicode-rbnf for Chinese number formatting. "
                    f"Install with: pip install unicode-rbnf"
                ) from e
            elif "sentence_stream" in error_msg or "sentence-stream" in error_msg or "no module named 'sentence_stream'" in error_msg:
                raise RuntimeError(
                    f"Piper TTS requires sentence-stream for Chinese sentence processing. "
                    f"Install with: pip install sentence-stream"
                ) from e
            
            # Build dependency list based on language
            deps = "piper-tts huggingface_hub"
            if language_lower in ["zh", "cmn", "zho", "chinese", "mandarin", "zh-cn"]:
                deps += " g2pw unicode-rbnf sentence-stream"
            
            raise RuntimeError(
                f"Piper TTS failed for {language}. Error: {e}. "
                f"For offline operation, ensure the voice is downloaded and dependencies are installed: "
                f"pip install {deps}"
            ) from e
    
    # If we get here, language is not supported or dependencies are missing
    supported = []
    if _HAS_TORCH:
        supported.append("en/english (MMS-TTS - offline-capable)")
    if _HAS_PYKOKORO:
        supported.append("ja/japanese (PyKokoro - offline-capable)")
    if _HAS_PIPER_TTS:
        supported.append("zh/chinese (Piper TTS - offline-capable)")
    
    if not supported:
        raise RuntimeError(
            f"TTS is not available. Install dependencies:\n"
            f"  - For English (offline-capable): pip install torch transformers datasets soundfile\n"
            f"  - For Japanese (offline-capable): pip install pykokoro\n"
            f"  - For Chinese (offline-capable): pip install piper-tts huggingface_hub g2pw unicode-rbnf sentence-stream"
        )
    
    
    raise ValueError(
        f"Unsupported language: {language}. "
        f"Supported languages: {', '.join(supported) if supported else 'None (install dependencies)'}. "
        f"All TTS engines are offline-capable after initial model download."
    )
