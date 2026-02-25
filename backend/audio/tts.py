"""Text-to-Speech module for multi-language TTS"""
import io
import logging
import re
import os
import sys
import subprocess
import inspect

# CRITICAL: Disable MPS BEFORE any torch imports (for MeloTTS on macOS)
if sys.platform == "darwin":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['TRANSFORMERS_NO_MPS'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

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
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    torch = None

try:
    from transformers import AutoProcessor, VitsModel
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False
    AutoProcessor = None
    VitsModel = None

# Try to import MeloTTS for Chinese and Japanese
_HAS_MELOTTS = False
_TTS_CLASS = None
try:
    from melo.api import TTS
    _TTS_CLASS = TTS
    _HAS_MELOTTS = True
except ImportError:
    _HAS_MELOTTS = False
    _TTS_CLASS = None

# Try to import librosa for speed/tempo adjustment
try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

# Try to import PyKokoro for English TTS (replaces MMS-TTS for English)
try:
    from pykokoro import build_pipeline
    _HAS_PYKOKORO = True
except ImportError:
    _HAS_PYKOKORO = False
    build_pipeline = None

# Try to import ONNX Runtime for provider/device detection used by PyKokoro
try:
    import onnxruntime as ort
    _HAS_ONNXRUNTIME = True
except ImportError:
    ort = None
    _HAS_ONNXRUNTIME = False


logger = logging.getLogger(__name__)

# Cache for loaded MMS-TTS models to avoid reloading on each request
_mms_model_cache = {}  # Maps (model_id, device_str) -> (model, processor)

# Cache for MeloTTS models (per language and device)
_melotts_cache = {}  # Maps (language, device) -> TTS instance

# Cache for PyKokoro TTS instances (English only)
_pykokoro_cache_en = {}  # Maps device -> English pipeline (82M model)

# Constants for TTS configuration
SPEED_MIN = 0.5
SPEED_MAX = 2.0
SPEED_DEFAULT = 1.0
CHINESE_SPEED_MULTIPLIER = 0.75  # Natural speech speed for Chinese

# Pause durations (in seconds) for natural speech
PAUSE_SENTENCE_END = (0.4, 0.5)  # After 。！？
PAUSE_COMMA = (0.2, 0.25)  # After ，、：；
PAUSE_DEFAULT = (0.25, 0.3)  # Default between segments
PAUSE_JAPANESE = 0.1  # Simple pause for Japanese

# Fade samples for natural pause transitions
PAUSE_FADE_SAMPLES = 100

# Language to MMS-TTS model mapping (offline-capable) - DEPRECATED: English now uses PyKokoro
# Keeping for potential future use or fallback
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


def _sanitize_tts_input_text(text: str) -> str:
    """
    Normalize text before synthesis.
    - Removes citation markers like [1], [2][3], 【4】 that often cause odd prosody.
    - Collapses repeated whitespace.
    """
    if not text:
        return ""

    cleaned = str(text)
    cleaned = re.sub(r'\[\d+(?:\s*,\s*\d+)*\]', '', cleaned)
    cleaned = re.sub(r'【\d+(?:\s*,\s*\d+)*】', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _remove_immediate_cjk_repetition(text: str) -> str:
    """
    Collapse immediate duplicated Chinese chunks, e.g. "中国和德国中国和德国" -> "中国和德国".
    This only targets contiguous CJK repetitions to avoid altering normal prose.
    """
    if not text:
        return ""

    result = text
    # Apply multiple passes in case there are nested/stacked repeats.
    for _ in range(3):
        updated = re.sub(r'([\u4e00-\u9fff]{2,16})\1+', r'\1', result)
        if updated == result:
            break
        result = updated
    return result


def _normalize_lang_code(language: str) -> str:
    """Normalize user-facing language labels to en/zh/ja (MeloTTS format: EN/ZH/JP)."""
    lang = (language or "").lower().strip()
    if lang in ("zh", "cmn", "zho", "chinese", "mandarin", "zh-cn"):
        return "ZH"  # MeloTTS uses 'ZH' for Chinese
    if lang in ("ja", "jpn", "japanese"):
        return "JP"  # MeloTTS uses 'JP' for Japanese
    return "EN"  # MeloTTS uses 'EN' for English


def _patch_melotts_for_cpu():
    """
    Patch PyTorch and MeloTTS to force CPU usage for BERT models on macOS.
    This prevents MPS device errors.
    """
    if sys.platform != "darwin":
        return
    
    try:
        import torch
        
        # Disable MPS availability check
        if hasattr(torch.backends, 'mps'):
            original_is_available = torch.backends.mps.is_available
            def patched_is_available():
                return False
            torch.backends.mps.is_available = staticmethod(patched_is_available)
        
        # Patch torch.device to convert MPS to CPU
        original_device_init = torch.device.__init__
        def patched_device_init(self, device):
            if isinstance(device, str) and 'mps' in device.lower():
                original_device_init(self, 'cpu')
            else:
                original_device_init(self, device)
        torch.device.__init__ = patched_device_init
        
        # Patch MeloTTS's chinese_bert module if available
        try:
            from melo.text import chinese_bert
            if hasattr(chinese_bert, 'get_bert_feature'):
                original_get_bert_feature = chinese_bert.get_bert_feature
                
                def patched_get_bert_feature(text, word2ph, model_id='bert-base-multilingual-uncased', device='cpu'):
                    """Force CPU device for BERT model"""
                    if isinstance(device, str) and 'mps' in device.lower():
                        device = 'cpu'
                    elif hasattr(device, 'type') and device.type == 'mps':
                        device = torch.device('cpu')
                    return original_get_bert_feature(text, word2ph, model_id, device)
                
                chinese_bert.get_bert_feature = patched_get_bert_feature
        except (ImportError, AttributeError):
            pass  # Will patch after MeloTTS is imported
            
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"Could not fully patch MeloTTS for CPU: {e}")


def _get_device(device_preference: str) -> str:
    """Determine execution device, preferring CUDA GPUs when available."""
    return _get_device_for_engine(device_preference=device_preference, engine="general")


def _torch_cuda_available() -> bool:
    """Return True when PyTorch can use CUDA."""
    try:
        return torch is not None and torch.cuda.is_available()
    except Exception:
        return False


def _onnx_gpu_providers() -> list[str]:
    """Return ONNX Runtime providers that can execute on GPU."""
    if not _HAS_ONNXRUNTIME or ort is None:
        return []
    try:
        providers = ort.get_available_providers()
        gpu_providers = []
        for name in ("CUDAExecutionProvider", "TensorrtExecutionProvider", "DmlExecutionProvider"):
            if name in providers:
                gpu_providers.append(name)
        return gpu_providers
    except Exception:
        return []


def _onnx_cuda_available() -> bool:
    """Return True when ONNX Runtime reports CUDA provider availability."""
    return "CUDAExecutionProvider" in _onnx_gpu_providers()


def _get_device_for_engine(device_preference: str, engine: str = "general") -> str:
    """
    Determine execution device for the given engine.
    - general: Torch-based paths (MeloTTS/MMS) rely on torch CUDA availability.
    - pykokoro: ONNX-based path relies on ONNX Runtime CUDA provider availability.
    """
    pref = (device_preference or "auto").lower().strip()
    if pref in ("gpu", "cuda"):
        pref = "cuda:0"

    torch_cuda_ok = _torch_cuda_available()
    onnx_gpu_providers = _onnx_gpu_providers()
    onnx_cuda_ok = "CUDAExecutionProvider" in onnx_gpu_providers
    onnx_gpu_ok = len(onnx_gpu_providers) > 0
    cuda_available = onnx_gpu_ok if engine == "pykokoro" else torch_cuda_ok

    if pref != "auto":
        if pref.startswith("cuda"):
            if cuda_available:
                return "cuda:0" if pref == "cuda" else pref
            if engine == "pykokoro":
                logger.warning(
                    "GPU requested for PyKokoro, but no ONNX Runtime GPU provider is available. "
                    "Install onnxruntime-gpu (or onnxruntime-directml) and restart. Falling back to CPU."
                )
            else:
                logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        if pref == "mps":
            try:
                if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass
            logger.warning("MPS requested but unavailable. Falling back to CPU.")
            return "cpu"
        return pref

    # Auto mode: prefer CUDA on Windows/Linux; prefer MPS on macOS when available.
    if sys.platform == "darwin":
        try:
            if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    if cuda_available:
        return "cuda:0"

    if engine == "pykokoro":
        logger.info(
            "PyKokoro running on CPU. ONNX Runtime GPU provider not detected. Current providers: %s",
            ort.get_available_providers() if _HAS_ONNXRUNTIME and ort is not None else ["not-installed"],
        )
    return "cpu"


def _build_pykokoro_pipeline_for_device(lang: str, target_device: str):
    """Build a PyKokoro pipeline with provider-aware GPU selection."""
    from pykokoro import PipelineConfig, GenerationConfig

    generation = GenerationConfig(lang=lang)

    pref = (target_device or "auto").lower().strip()
    provider = "auto"
    provider_options = None

    if pref.startswith("cuda"):
        provider = "cuda"
        # Allow "cuda:1" style device hints when multiple GPUs exist.
        if ":" in pref:
            try:
                provider_options = {"device_id": int(pref.split(":", 1)[1])}
            except Exception:
                provider_options = {"device_id": 0}
        else:
            provider_options = {"device_id": 0}
    elif pref == "cpu":
        provider = "cpu"
    elif pref == "mps":
        # PyKokoro uses CoreML provider naming on Apple platforms.
        provider = "coreml"

    config = PipelineConfig(
        generation=generation,
        provider=provider,
        provider_options=provider_options,
    )

    # Force early provider/session initialization so we fail fast instead of silently running CPU.
    return build_pipeline(config=config, eager=True)


def _has_mixed_scripts(text: str) -> bool:
    """
    Detect if text mixes CJK and Latin scripts (common mixed-language case).
    Now detects ANY English words in Japanese/Chinese text to ensure proper synthesis.
    """
    has_han = re.search(r'[\u4e00-\u9fff]', text) is not None
    has_kana = re.search(r'[\u3040-\u30ff]', text) is not None
    has_latin = re.search(r'[A-Za-z]', text) is not None
    
    if not has_latin:
        return False
    
    # If there's any Latin text with CJK text, it's mixed
    # This ensures English words in Japanese/Chinese are properly handled
    return (has_han and has_latin) or (has_kana and has_latin)


def _classify_token_lang(token: str, default_lang: str) -> str | None:
    """
    Assign a language class to one token; punctuation returns None.
    Only distinguishes English (Latin) from CJK. CJK languages (Chinese/Japanese) are treated as default_lang.
    """
    if re.search(r'[A-Za-z]', token):
        return "en"  # Only English (Latin) is distinguished
    # All CJK characters (Chinese, Japanese) are treated as the default language
    if re.search(r'[\u3040-\u30ff\u4e00-\u9fff]', token):
        return default_lang  # Treat all CJK as default_lang (don't split Chinese/Japanese)
    if re.search(r'\d', token):
        return default_lang
    return None


def _split_mixed_language_segments(text: str, default_lang: str = "zh") -> list[tuple[str, str]]:
    """
    Split mixed text into contiguous language segments.
    Only splits English (Latin) from CJK. All CJK text (Chinese/Japanese) is kept together as default_lang.
    Returns list of (segment_text, lang) where lang is either "en" or default_lang.
    """
    tokens = re.findall(
        r"[A-Za-z]+(?:['-][A-Za-z]+)*|[\u3040-\u30ff]+|[\u4e00-\u9fff]+|\d+|[^\s]",
        text
    )
    if not tokens:
        return []

    segments: list[tuple[str, str]] = []
    current_text = ""
    current_lang = None

    def flush_current():
        nonlocal current_text, current_lang
        if current_text.strip() and current_lang:
            segments.append((current_text.strip(), current_lang))
        current_text = ""
        current_lang = None

    for token in tokens:
        token_lang = _classify_token_lang(token, default_lang)
        if token_lang is None:
            # Keep punctuation in current segment for natural prosody.
            if current_text:
                current_text += token
            continue

        if current_lang is None:
            current_lang = token_lang
            current_text = token
            continue

        if token_lang == current_lang:
            # Same language - append to current segment
            if current_lang == "en" and re.search(r'[A-Za-z0-9]$', current_text) and re.search(r'^[A-Za-z0-9]', token):
                current_text += " " + token
            else:
                current_text += token
        else:
            # Language changed - flush current and start new segment
            # This only happens when switching between English and CJK (default_lang)
            flush_current()
            current_lang = token_lang
            current_text = token

    flush_current()
    return segments


def _get_speaker_id(model, language: str, preferred_names: list[str]) -> tuple[int, str]:
    """
    Get speaker ID from model with fallback logic.
    
    Args:
        model: MeloTTS model instance
        language: Language code for logging
        preferred_names: List of preferred speaker names to try
        
    Returns:
        Tuple of (speaker_id, speaker_name)
    """
    try:
        speaker_ids = model.hps.data.spk2id
        logger.info(f"Available speakers: {list(speaker_ids.keys())}")
        
        # Try preferred names first
        for name in preferred_names:
            if name in speaker_ids:
                speaker_id = speaker_ids[name]
                logger.info(f"Using speaker: {name} (ID: {speaker_id}) for {language}")
                return speaker_id, name
        
        # Fallback to first available speaker
        speaker_name = list(speaker_ids.keys())[0]
        speaker_id = speaker_ids[speaker_name]
        logger.info(f"Using first available speaker: {speaker_name} (ID: {speaker_id}) for {language}")
        return speaker_id, speaker_name
    except Exception as e:
        logger.error(f"Could not get speaker IDs: {e}")
        # Try direct access as fallback
        try:
            if hasattr(model, 'hps') and hasattr(model.hps, 'data') and hasattr(model.hps.data, 'spk2id'):
                speaker_ids = model.hps.data.spk2id
                logger.info(f"Fallback: Available speakers: {list(speaker_ids.keys())}")
                speaker_name = list(speaker_ids.keys())[0]
                speaker_id = speaker_ids[speaker_name]
                logger.info(f"Fallback: Using first available speaker: {speaker_name} (ID: {speaker_id})")
                return speaker_id, speaker_name
            else:
                raise ValueError("Cannot access speaker IDs from model")
        except Exception as e2:
            logger.error(f"All fallbacks failed, using speaker_id=0: {e2}")
            return 0, "default"


def _split_text_by_punctuation(text: str, punctuation_pattern: str) -> list[str]:
    """
    Split text into sentences based on punctuation pattern.
    
    Args:
        text: Input text to split
        punctuation_pattern: Regex pattern for punctuation marks
        
    Returns:
        List of sentence segments
    """
    sentences = re.split(f'([{punctuation_pattern}])', text)
    text_segments = []
    for i in range(0, len(sentences), 2):
        if i + 1 < len(sentences):
            segment = sentences[i] + sentences[i + 1]
        else:
            segment = sentences[i]
        segment = segment.strip()
        if segment:
            text_segments.append(segment)
    return text_segments if text_segments else ([text.strip()] if text.strip() else [])


def _normalize_audio(seg_audio: np.ndarray) -> np.ndarray:
    """
    Normalize audio to float32 format with proper range.
    
    Args:
        seg_audio: Audio array (may be stereo or mono, various formats)
        
    Returns:
        Normalized mono float32 audio array
    """
    # Convert to mono if stereo
    if len(seg_audio.shape) > 1:
        seg_audio = np.mean(seg_audio, axis=1)
    
    # Ensure float32
    seg_audio = seg_audio.astype(np.float32)
    
    # Minimal normalization - only if values are clearly out of range
    max_val = np.abs(seg_audio).max()
    if max_val > 1.0:
        seg_audio = seg_audio / max_val
    elif max_val > 32767:
        seg_audio = seg_audio / 32768.0
    
    return seg_audio


def _create_pause(sample_rate: int, duration: float, fade: bool = True) -> np.ndarray:
    """
    Create a pause (silence) with optional fade-in/fade-out.
    
    Args:
        sample_rate: Audio sample rate
        duration: Pause duration in seconds
        fade: Whether to add fade-in/fade-out
        
    Returns:
        Pause audio array
    """
    pause = np.zeros(int(sample_rate * duration), dtype=np.float32)
    if fade and len(pause) > 0:
        fade_samples = min(PAUSE_FADE_SAMPLES, len(pause) // 10)
        if fade_samples > 0:
            fade_curve = np.linspace(0, 1, fade_samples)
            pause[:fade_samples] *= fade_curve
            pause[-fade_samples:] *= fade_curve[::-1]
    return pause


def _calculate_pause_duration(sentence: str, idx: int, use_variable: bool = True) -> float:
    """
    Calculate pause duration based on sentence punctuation.
    
    Args:
        sentence: Current sentence text
        idx: Sentence index for variation
        use_variable: Whether to use variable pauses (Chinese) or fixed (Japanese)
        
    Returns:
        Pause duration in seconds
    """
    if not use_variable:
        return PAUSE_JAPANESE
    
    sentence_end = sentence.rstrip()
    if sentence_end.endswith(('。', '！', '？')):
        # Longer pause after sentence-ending punctuation
        min_dur, max_dur = PAUSE_SENTENCE_END
        return min_dur + (idx % 3) * ((max_dur - min_dur) / 2)
    elif sentence_end.endswith(('，', '、', '：', '；')):
        # Medium pause after commas/semicolons
        min_dur, max_dur = PAUSE_COMMA
        return min_dur + (idx % 2) * ((max_dur - min_dur) / 1)
    else:
        # Default pause between segments
        min_dur, max_dur = PAUSE_DEFAULT
        return min_dur + (idx % 2) * ((max_dur - min_dur) / 1)


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Dependency-free fallback resampler."""
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    if len(audio) == 0:
        return audio.astype(np.float32)

    src_x = np.arange(len(audio), dtype=np.float64)
    dst_len = max(1, int(round(len(audio) * float(dst_sr) / float(src_sr))))
    dst_x = np.linspace(0, len(audio) - 1, num=dst_len, dtype=np.float64)
    return np.interp(dst_x, src_x, audio.astype(np.float64)).astype(np.float32)


def _resample_if_needed(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample audio to target sample-rate with librosa when available."""
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    if _HAS_LIBROSA:
        return librosa.resample(audio.astype(np.float32), orig_sr=src_sr, target_sr=dst_sr).astype(np.float32)
    return _resample_linear(audio, src_sr, dst_sr)


def _ensure_spacy_model(model_name: str) -> bool:
    """
    Ensure a spaCy language model is available.
    Returns True if present (or successfully downloaded), False otherwise.
    """
    try:
        import spacy
    except ImportError:
        logger.warning("spaCy is not installed; cannot auto-install language model '%s'.", model_name)
        return False

    try:
        spacy.load(model_name)
        return True
    except OSError:
        logger.info("spaCy model '%s' not found. Attempting automatic download...", model_name)
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                check=True,
                capture_output=True,
                text=True,
            )
            spacy.load(model_name)
            logger.info("✓ spaCy model '%s' downloaded successfully.", model_name)
            return True
        except Exception as download_error:
            logger.warning(
                "Automatic spaCy model download failed for '%s': %s",
                model_name,
                download_error,
            )
            return False
    except Exception as check_error:
        logger.warning("Unable to verify spaCy model '%s': %s", model_name, check_error)
        return False


def _synthesize_melotts(
    text: str,
    language: str,
    speed: float,
    device_preference: str,
    speed_multiplier: float = 1.0,
    use_variable_pauses: bool = True,
    punctuation_pattern: str = "。！？；"
) -> tuple[bytes, int]:
    """
    Shared MeloTTS synthesis function for Chinese and Japanese.
    
    Args:
        text: Text to synthesize
        language: Language code ('ZH' or 'JP')
        speed: Base speed multiplier
        device_preference: Device preference for synthesis
        speed_multiplier: Additional speed adjustment (e.g., 0.75 for Chinese)
        use_variable_pauses: Whether to use variable pause durations
        punctuation_pattern: Punctuation pattern for sentence splitting
        
    Returns:
        Tuple of (audio_bytes, sample_rate)
    """
    global _melotts_cache
    
    if not _HAS_MELOTTS:
        raise RuntimeError(
            f"MeloTTS is not installed. Install with:\n"
            f"  git clone https://github.com/myshell-ai/MeloTTS.git\n"
            f"  cd MeloTTS\n"
            f"  pip install -e .\n"
            f"{'  python -m unidic download' if language == 'JP' else ''}"
        )
    
    original_text = text
    text = _sanitize_tts_input_text(text)
    if not text:
        raise ValueError("Text is empty after sanitization.")
    
    # Patch for macOS before initializing
    if sys.platform == "darwin":
        _patch_melotts_for_cpu()
    
    # Get device
    device = _get_device(device_preference)
    if sys.platform == "darwin" and device != "cpu":
        logger.warning(f"Overriding device '{device}' to 'cpu' on macOS to avoid MPS issues")
        device = "cpu"

    # MeloTTS accepts "cuda" but not all builds accept "cuda:N" style IDs.
    melotts_device = "cuda" if isinstance(device, str) and device.startswith("cuda") else device
    
    # Initialize or get cached model
    cache_key = f"{language}_{melotts_device}"
    if cache_key not in _melotts_cache:
        logger.info(f"Initializing MeloTTS model for language: {language}, device: {melotts_device}")
        try:
            model = _TTS_CLASS(language=language, device=melotts_device)
            _melotts_cache[cache_key] = model
            logger.info(f"✓ MeloTTS model initialized and cached ({language}, {melotts_device})")
        except Exception as e:
            logger.error(f"Failed to initialize MeloTTS model: {e}")
            raise RuntimeError(
                f"Failed to initialize MeloTTS for {language}. "
                f"Error: {e}\n\n"
                f"Installation:\n"
                f"  git clone https://github.com/myshell-ai/MeloTTS.git\n"
                f"  cd MeloTTS\n"
                f"  pip install -e .\n"
                f"{'  python -m unidic download' if language == 'JP' else ''}"
            ) from e
    else:
        logger.debug(f"Using cached MeloTTS model ({language}, {melotts_device})")
    
    model = _melotts_cache[cache_key]
    
    # Get speaker ID
    preferred_names = ['ZH', 'Chinese', 'CN'] if language == 'ZH' else ['JP', 'Japanese']
    speaker_id, speaker_name = _get_speaker_id(model, language, preferred_names)
    
    # Split text into sentences
    text_segments = _split_text_by_punctuation(text, punctuation_pattern)
    logger.debug(f"{language} text split into {len(text_segments)} sentence segments for processing")
    
    # Validate and adjust speed
    if speed <= 0:
        speed = SPEED_DEFAULT
    speed = max(SPEED_MIN, min(SPEED_MAX, speed)) * speed_multiplier
    
    # Process each sentence
    import tempfile
    audio_segments = []
    sample_rate = None
    
    for idx, sentence in enumerate(text_segments):
        if not sentence.strip():
            continue
        
        logger.debug(f"Processing sentence {idx + 1}/{len(text_segments)}: '{sentence[:30]}...'")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            model.tts_to_file(sentence, speaker_id, tmp_path, speed=speed)
            
            seg_audio, seg_sr = sf.read(tmp_path, dtype='float32')
            if sample_rate is None:
                sample_rate = int(seg_sr)
            
            seg_audio = _normalize_audio(seg_audio)
            audio_segments.append(seg_audio)
            
            # Add pause between sentences
            if idx < len(text_segments) - 1:
                pause_duration = _calculate_pause_duration(sentence, idx, use_variable_pauses)
                pause = _create_pause(sample_rate, pause_duration, fade=use_variable_pauses)
                audio_segments.append(pause)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    
    # Concatenate and convert to bytes
    if not audio_segments:
        raise RuntimeError("No audio segments generated")
    
    audio_array = np.concatenate(audio_segments)
    output_buffer = io.BytesIO()
    try:
        sf.write(output_buffer, np.clip(audio_array, -1.0, 1.0), samplerate=sample_rate, format='WAV')
        audio_bytes = output_buffer.getvalue()
        logger.info(f"✓ MeloTTS synthesis successful ({len(audio_bytes)} bytes, {sample_rate} Hz, {len(text_segments)} sentences)")
        if original_text != text:
            logger.info(f"{language} TTS input normalized: '{original_text[:40]}...' -> '{text[:40]}...'")
        return audio_bytes, sample_rate
    finally:
        output_buffer.close()


def _synthesize_mixed_text(text: str, speed: float, device_preference: str, default_lang: str = "zh") -> tuple[bytes, int]:
    """Synthesize mixed-language text by routing each segment to its language model."""
    # Normalize default_lang to standard format (ja, zh, en) for segment classification
    # Convert MeloTTS format (JP, ZH, EN) to standard format
    default_lang_normalized = default_lang.lower()
    if default_lang_normalized == "jp":
        default_lang_normalized = "ja"
    elif default_lang_normalized == "zh":
        default_lang_normalized = "zh"
    elif default_lang_normalized in ("en", "eng", "english"):
        default_lang_normalized = "en"
    else:
        # Try to normalize using _normalize_lang_code and convert back
        if default_lang_normalized in ("jpn", "japanese"):
            default_lang_normalized = "ja"
        elif default_lang_normalized in ("cmn", "zho", "chinese", "mandarin", "zh-cn"):
            default_lang_normalized = "zh"
    
    segments = _split_mixed_language_segments(text, default_lang=default_lang_normalized)
    if not segments:
        raise ValueError("No mixed-language segments to synthesize.")

    logger.info(f"Mixed-language routing enabled: {[(s[:24], l) for s, l in segments]}")

    stitched_audio = []
    output_sr = None
    boundary_pause = None

    for segment_text, segment_lang in segments:
        if not segment_text.strip():
            continue

        # Normalize segment language code (convert JP -> ja, ZH -> zh, etc.)
        if segment_lang.upper() == "JP":
            segment_lang = "ja"
        elif segment_lang.upper() == "ZH":
            segment_lang = "zh"
        elif segment_lang.upper() == "EN":
            segment_lang = "en"
        
        original_segment_lang = segment_lang  # Keep track of original for fallback logic
        
        # Debug logging
        logger.debug(f"Processing segment: '{segment_text[:30]}...' (lang={segment_lang}, default_lang={default_lang_normalized}, len={len(segment_text.strip())})")
        
        # Always try to use PyKokoro for English words to ensure proper pronunciation
        # Only fall back to default language TTS if PyKokoro fails
        use_default_for_short = False
        if segment_lang == "en":
            logger.info(f"English segment '{segment_text}' ({len(segment_text.strip())} chars) - will try PyKokoro first, fallback to {default_lang_normalized} if needed")

        # Try to synthesize with the determined language
        seg_bytes = None
        seg_sr = None
        synthesis_succeeded = False
        
        try:
            seg_bytes, seg_sr = synthesize_speech(
                text=segment_text,
                language=segment_lang,
                speed=speed,
                device_preference=device_preference,
                skip_mixed_detection=True  # Prevent recursive mixed-language routing
            )
            synthesis_succeeded = True
        except (RuntimeError, Exception) as e:
            error_msg = str(e).lower()
            # If English TTS fails (e.g., spaCy not installed, PyKokoro unavailable), fall back to default language TTS
            if original_segment_lang == "en" and default_lang_normalized in ("zh", "ja"):
                logger.warning(f"English TTS (PyKokoro) failed for '{segment_text}': {e}")
                logger.info(f"Falling back to {default_lang_normalized} TTS for English segment '{segment_text}'")
                try:
                    seg_bytes, seg_sr = synthesize_speech(
                        text=segment_text,
                        language=default_lang_normalized,
                        speed=speed,
                        device_preference=device_preference,
                        skip_mixed_detection=True  # Prevent recursive mixed-language routing
                    )
                    synthesis_succeeded = True
                    logger.info(f"✓ Successfully synthesized '{segment_text}' using {default_lang_normalized} TTS fallback")
                except Exception as fallback_error:
                    logger.error(f"Fallback to {default_lang_normalized} TTS also failed for '{segment_text}': {fallback_error}")
                    # Don't raise here - continue with other segments, but log the error
                    logger.error(f"Skipping segment '{segment_text}' due to synthesis failure")
                    continue  # Skip this segment and continue with others
            elif segment_lang not in ("en", "zh", "ja"):
                # Normalize unknown language codes and try again, or use default
                logger.warning(f"Unknown language code '{segment_lang}' for segment '{segment_text}', using default '{default_lang_normalized}'")
                try:
                    seg_bytes, seg_sr = synthesize_speech(
                        text=segment_text,
                        language=default_lang_normalized,
                        speed=speed,
                        device_preference=device_preference,
                        skip_mixed_detection=True
                    )
                    synthesis_succeeded = True
                except Exception as fallback_error:
                    logger.error(f"Failed to synthesize '{segment_text}' with default language '{default_lang_normalized}': {fallback_error}")
                    continue
            else:
                # For non-English segments or if already using default, re-raise
                logger.error(f"TTS synthesis failed for '{segment_text}' (lang={segment_lang}): {e}")
                raise
        
        if not synthesis_succeeded or seg_bytes is None:
            logger.warning(f"Skipping segment '{segment_text}' - synthesis did not produce audio")
            continue
        seg_audio, read_sr = sf.read(io.BytesIO(seg_bytes), dtype="float32")
        if hasattr(seg_audio, "ndim") and seg_audio.ndim > 1:
            seg_audio = np.mean(seg_audio, axis=1)
        seg_audio = seg_audio.astype(np.float32)

        if output_sr is None:
            output_sr = int(read_sr)
            boundary_pause = np.zeros(max(1, int(output_sr * 0.04)), dtype=np.float32)

        seg_audio = _resample_if_needed(seg_audio, int(read_sr), int(output_sr))
        stitched_audio.append(np.clip(seg_audio, -1.0, 1.0))
        stitched_audio.append(boundary_pause)

    if not stitched_audio:
        raise RuntimeError("Mixed-language synthesis produced no audio.")

    if len(stitched_audio) > 1:
        stitched_audio = stitched_audio[:-1]  # drop trailing pause

    merged = np.concatenate(stitched_audio).astype(np.float32)
    output_buffer = io.BytesIO()
    try:
        sf.write(output_buffer, merged, samplerate=int(output_sr), format='WAV')
        return output_buffer.getvalue(), int(output_sr)
    finally:
        output_buffer.close()

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
    skip_mixed_detection: bool = False,
) -> tuple[bytes, int]:
    """Synthesize speech from text and return audio data as bytes.

    Args:
        text: The content to speak.
        language: Language code ("en", "ja", "zh", etc.) or "auto" for auto-detection.
        speed: Playback speed multiplier (1.0 = normal, 1.2 = 20% faster, 0.9 = 10% slower).
        device_preference: Device to use ("auto", "cpu", "cuda", "cuda:0", or "mps").

    Returns:
        Tuple of (audio_bytes, sample_rate) where audio_bytes is WAV file bytes.
    """
    global _melotts_cache, _pykokoro_cache_en
    
    if not _HAS_CORE_DEPS:
        raise RuntimeError(
            "TTS requires numpy and soundfile. Install with: pip install numpy soundfile"
        )
    
    if not text:
        raise ValueError("Text must not be empty.")

    original_text = text
    text = _sanitize_tts_input_text(text)
    if not text:
        raise ValueError("Text is empty after sanitization.")

    # Auto-detect language if not specified or set to "auto"
    if language.lower().strip() in ("auto", ""):
        language = detect_language(text)
        logger.info(f"Auto-detected language: {language} for text: '{text[:50]}...'")
    
    language_lower = language.lower().strip()
    normalized_lang = _normalize_lang_code(language_lower if language_lower not in ("", "auto") else "zh")

    # In Chinese/Japanese context with embedded English, route each segment to the correct model.
    # This ensures English words are properly synthesized using PyKokoro instead of being skipped
    # Skip mixed detection if explicitly requested (to prevent recursive routing)
    if not skip_mixed_detection and _has_mixed_scripts(text) and (
        language_lower in ("auto", "", "zh", "cmn", "zho", "chinese", "mandarin", "zh-cn") or
        language_lower in ("ja", "jpn", "japanese")
    ):
        return _synthesize_mixed_text(
            text=text,
            speed=speed,
            device_preference=device_preference,
            default_lang=normalized_lang
        )
    
    # For English: Use PyKokoro-82M (replacing MMS-TTS)
    if language_lower in ["en", "eng", "english"]:
        if _HAS_PYKOKORO:
            try:
                logger.info(f"Using PyKokoro-82M for English TTS (offline-capable)")

                # PyKokoro English sentence splitting depends on this spaCy model.
                # Auto-install once if missing to avoid crashing on first use.
                _ensure_spacy_model("en_core_web_sm")
                target_device = _get_device_for_engine(device_preference, engine="pykokoro")
                
                # Check cache first - use separate cache for English
                if not isinstance(_pykokoro_cache_en, dict):
                    _pykokoro_cache_en = {}
                
                if target_device not in _pykokoro_cache_en:
                    logger.info(
                        "Initializing PyKokoro English pipeline (82M model) on device: %s",
                        target_device,
                    )
                    _pykokoro_cache_en[target_device] = _build_pykokoro_pipeline_for_device(
                        lang='en',
                        target_device=target_device,
                    )
                    logger.info(
                        "✓ PyKokoro pipeline initialized and cached (English mode, device=%s)",
                        target_device,
                    )
                else:
                    logger.debug("Using cached PyKokoro pipeline (English, device=%s)", target_device)
                
                # Synthesize with PyKokoro
                from pykokoro import GenerationConfig
                result = _pykokoro_cache_en[target_device].run(text, generation=GenerationConfig(lang='en'))
                
                # Extract audio data from AudioResult
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
                    auto_fix_hint = ""
                    if _ensure_spacy_model("en_core_web_sm"):
                        auto_fix_hint = (
                            "Automatic model installation completed. "
                            "Please retry the request.\n"
                        )
                    raise RuntimeError(
                        f"PyKokoro requires spaCy language models for English. "
                        f"{auto_fix_hint}"
                        f"Install with:\n"
                        f"  pip install spacy\n"
                        f"  python -m spacy download en_core_web_sm  # Required for English\n"
                        f"Original error: {error_msg}"
                    ) from e
                else:
                    raise RuntimeError(
                        f"PyKokoro failed for English. Error: {error_msg}. "
                        f"Install with: pip install pykokoro spacy"
                    ) from e
        else:
            raise RuntimeError(
                "English TTS requires PyKokoro. "
                "Install with: pip install pykokoro"
            )
    
    # For Chinese: Use MeloTTS (optimized shared function)
    if language_lower in ["zh", "cmn", "zho", "chinese", "mandarin", "zh-cn"]:
        try:
            return _synthesize_melotts(
                text=text,
                language='ZH',
                speed=speed,
                device_preference=device_preference,
                speed_multiplier=CHINESE_SPEED_MULTIPLIER,
                use_variable_pauses=True,
                punctuation_pattern="。！？；"
            )
        except Exception as e:
            error_msg = str(e)
            # Handle NLTK resource errors with auto-download
            if "averaged_perceptron_tagger_eng" in error_msg or ("NLTK" in error_msg and "not found" in error_msg):
                try:
                    logger.info("Attempting to automatically download NLTK resource 'averaged_perceptron_tagger_eng'...")
                    import nltk
                    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
                    logger.info("✓ NLTK resource downloaded successfully. Retrying synthesis...")
                    return _synthesize_melotts(
                        text=text,
                        language='ZH',
                        speed=speed,
                        device_preference=device_preference,
                        speed_multiplier=CHINESE_SPEED_MULTIPLIER,
                        use_variable_pauses=True,
                        punctuation_pattern="。！？；"
                    )
                except Exception as nltk_error:
                    logger.error(f"Failed to automatically download/use NLTK resource: {nltk_error}")
                    raise RuntimeError(
                        f"MeloTTS requires NLTK resources for processing mixed Chinese-English text.\n\n"
                        f"INSTALLATION:\n"
                        f"1. Install NLTK (if not already installed):\n"
                        f"   pip install nltk\n\n"
                        f"2. Download the required NLTK resource:\n"
                        f"   python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"\n\n"
                        f"3. Restart the server\n\n"
                        f"This is required when Chinese text contains English words (like 'GDP').\n"
                        f"Original error: {error_msg[:200]}"
                    ) from e
            raise
    
    # Chinese synthesis path is implemented via _synthesize_melotts() above.
    
    # For Japanese: Use MeloTTS
    if language_lower in ["ja", "jpn", "japanese"]:
        if not _HAS_MELOTTS:
            raise RuntimeError(
                "MeloTTS is not installed. Install with:\n"
                "  git clone https://github.com/myshell-ai/MeloTTS.git\n"
                "  cd MeloTTS\n"
                "  pip install -e .\n"
                "  python -m unidic download\n"
                "\nOr via pip:\n"
                "  pip install melotts"
            )
        
        # Patch for macOS before initializing
        if sys.platform == "darwin":
            _patch_melotts_for_cpu()
        
        # Get device
        device = _get_device(device_preference)
        if sys.platform == "darwin" and device != "cpu":
            logger.warning(f"Overriding device '{device}' to 'cpu' on macOS to avoid MPS issues")
            device = "cpu"
        
        # Initialize or get cached model
        cache_key = f"JP_{device}"
        
        if cache_key not in _melotts_cache:
            logger.info(f"Initializing MeloTTS model for language: JP, device: {device}")
            try:
                model = _TTS_CLASS(language='JP', device=device)
                _melotts_cache[cache_key] = model
                logger.info(f"✓ MeloTTS model initialized and cached (JP, {device})")
            except Exception as e:
                logger.error(f"Failed to initialize MeloTTS model: {e}")
                raise RuntimeError(
                    f"Failed to initialize MeloTTS for Japanese. "
                    f"Error: {e}\n\n"
                    f"Installation:\n"
                    f"  git clone https://github.com/myshell-ai/MeloTTS.git\n"
                    f"  cd MeloTTS\n"
                    f"  pip install -e .\n"
                    f"  python -m unidic download"
                ) from e
        else:
            logger.debug(f"Using cached MeloTTS model (JP, {device})")
        
        model = _melotts_cache[cache_key]
        
        # Get speaker ID (required for quality synthesis) - matching test_melotts_japanese.py
        speaker_id = None
        speaker_name = None
        try:
            speaker_ids = model.hps.data.spk2id
            logger.info(f"Available speakers: {list(speaker_ids.keys())}")
            
            # Try to find a Japanese speaker
            if 'JP' in speaker_ids:
                speaker_id = speaker_ids['JP']
                speaker_name = 'JP'
            elif 'Japanese' in speaker_ids:
                speaker_id = speaker_ids['Japanese']
                speaker_name = 'Japanese'
            else:
                # Use first available speaker
                speaker_name = list(speaker_ids.keys())[0]
                speaker_id = speaker_ids[speaker_name]
            
            logger.info(f"Using speaker: {speaker_name} (ID: {speaker_id}) for Japanese")
        except Exception as e:
            logger.error(f"Could not get speaker IDs: {e}")
            # Try to get speaker IDs from model directly as fallback
            try:
                if hasattr(model, 'hps') and hasattr(model.hps, 'data') and hasattr(model.hps.data, 'spk2id'):
                    speaker_ids = model.hps.data.spk2id
                    logger.info(f"Fallback: Available speakers: {list(speaker_ids.keys())}")
                    speaker_name = list(speaker_ids.keys())[0]
                    speaker_id = speaker_ids[speaker_name]
                    logger.info(f"Fallback: Using first available speaker: {speaker_name} (ID: {speaker_id})")
                else:
                    raise ValueError("Cannot access speaker IDs from model")
            except Exception as e2:
                logger.error(f"All fallbacks failed, using speaker_id=0: {e2}")
                speaker_id = 0  # Last resort default
                speaker_name = "default"
        
        # Synthesize speech - process sentence by sentence (matching test_melotts_japanese.py)
        try:
            # Split text into sentences for processing (preserve segmentation for long context)
            # Split by Japanese punctuation marks
            sentences = re.split(r'([。！？；])', text)
            # Recombine punctuation with previous sentence
            text_segments = []
            for i in range(0, len(sentences) - 1, 2):
                if i + 1 < len(sentences):
                    segment = sentences[i] + sentences[i + 1]
                else:
                    segment = sentences[i]
                segment = segment.strip()
                if segment:
                    text_segments.append(segment)
            if not text_segments:
                text_segments = [text.strip()] if text.strip() else []
            
            logger.debug(f"Japanese text split into {len(text_segments)} sentence segments for processing")
            
            # Ensure speed is valid (MeloTTS expects speed > 0)
            if speed <= 0:
                speed = 1.0
            speed = max(0.5, min(2.0, speed))  # Clamp between 0.5x and 2.0x
            
            # Process each sentence one by one (matching test script approach)
            import tempfile
            audio_segments = []
            sample_rate = None
            
            for idx, sentence in enumerate(text_segments):
                if not sentence.strip():
                    continue
                
                logger.debug(f"Processing sentence {idx + 1}/{len(text_segments)}: '{sentence[:30]}...'")
                
                # Use temporary file like test script does
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                try:
                    # Synthesize to file exactly like test script (matching test_melotts_japanese.py line 179-184)
                    model.tts_to_file(
                        sentence,
                        speaker_id,
                        tmp_path,
                        speed=speed
                    )
                    
                    # Read the generated audio file
                    seg_audio, seg_sr = sf.read(tmp_path, dtype='float32')
                    if sample_rate is None:
                        sample_rate = int(seg_sr)
                    
                    # Convert to mono if stereo
                    if len(seg_audio.shape) > 1:
                        seg_audio = np.mean(seg_audio, axis=1)
                    
                    # Ensure float32
                    seg_audio = seg_audio.astype(np.float32)
                    
                    # Minimal normalization - only if values are clearly out of range
                    max_val = np.abs(seg_audio).max()
                    if max_val > 1.0:
                        seg_audio = seg_audio / max_val
                    elif max_val > 32767:
                        seg_audio = seg_audio / 32768.0
                    
                    audio_segments.append(seg_audio)
                    
                    # Add small pause between sentences (matching natural speech)
                    if idx < len(text_segments) - 1:
                        pause = np.zeros(int(sample_rate * 0.1), dtype=np.float32)
                        audio_segments.append(pause)
                        
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
            
            # Concatenate all sentence audio segments
            if audio_segments:
                audio_array = np.concatenate(audio_segments)
            else:
                raise RuntimeError("No audio segments generated")
            
            # Convert to bytes (WAV format)
            output_buffer = io.BytesIO()
            try:
                sf.write(output_buffer, np.clip(audio_array, -1.0, 1.0), samplerate=sample_rate, format='WAV')
                audio_bytes = output_buffer.getvalue()
                logger.info(f"✓ MeloTTS synthesis successful ({len(audio_bytes)} bytes, {sample_rate} Hz, {len(text_segments)} sentences)")
                if original_text != text:
                    logger.info(f"Japanese TTS input normalized: '{original_text[:40]}...' -> '{text[:40]}...'")
                return audio_bytes, sample_rate
            finally:
                output_buffer.close()
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"MeloTTS synthesis error: {e}", exc_info=True)
            
            # Check if it's an NLTK resource error (for mixed Japanese-English text)
            if "averaged_perceptron_tagger_eng" in error_msg or "NLTK" in error_msg or "nltk" in error_msg.lower():
                raise RuntimeError(
                    f"MeloTTS requires NLTK resources for processing mixed Japanese-English text.\n\n"
                    f"INSTALLATION:\n"
                    f"1. Install NLTK (if not already installed):\n"
                    f"   pip install nltk\n\n"
                    f"2. Download the required NLTK resource:\n"
                    f"   python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"\n\n"
                    f"   OR download all NLTK data:\n"
                    f"   python -c \"import nltk; nltk.download('all')\"\n\n"
                    f"3. Restart the server\n\n"
                    f"This is required when Japanese text contains English words.\n"
                    f"Original error: {error_msg[:200]}"
                ) from e
            
            raise RuntimeError(
                f"MeloTTS synthesis failed. Error: {error_msg[:500]}\n\n"
                f"Installation:\n"
                f"  git clone https://github.com/myshell-ai/MeloTTS.git\n"
                f"  cd MeloTTS\n"
                f"  pip install -e .\n"
                f"  python -m unidic download"
            ) from e
    
    # MMS-TTS is deprecated for English (now using PyKokoro-82M)
    # Keeping MMS-TTS code below as fallback for other languages if needed
    model_id = LANGUAGE_MODEL_MAP.get(language_lower)
    
    # If MMS-TTS model exists for this language, use it (fallback only, English should use PyKokoro above)
    if model_id is not None and _HAS_TORCH and _HAS_TRANSFORMERS:
        try:
            logger.info(f"Using MMS-TTS model for language: {language} (offline-capable)")
            
            # Set up device with unified selection logic (auto prefers CUDA when available)
            selected_device = _get_device(device_preference)
            try:
                device = torch.device(selected_device)
            except Exception:
                logger.warning("Invalid device '%s' for MMS-TTS. Falling back to CPU.", selected_device)
                device = torch.device("cpu")
            
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
    
    # If we get here, language is not supported or dependencies are missing
    supported = []
    if _HAS_PYKOKORO:
        supported.append("en/english (PyKokoro-82M - offline-capable)")
    if _HAS_MELOTTS:
        supported.append("zh/chinese (MeloTTS - offline-capable)")
        supported.append("ja/japanese (MeloTTS - offline-capable)")
    elif _HAS_TORCH and _HAS_TRANSFORMERS:
        # Fallback to MMS-TTS if PyKokoro not available
        supported.append("en/english (MMS-TTS - offline-capable, deprecated)")
    
    if not supported:
        raise RuntimeError(
            f"TTS is not available. Install dependencies:\n"
            f"  - For English (offline-capable): pip install pykokoro spacy\n"
            f"    Then: python -m spacy download en_core_web_sm\n"
            f"  - For Chinese/Japanese (offline-capable):\n"
            f"    git clone https://github.com/myshell-ai/MeloTTS.git\n"
            f"    cd MeloTTS\n"
            f"    pip install -e .\n"
            f"    python -m unidic download  # For Japanese only"
        )
    
    raise ValueError(
        f"Unsupported language: {language}. "
        f"Supported languages: {', '.join(supported) if supported else 'None (install dependencies)'}. "
        f"All TTS engines are offline-capable after initial model download."
    )
