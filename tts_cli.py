#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from io import BytesIO

import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Processor,
    VitsModel,
)

# Try to import gTTS for Japanese and Chinese support
try:
    from gtts import gTTS
    import tempfile
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

# Language to MMS-TTS model mapping (only for English - other languages use gTTS)
LANGUAGE_MODEL_MAP = {
    "en": "facebook/mms-tts-eng",
    "eng": "facebook/mms-tts-eng",
    "english": "facebook/mms-tts-eng",
}

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
    speaker_id: int = 7306,
    output_path: Path | str = "output.wav",
    do_sample: bool = True,
    seed: int | None = None,
    backend: str = "vits",
    device_preference: str = "auto",
    language: str = "en",
    speed: float = 1.0,
):
    """Synthesize speech from text and save to a WAV file.

    Args:
        text: The content to speak.
        speaker_id: Index into cmu-arctic-xvectors (used only for SpeechT5 backend).
        output_path: Where to save the generated WAV file.
        do_sample: Whether to sample for more natural variety.
        seed: Random seed for reproducibility when sampling.
        backend: "vits" (default, MMS-TTS) or "speecht5".
        device_preference: Device to use ("auto", "cpu", or "mps").
        language: Language code ("en", "ja", "zh", etc.) or full name ("english", "japanese", "chinese").
        speed: Playback speed multiplier (1.0 = normal, 1.2 = 20% faster, 0.9 = 10% slower).
    """

    if not text:
        raise ValueError("Text must not be empty.")

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

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

    if backend.lower() == "vits":
        # MMS-TTS VITS model (supports multiple languages)
        language_lower = language.lower().strip()
        model_id = LANGUAGE_MODEL_MAP.get(language_lower)
        
        # If MMS-TTS model exists for this language, use it
        if model_id is not None:
            try:
                model = VitsModel.from_pretrained(model_id).to(device)
                processor = AutoProcessor.from_pretrained(model_id)
                inputs = processor(text=text, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                try:
                    with torch.no_grad():
                        outputs = model(**inputs)
                except RuntimeError as e:
                    if str(device) == "mps":
                        # Fallback to CPU if an op is unsupported on MPS
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
                
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(output_path), np.clip(speech, -1.0, 1.0), samplerate=sampling_rate)
                return
            except Exception as e:
                # If model loading fails, fall through to gTTS
                if "not a valid model identifier" in str(e) or "does not exist" in str(e).lower():
                    pass  # Fall through to gTTS
                else:
                    raise e
        
        # Fallback to gTTS for Japanese and Chinese (and other unsupported languages)
        gtts_lang = GTTS_LANGUAGE_MAP.get(language_lower)
        if gtts_lang and _HAS_GTTS:
            # Use gTTS for Japanese and Chinese
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
                tmp_mp3_path = tmp_mp3.name
            
            try:
                tts = gTTS(text=text, lang=gtts_lang, slow=False)
                tts.save(tmp_mp3_path)
                
                # Convert MP3 to WAV using ffmpeg or pydub if available
                # Try using pydub first, then ffmpeg
                try:
                    from pydub import AudioSegment
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
                    
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    sf.write(str(output_path), np.clip(samples, -1.0, 1.0), samplerate=int(sr))
                    Path(tmp_mp3_path).unlink()  # Clean up
                    return
                except ImportError:
                    # Fallback to ffmpeg if pydub not available
                    try:
                        # First convert to WAV
                        tmp_wav_path = str(output_path).replace(".wav", "_tmp.wav")
                        subprocess.run(
                            ["ffmpeg", "-i", tmp_mp3_path, "-y", tmp_wav_path],
                            check=True,
                            capture_output=True,
                        )
                        Path(tmp_mp3_path).unlink()
                        
                        # Load and apply speed adjustment
                        if abs(speed - 1.0) > 1e-6:
                            wav_data, sr = sf.read(tmp_wav_path)
                            if len(wav_data.shape) > 1:
                                wav_data = wav_data.mean(axis=1)  # Convert stereo to mono
                            wav_data, sr = _apply_speed_adjustment(wav_data, sr, speed)
                            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                            sf.write(str(output_path), np.clip(wav_data, -1.0, 1.0), samplerate=int(sr))
                            Path(tmp_wav_path).unlink()
                        else:
                            Path(tmp_wav_path).rename(output_path)
                        return
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        # If both fail, try to read MP3 directly with soundfile (may not work)
                        raise ValueError(
                            "gTTS requires either 'pydub' or 'ffmpeg' to convert MP3 to WAV. "
                            "Install with: pip install pydub"
                        )
            except Exception as e:
                if Path(tmp_mp3_path).exists():
                    Path(tmp_mp3_path).unlink()
                raise e
        
        # If we get here, language is not supported
        supported = ["en/english"]
        if _HAS_GTTS:
            supported.extend(["ja/japanese", "zh/chinese"])
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported languages: {', '.join(supported)}. "
            f"For Japanese and Chinese, install gTTS: pip install gtts pydub"
        )

    # Fallback: SpeechT5 backend (requires speaker embeddings dataset)
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    inputs = processor(text=text, return_tensors="pt").to(device)

    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors", split="validation"
    )
    if speaker_id < 0 or speaker_id >= len(embeddings_dataset):
        raise ValueError(
            f"speaker_id must be between 0 and {len(embeddings_dataset)-1}"
        )
    speaker_embeddings = (
        torch.tensor(embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        try:
            speech_tensor = model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings,
                vocoder=vocoder,
                do_sample=do_sample,
            )
        except RuntimeError as e:
            if str(device) == "mps":
                model = model.to("cpu")
                vocoder = vocoder.to("cpu")
                inputs = inputs.to("cpu")
                speaker_embeddings = speaker_embeddings.to("cpu")
                speech_tensor = model.generate_speech(
                    inputs["input_ids"],
                    speaker_embeddings,
                    vocoder=vocoder,
                    do_sample=do_sample,
                )
            else:
                raise e

    waveform_np = speech_tensor.cpu().numpy().astype(np.float32)
    
    # Apply speed adjustment if requested
    if abs(speed - 1.0) > 1e-6:
        waveform_np, _ = _apply_speed_adjustment(waveform_np, 16000, speed)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), np.clip(waveform_np, -1.0, 1.0), samplerate=16000)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Synthesize speech from text using Hugging Face SpeechT5",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="The text to synthesize. If omitted, reads from stdin or uses a default.",
    )
    parser.add_argument(
        "--output",
        default="output.wav",
        help="Path to output WAV file.",
    )
    parser.add_argument(
        "--backend",
        choices=["vits", "speecht5"],
        default="vits",
        help="TTS backend to use. 'vits' requires no external speaker embeddings.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps"],
        default="auto",
        help="Device to run on. 'auto' prefers MPS if available, else CPU.",
    )
    parser.add_argument(
        "--speaker-id",
        type=int,
        default=7306,
        help="Index for speaker embedding from cmu-arctic-xvectors validation split.",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling for deterministic output (may sound more robotic).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility when sampling.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code or name. English: 'en'/'english' (uses MMS-TTS). "
        "Japanese: 'ja'/'japanese' (uses gTTS, requires: pip install gtts pydub). "
        "Chinese: 'zh'/'chinese' (uses gTTS, requires: pip install gtts pydub). Default: 'en'.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = normal, 1.2 = 20%% faster, 0.9 = 10%% slower). "
        "Requires librosa: pip install librosa. Default: 1.0",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        # Allow stdin or default text when --text is omitted
        provided_text = args.text
        if provided_text is None:
            if not sys.stdin.isatty():
                provided_text = sys.stdin.read().strip()
            else:
                provided_text = input("Enter text to synthesize: ").strip()
                if not provided_text:
                    raise ValueError("No text provided (empty input).")

        synthesize_speech(
            text=provided_text,
            speaker_id=args.speaker_id,
            output_path=Path(args.output),
            do_sample=not args.no_sample,
            seed=args.seed,
            backend=args.backend,
            device_preference=args.device,
            language=args.language,
            speed=args.speed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved synthesized audio to: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
