#!/usr/bin/env python3
"""Pre-download TTS models for offline use"""
import sys
import logging
import argparse
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Piper TTS voice mapping
PIPER_TTS_VOICES = {
    # Chinese voices
    "zh": "zh_CN/xiaoyan/medium",
    "cmn": "zh_CN/xiaoyan/medium",
    "zho": "zh_CN/xiaoyan/medium",
    "chinese": "zh_CN/xiaoyan/medium",
    "mandarin": "zh_CN/xiaoyan/medium",
    "zh-cn": "zh_CN/xiaoyan/medium",
}

def download_mms_tts_model(model_id="facebook/mms-tts-eng", language_name="English"):
    """Download MMS-TTS model for offline use"""
    try:
        import torch
        from transformers import VitsModel, AutoProcessor
    except ImportError:
        logger.error("PyTorch and transformers not installed. Install with: pip install torch transformers")
        return False
    
    try:
        logger.info(f"Downloading MMS-TTS model: {model_id}")
        logger.info("This may take a few minutes and requires internet connection...")
        
        # Download model (will be cached for offline use)
        logger.info("Downloading model...")
        model = VitsModel.from_pretrained(model_id)
        logger.info("✓ Model downloaded successfully")
        
        logger.info("Downloading processor...")
        processor = AutoProcessor.from_pretrained(model_id)
        logger.info("✓ Processor downloaded successfully")
        
        logger.info("")
        logger.info(f"✅ MMS-TTS model downloaded and cached successfully!")
        logger.info(f"   Model location: ~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}")
        logger.info(f"   {language_name} TTS will now work offline!")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        if "connection" in str(e).lower() or "network" in str(e).lower():
            logger.error("Network error - please check your internet connection")
        return False

def download_pykokoro_japanese():
    """Download PyKokoro Japanese model for offline use"""
    try:
        from pykokoro import build_pipeline
    except ImportError:
        logger.error("PyKokoro not installed. Install with: pip install pykokoro")
        return False
    
    # Check and download spaCy models if needed
    try:
        import spacy
        logger.info("Checking spaCy language models...")
        
        # Check for English model
        try:
            spacy.load("en_core_web_sm")
            logger.info("✓ English spaCy model (en_core_web_sm) found")
        except OSError:
            logger.info("Downloading English spaCy model (en_core_web_sm)...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            logger.info("✓ English spaCy model downloaded")
        
        # Check for Japanese model (optional but recommended)
        try:
            spacy.load("ja_core_news_sm")
            logger.info("✓ Japanese spaCy model (ja_core_news_sm) found")
        except OSError:
            logger.info("Downloading Japanese spaCy model (ja_core_news_sm)...")
            import subprocess
            try:
                subprocess.run(["python", "-m", "spacy", "download", "ja_core_news_sm"], check=True)
                logger.info("✓ Japanese spaCy model downloaded")
            except subprocess.CalledProcessError:
                logger.warning("Japanese spaCy model download failed (optional, continuing anyway)")
        
    except ImportError:
        logger.error("spaCy not installed. Install with: pip install spacy")
        logger.info("Then download models: python -m spacy download en_core_web_sm")
        return False
    except Exception as e:
        logger.warning(f"spaCy model check failed: {e}, continuing anyway...")
    
    try:
        logger.info("Initializing PyKokoro pipeline (will download models if needed)...")
        logger.info("This may take a few minutes and requires internet connection...")
        
        # Initialize PyKokoro pipeline (will download models automatically)
        pipeline = build_pipeline()
        
        # Test with a simple Japanese text to trigger model download
        try:
            test_result = pipeline.run("こんにちは")
            logger.info(f"✓ Test synthesis successful (sample rate: {test_result.sample_rate} Hz)")
        except Exception as test_error:
            logger.warning(f"Test synthesis had issues: {test_error}")
            logger.info("Pipeline initialized, but test failed (may need additional setup)")
        
        logger.info("")
        logger.info("✅ PyKokoro Japanese model downloaded and cached successfully!")
        logger.info("   Japanese TTS will now work offline!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize PyKokoro: {e}")
        if "connection" in str(e).lower() or "network" in str(e).lower():
            logger.error("Network error - please check your internet connection")
        elif "spacy" in str(e).lower():
            logger.error("spaCy model issue. Try:")
            logger.error("  pip install spacy")
            logger.error("  python -m spacy download en_core_web_sm")
            logger.error("  python -m spacy download ja_core_news_sm")
        return False

def download_piper_tts_voice(lang_code: str):
    """Download Piper TTS voice for a specific language"""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return False
    
    voice_name = PIPER_TTS_VOICES.get(lang_code.lower())
    if not voice_name:
        logger.error(f"Unsupported language code for Piper TTS: {lang_code}")
        logger.info("Supported languages: zh (Chinese), ja (Japanese), or their variants")
        return False
    
    # Extract voice path components
    parts = voice_name.split("/")
    if len(parts) != 3:
        logger.error(f"Invalid voice name format: {voice_name}")
        return False
    
    lang_code_short = parts[0].split("_")[0]  # "zh"
    voice_path = "/".join(parts[:2])  # "zh_CN/xiaoyan"
    quality = parts[2]  # "medium"
    
    # Model filename format: zh_CN-xiaoyan-medium.onnx
    model_filename = f"{parts[0]}-{parts[1]}-{quality}.onnx"
    config_filename = f"{parts[0]}-{parts[1]}-{quality}.onnx.json"
    
    # Cache directory
    cache_dir = Path.home() / ".local" / "share" / "piper" / "voices"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Voice directory
    voice_dir = cache_dir / voice_name
    voice_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = voice_dir / model_filename
    config_path = voice_dir / config_filename
    
    # Check if already downloaded
    if model_path.exists() and config_path.exists():
        logger.info(f"Piper TTS voice already downloaded: {voice_name}")
        return True
    
    try:
        logger.info(f"Downloading Piper TTS voice: {voice_name}")
        logger.info("This may take a few minutes and requires internet connection...")
        
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
        
        # Download to temporary location first
        temp_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"temp_piper_{voice_name.replace('/', '_')}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info("Downloading model file...")
            downloaded_model = hf_hub_download(
                repo_id=repo_id,
                filename=model_file_path,
                cache_dir=str(temp_dir),
                local_files_only=False
            )
            logger.info("✓ Model file downloaded")
            
            logger.info("Downloading config file...")
            downloaded_config = hf_hub_download(
                repo_id=repo_id,
                filename=config_file_path,
                cache_dir=str(temp_dir),
                local_files_only=False
            )
            logger.info("✓ Config file downloaded")
            
            # Copy to target location
            shutil.copy2(downloaded_model, model_path)
            shutil.copy2(downloaded_config, config_path)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            logger.info("")
            logger.info(f"✅ Piper TTS voice downloaded and cached successfully!")
            logger.info(f"   Voice location: {voice_dir}")
            lang_display = "Chinese" if lang_code.lower() in ["zh", "cmn", "zho", "chinese", "mandarin", "zh-cn"] else "Japanese"
            logger.info(f"   {lang_display} TTS will now work offline!")
            
            return True
        except Exception as download_error:
            # Clean up temp directory on error
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise download_error
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ["connection", "network", "timeout", "unreachable", "offline", "not found", "404"]):
            logger.error(f"Piper TTS voice not found or cannot download (no internet).")
            logger.error(f"Please check your internet connection and try again.")
        else:
            logger.error(f"Failed to download Piper TTS voice: {e}")
        return False

def download_all_models():
    """Download all available TTS models for offline use"""
    logger.info("Downloading all TTS models (English, Chinese, Japanese)...")
    logger.info("")
    success_count = 0
    total = 3
    
    # Download English MMS-TTS
    logger.info("=" * 60)
    if download_mms_tts_model():
        success_count += 1
    logger.info("")
    
    # Download Chinese Piper TTS
    logger.info("=" * 60)
    if download_piper_tts_voice("zh"):
        success_count += 1
    logger.info("")
    
    # Download Japanese PyKokoro model
    logger.info("=" * 60)
    if download_pykokoro_japanese():
        success_count += 1
    logger.info("")
    
    logger.info("=" * 60)
    if success_count == total:
        logger.info(f"✅ All {total} TTS models downloaded successfully!")
        logger.info("English, Chinese, and Japanese TTS will now work offline!")
    else:
        logger.warning(f"⚠️  {success_count}/{total} models downloaded successfully")
        logger.warning("Some languages may not work offline")
    
    return success_count == total

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TTS models for offline use")
    parser.add_argument("--lang", type=str, help="Language code (en, zh, ja, or 'all' for all languages)", default="all")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("TTS Model Downloader for Offline TTS")
    logger.info("=" * 60)
    logger.info("")
    
    if args.lang.lower() == "all":
        success = download_all_models()
    elif args.lang.lower() in ["en", "eng", "english"]:
        success = download_mms_tts_model()
    elif args.lang.lower() in ["zh", "cmn", "zho", "chinese", "mandarin", "zh-cn"]:
        success = download_piper_tts_voice(args.lang.lower())
    elif args.lang.lower() in ["ja", "jpn", "japanese"]:
        success = download_pykokoro_japanese()
    else:
        logger.error(f"Unsupported language: {args.lang}")
        logger.info("Supported languages: en (English), zh (Chinese), ja (Japanese), or 'all'")
        sys.exit(1)
    
    if success:
        logger.info("")
        logger.info("You can now use TTS offline!")
        sys.exit(0)
    else:
        logger.error("")
        logger.error("Failed to download model. Please check your internet connection and try again.")
        sys.exit(1)
