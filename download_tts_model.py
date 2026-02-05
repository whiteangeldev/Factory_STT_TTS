#!/usr/bin/env python3
"""Pre-download TTS models for offline use"""
import sys
import logging
import argparse
from pathlib import Path
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Piper TTS removed - Chinese now uses PyKokoro (faster, no g2pw overhead)

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

def _download_spacy_models():
    """Download required spaCy models for PyKokoro"""
    try:
        import spacy
        logger.info("Checking spaCy language models...")
        
        # Check for English model (required)
        try:
            spacy.load("en_core_web_sm")
            logger.info("✓ English spaCy model (en_core_web_sm) found")
        except OSError:
            logger.info("Downloading English spaCy model (en_core_web_sm)...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            logger.info("✓ English spaCy model downloaded")
        
        return True
    except ImportError:
        logger.error("spaCy not installed. Install with: pip install spacy")
        return False
    except Exception as e:
        logger.warning(f"spaCy model check failed: {e}, continuing anyway...")
        return True

def download_pykokoro_japanese():
    """Download PyKokoro Japanese model for offline use"""
    try:
        from pykokoro import build_pipeline, PipelineConfig, GenerationConfig
    except ImportError:
        logger.error("PyKokoro not installed. Install with: pip install pykokoro")
        return False
    
    # Download spaCy models
    if not _download_spacy_models():
        return False
    
    # Check for Japanese model (optional but recommended)
    try:
        import spacy
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
    except Exception:
        pass
    
    try:
        logger.info("Initializing PyKokoro pipeline for Japanese (will download models if needed)...")
        logger.info("This may take a few minutes and requires internet connection...")
        
        # Initialize PyKokoro pipeline with Japanese config
        config = PipelineConfig(generation=GenerationConfig(lang='ja'))
        pipeline = build_pipeline(config=config)
        
        # Test with a simple Japanese text to trigger model download
        try:
            test_result = pipeline.run("こんにちは", generation=GenerationConfig(lang='ja'))
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

def download_pykokoro_chinese():
    """Download PyKokoro Chinese model for offline use"""
    try:
        from pykokoro import build_pipeline, PipelineConfig, GenerationConfig
    except ImportError:
        logger.error("PyKokoro not installed. Install with: pip install pykokoro")
        return False
    
    # Check for cn2an (required for Chinese number conversion)
    try:
        import cn2an
        logger.info("✓ cn2an module found (required for Chinese number conversion)")
    except ImportError:
        logger.error("cn2an not installed. Install with: pip install cn2an")
        logger.error("cn2an is required for Chinese TTS number conversion")
        return False
    
    # Check for jieba (required for Chinese word segmentation)
    try:
        import jieba
        logger.info("✓ jieba module found (required for Chinese word segmentation)")
    except ImportError:
        logger.error("jieba not installed. Install with: pip install jieba")
        logger.error("jieba is required for Chinese TTS word segmentation")
        return False
    
    # Download spaCy models
    if not _download_spacy_models():
        return False
    
    # Check for Chinese model (required for Chinese TTS)
    try:
        import spacy
        try:
            spacy.load("zh_core_web_sm")
            logger.info("✓ Chinese spaCy model (zh_core_web_sm) found")
        except OSError:
            logger.info("Downloading Chinese spaCy model (zh_core_web_sm)...")
            import subprocess
            try:
                subprocess.run(["python", "-m", "spacy", "download", "zh_core_web_sm"], check=True)
                logger.info("✓ Chinese spaCy model downloaded")
            except subprocess.CalledProcessError:
                logger.error("Chinese spaCy model download failed (required for Chinese TTS)")
                return False
    except Exception as e:
        logger.warning(f"spaCy model check failed: {e}, continuing anyway...")
    
    try:
        logger.info("Initializing PyKokoro pipeline for Chinese (will download models if needed)...")
        logger.info("This may take a few minutes and requires internet connection...")
        
        # Initialize PyKokoro pipeline with Chinese config
        config = PipelineConfig(generation=GenerationConfig(lang='zh'))
        pipeline = build_pipeline(config=config)
        
        # Test with a simple Chinese text to trigger model download
        try:
            test_result = pipeline.run("你好", generation=GenerationConfig(lang='zh'))
            logger.info(f"✓ Test synthesis successful (sample rate: {test_result.sample_rate} Hz)")
        except Exception as test_error:
            logger.warning(f"Test synthesis had issues: {test_error}")
            logger.info("Pipeline initialized, but test failed (may need additional setup)")
        
        logger.info("")
        logger.info("✅ PyKokoro Chinese model downloaded and cached successfully!")
        logger.info("   Chinese TTS will now work offline (faster than Piper)!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize PyKokoro: {e}")
        if "connection" in str(e).lower() or "network" in str(e).lower():
            logger.error("Network error - please check your internet connection")
        elif "spacy" in str(e).lower() or "zh_core_web_sm" in str(e).lower():
            logger.error("spaCy model issue. Try:")
            logger.error("  pip install spacy")
            logger.error("  python -m spacy download en_core_web_sm  # Required")
            logger.error("  python -m spacy download zh_core_web_sm  # Required for Chinese")
        elif "cn2an" in str(e).lower():
            logger.error("cn2an module missing. Install with:")
            logger.error("  pip install cn2an  # Required for Chinese number conversion")
        elif "jieba" in str(e).lower():
            logger.error("jieba module missing. Install with:")
            logger.error("  pip install jieba  # Required for Chinese word segmentation")
        return False

# Piper TTS download function removed - Chinese now uses PyKokoro

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
    
    # Download Chinese PyKokoro model
    logger.info("=" * 60)
    if download_pykokoro_chinese():
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
        success = download_pykokoro_chinese()
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
