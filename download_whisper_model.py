#!/usr/bin/env python3
"""Pre-download Whisper models for offline STT"""
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_whisper_model(model_size="base"):
    """Download Whisper model for offline use"""
    try:
        import whisper
    except ImportError:
        logger.error("Whisper not installed. Install with: pip install openai-whisper")
        return False
    
    try:
        logger.info(f"Downloading Whisper model: {model_size}")
        logger.info("This may take a few minutes and requires internet connection...")
        
        model = whisper.load_model(model_size)
        logger.info(f"✓ Whisper model '{model_size}' downloaded and cached successfully!")
        logger.info(f"   Model location: ~/.cache/whisper/")
        logger.info("   STT will now work offline!")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        if "connection" in str(e).lower() or "network" in str(e).lower():
            logger.error("Network error - please check your internet connection")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Whisper model for offline STT")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"],
                       help="Model size (default: base)")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Whisper Model Downloader for Offline STT")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Model sizes:")
    logger.info("  - tiny:   ~39MB,  fastest, lower accuracy")
    logger.info("  - base:  ~74MB,  fast, good accuracy (recommended)")
    logger.info("  - small: ~244MB, medium speed, better accuracy")
    logger.info("  - medium: ~769MB, slow, high accuracy")
    logger.info("  - large: ~1550MB, slowest, highest accuracy")
    logger.info("")
    
    success = download_whisper_model(args.model)
    
    if success:
        logger.info("")
        logger.info("You can now use STT offline!")
        sys.exit(0)
    else:
        logger.error("")
        logger.error("Failed to download model. Please check your internet connection and try again.")
        sys.exit(1)
