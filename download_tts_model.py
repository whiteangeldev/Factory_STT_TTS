#!/usr/bin/env python3
"""Pre-download TTS models for offline use"""
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_mms_tts_model():
    """Download MMS-TTS English model for offline use"""
    try:
        import torch
        from transformers import VitsModel, AutoProcessor
    except ImportError:
        logger.error("PyTorch and transformers not installed. Install with: pip install torch transformers")
        return False
    
    model_id = "facebook/mms-tts-eng"
    
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
        logger.info("✅ MMS-TTS model downloaded and cached successfully!")
        logger.info(f"   Model location: ~/.cache/huggingface/hub/models--facebook--mms-tts-eng")
        logger.info("   English TTS will now work offline!")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        if "connection" in str(e).lower() or "network" in str(e).lower():
            logger.error("Network error - please check your internet connection")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("MMS-TTS Model Downloader for Offline TTS")
    logger.info("=" * 60)
    logger.info("")
    
    success = download_mms_tts_model()
    
    if success:
        logger.info("")
        logger.info("You can now use English TTS offline!")
        sys.exit(0)
    else:
        logger.error("")
        logger.error("Failed to download model. Please check your internet connection and try again.")
        sys.exit(1)
