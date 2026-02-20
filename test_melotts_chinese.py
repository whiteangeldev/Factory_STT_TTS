#!/usr/bin/env python3
"""
Test script for Chinese MeloTTS (MyShell AI)

This script tests the MeloTTS Chinese TTS system with various sample texts.
It checks for installation, initializes the model, and generates audio files.

Installation:
    git clone https://github.com/myshell-ai/MeloTTS.git
    cd MeloTTS
    pip install -e .
"""

import os
import sys

# CRITICAL: Disable MPS BEFORE any torch imports
# This must happen before MeloTTS or transformers imports torch
if sys.platform == "darwin":
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['TRANSFORMERS_NO_MPS'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    # Try to disable MPS at the torch level if available
    try:
        import torch
        if hasattr(torch.backends, 'mps'):
            # Monkey patch is_available to always return False
            torch.backends.mps.is_available = lambda: False
    except ImportError:
        pass  # torch not installed yet, will patch later

import argparse
from pathlib import Path

def check_melotts_installation():
    """Check if MeloTTS is installed and available."""
    try:
        from melo.api import TTS
        return True, TTS
    except ImportError as e:
        return False, None

def check_nltk_resources():
    """Check if required NLTK resources are available for mixed Chinese-English text."""
    try:
        import nltk
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger_eng')
            return True
        except LookupError:
            return False
    except ImportError:
        return None  # NLTK not installed

def patch_melotts_for_cpu():
    """
    Patch PyTorch and MeloTTS to force CPU usage for BERT models.
    This prevents MPS device errors on macOS.
    """
    try:
        import torch
        
        # Most important: Disable MPS availability check
        if hasattr(torch.backends, 'mps'):
            original_is_available = torch.backends.mps.is_available
            def patched_is_available():
                return False
            torch.backends.mps.is_available = staticmethod(patched_is_available)
            print("✓ Disabled torch.backends.mps.is_available")
        
        # Patch torch.device to convert MPS to CPU
        original_device_init = torch.device.__init__
        def patched_device_init(self, device):
            if isinstance(device, str) and 'mps' in device.lower():
                original_device_init(self, 'cpu')
            else:
                original_device_init(self, device)
        
        # Store original and patch
        torch.device.__init__ = patched_device_init
        print("✓ Patched torch.device to convert MPS to CPU")
        
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
                print("✓ Patched MeloTTS chinese_bert.get_bert_feature")
        except (ImportError, AttributeError):
            pass  # Will patch after MeloTTS is imported
            
    except ImportError:
        print("⚠ torch not available for patching")
    except Exception as e:
        print(f"⚠ Warning: Could not fully patch: {e}")

def check_and_setup_device(device_preference: str):
    """
    Determine the best device to use, avoiding MPS on macOS due to BERT model issues.
    
    Returns:
        str: Device string to use ('cpu', 'cuda:0', etc.)
    """
    if device_preference != "auto":
        return device_preference
    
    # On macOS, force CPU to avoid MPS issues with BERT models
    if sys.platform == "darwin":
        print("⚠ macOS detected: Forcing CPU to avoid MPS device issues with BERT models")
        print("   (MPS has known issues with BERT embeddings in MeloTTS)")
        # Patch MeloTTS before initialization
        patch_melotts_for_cpu()
        return "cpu"
    
    # For other platforms, check for CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
        else:
            return "cpu"
    except ImportError:
        return "cpu"

def test_chinese_melotts(
    text: str = None,
    output_dir: str = "test_melotts_output",
    speed: float = 0.75,  # Default to slower speed for quality checking
    device: str = "auto",
    speaker_id: str = None
):
    """
    Test Chinese MeloTTS with given text.
    
    Args:
        text: Chinese text to synthesize. If None, uses default test texts.
        output_dir: Directory to save output audio files.
        speed: Speech speed (1.0 = normal, >1.0 = faster, <1.0 = slower).
        device: Device to use ('auto', 'cpu', 'cuda:0', etc.).
        speaker_id: Specific speaker ID to use. If None, uses default Chinese speaker.
    """
    # Apply patches BEFORE importing/initializing MeloTTS to prevent MPS usage
    if sys.platform == "darwin":
        patch_melotts_for_cpu()
    
    # Check installation
    is_installed, TTS = check_melotts_installation()
    if not is_installed:
        print("❌ MeloTTS is not installed!")
        print("\nInstallation instructions:")
        print("1. Clone the repository:")
        print("   git clone https://github.com/myshell-ai/MeloTTS.git")
        print("   cd MeloTTS")
        print("2. Install the package:")
        print("   pip install -e .")
        print("\nAlternatively, install via pip:")
        print("   pip install melotts")
        return False
    
    print("✓ MeloTTS is installed")
    
    # Check NLTK resources for mixed Chinese-English text
    nltk_status = check_nltk_resources()
    if nltk_status is False:
        print("\n⚠ Warning: NLTK resources not found. Mixed Chinese-English text may fail.")
        print("   To fix, run:")
        print("   python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"")
        print("   Or: python -c \"import nltk; nltk.download('all')\"")
    elif nltk_status is True:
        print("✓ NLTK resources available for mixed text")
    
    # Determine device (avoid MPS on macOS)
    device = check_and_setup_device(device)
    print(f"✓ Using device: {device}")
    
    # Set environment variables to prevent MPS usage in transformers/BERT
    if device == "cpu" or sys.platform == "darwin":
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        os.environ['TRANSFORMERS_NO_MPS'] = '1'
        # Disable MPS for transformers
        os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"✓ Output directory: {output_path.absolute()}")
    
    # Initialize model
    try:
        print("\n🔄 Initializing MeloTTS Chinese model...")
        # Ensure device is CPU on macOS (even if user specified something else)
        if sys.platform == "darwin" and device != "cpu":
            print(f"⚠ Overriding device '{device}' to 'cpu' on macOS to avoid MPS issues")
            device = "cpu"
        
        # Apply patches one more time right before model initialization
        if sys.platform == "darwin":
            patch_melotts_for_cpu()
        
        model = TTS(language='ZH', device=device)
        print("✓ Model initialized successfully")
        
        # Get available speakers
        try:
            speaker_ids = model.hps.data.spk2id
            print(f"✓ Available speakers: {list(speaker_ids.keys())}")
            
            # Use specified speaker or default Chinese speaker
            if speaker_id is None:
                # Try to find a Chinese speaker
                if 'ZH' in speaker_ids:
                    selected_speaker = speaker_ids['ZH']
                    speaker_name = 'ZH'
                elif 'Chinese' in speaker_ids:
                    selected_speaker = speaker_ids['Chinese']
                    speaker_name = 'Chinese'
                elif 'CN' in speaker_ids:
                    selected_speaker = speaker_ids['CN']
                    speaker_name = 'CN'
                else:
                    # Use first available speaker
                    speaker_name = list(speaker_ids.keys())[0]
                    selected_speaker = speaker_ids[speaker_name]
                print(f"✓ Using speaker: {speaker_name} (ID: {selected_speaker})")
            else:
                if speaker_id in speaker_ids:
                    selected_speaker = speaker_ids[speaker_id]
                    speaker_name = speaker_id
                    print(f"✓ Using specified speaker: {speaker_name} (ID: {selected_speaker})")
                else:
                    print(f"⚠ Warning: Speaker '{speaker_id}' not found. Available: {list(speaker_ids.keys())}")
                    speaker_name = list(speaker_ids.keys())[0]
                    selected_speaker = speaker_ids[speaker_name]
                    print(f"✓ Using default speaker: {speaker_name} (ID: {selected_speaker})")
        except Exception as e:
            print(f"⚠ Warning: Could not get speaker IDs: {e}")
            selected_speaker = 0  # Default speaker ID
            speaker_name = "default"
            print(f"✓ Using default speaker ID: {selected_speaker}")
        
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure MeloTTS is properly installed: pip install -e . (from MeloTTS directory)")
        print("2. Check if you have sufficient disk space for model downloads")
        print("3. Verify that Chinese language support is available in your MeloTTS installation")
        return False
    
    # Test texts (if not provided)
    # Include texts specifically designed to test tones (声调) and quality
    test_texts = [
        "日本经济是高度发达的社会市场经济，日本的经济模式通常被称为东亚模式。"
    ]
    
    if text:
        texts_to_test = [text]
    else:
        texts_to_test = test_texts
    
    # Generate audio for each text
    print(f"\n🎤 Generating audio for {len(texts_to_test)} text(s)...")
    print("-" * 60)
    
    success_count = 0
    for i, test_text in enumerate(texts_to_test, 1):
        try:
            print(f"\n[{i}/{len(texts_to_test)}] Text: {test_text[:50]}{'...' if len(test_text) > 50 else ''}")
            
            # Generate output filename
            output_file = output_path / f"chinese_melotts_test_{i:02d}.wav"
            
            # Synthesize speech
            speed_display = f"{speed}x"
            if speed < 0.8:
                speed_display += " (慢速，适合检查质量)"
            elif speed > 1.2:
                speed_display += " (快速)"
            else:
                speed_display += " (正常)"
            print(f"   Speed: {speed_display}, Speaker: {speaker_name}")
            print(f"   Generating audio...")
            
            model.tts_to_file(
                test_text,
                selected_speaker,
                str(output_file),
                speed=speed
            )
            
            # Check if file was created
            if output_file.exists():
                file_size = output_file.stat().st_size
                print(f"   ✓ Success! Saved to: {output_file}")
                print(f"   ✓ File size: {file_size / 1024:.2f} KB")
                success_count += 1
            else:
                print(f"   ❌ Error: File was not created")
                
        except RuntimeError as e:
            error_msg = str(e)
            if "MPS device" in error_msg or "Placeholder storage" in error_msg:
                print(f"   ⚠ MPS device error detected. This is a known issue with BERT models on macOS.")
                print(f"   💡 Try running with --device cpu to force CPU usage")
                print(f"   Error: {error_msg}")
            elif "averaged_perceptron_tagger" in error_msg or "NLTK" in error_msg:
                print(f"   ⚠ NLTK resource missing for mixed Chinese-English text.")
                print(f"   💡 Install NLTK resources:")
                print(f"      python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"")
                print(f"   Error: {error_msg}")
            else:
                print(f"   ❌ Error generating audio: {error_msg}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            error_msg = str(e)
            if "averaged_perceptron_tagger" in error_msg or "NLTK" in error_msg:
                print(f"   ⚠ NLTK resource missing for mixed Chinese-English text.")
                print(f"   💡 Install NLTK resources:")
                print(f"      python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng')\"")
            else:
                print(f"   ❌ Error generating audio: {error_msg}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {success_count}/{len(texts_to_test)} successful")
    print(f"📁 Output directory: {output_path.absolute()}")
    
    if success_count > 0:
        print("\n✓ Chinese MeloTTS test completed successfully!")
        if speed >= 0.9:
            print(f"\n💡 Quality Checking Tips:")
            print(f"   Current speed: {speed}x (may be too fast for tone evaluation)")
            print(f"   For better tone (声调) quality checking, try slower speeds:")
            print(f"     python test_melotts_chinese.py --speed 0.7  # Recommended")
            print(f"     python test_melotts_chinese.py --speed 0.6  # Very slow, best for detailed checking")
        print(f"\nTo play the audio files, run:")
        print(f"   open {output_path}  # macOS")
        print(f"   xdg-open {output_path}  # Linux")
        print(f"   explorer {output_path}  # Windows")
        return True
    else:
        print("\n❌ All tests failed. Please check the error messages above.")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Test Chinese MeloTTS (MyShell AI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default sample texts
  python test_melotts_chinese.py
  
  # Test with custom text
  python test_melotts_chinese.py --text "你好，世界！"
  
  # Test with slower speed for quality checking (recommended for tone evaluation)
  python test_melotts_chinese.py --speed 0.7
  
  # Test with custom speed and output directory
  python test_melotts_chinese.py --speed 0.8 --output-dir my_output
  
  # Force CPU (recommended on macOS to avoid MPS issues)
  python test_melotts_chinese.py --device cpu
  
  # Test with GPU (Linux/Windows)
  python test_melotts_chinese.py --device cuda:0
  
  # Test with mixed Chinese-English text (requires NLTK resources)
  python test_melotts_chinese.py --text "MeloTTS 支持混合中文和英文的合成。"
  
Note: On macOS, CPU is automatically selected to avoid MPS device issues with BERT models.
For mixed Chinese-English text, install NLTK resources:
  python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
        """
    )
    
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Chinese text to synthesize (if not provided, uses default test texts)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_melotts_output",
        help="Directory to save output audio files (default: test_melotts_output)"
    )
    
    parser.add_argument(
        "--speed",
        type=float,
        default=0.75,
        help="Speech speed multiplier (default: 0.75 for quality checking, range: 0.5-2.0). "
             "Slower speeds (0.6-0.8) are better for checking tone quality (声调)."
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto' (CPU on macOS, CUDA on Linux/Windows), 'cpu', 'cuda:0', etc. (default: auto)"
    )
    
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Specific speaker ID to use (default: auto-select Chinese speaker)"
    )
    
    args = parser.parse_args()
    
    # Validate speed
    if args.speed < 0.5 or args.speed > 2.0:
        print("⚠ Warning: Speed should be between 0.5 and 2.0. Clamping to valid range.")
        args.speed = max(0.5, min(2.0, args.speed))
    
    # Provide speed recommendations
    if args.speed > 0.9:
        print("💡 Tip: For better quality checking (especially tones/声调), try slower speeds:")
        print("   --speed 0.7  (recommended for tone evaluation)")
        print("   --speed 0.6  (very slow, best for detailed quality checking)")
        print()
    
    print("=" * 60)
    print("Chinese MeloTTS Test Script")
    print("=" * 60)
    print()
    
    success = test_chinese_melotts(
        text=args.text,
        output_dir=args.output_dir,
        speed=args.speed,
        device=args.device,
        speaker_id=args.speaker
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
