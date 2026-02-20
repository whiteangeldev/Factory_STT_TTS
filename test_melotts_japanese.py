#!/usr/bin/env python3
"""
Test script for Japanese MeloTTS (MyShell AI)

This script tests the MeloTTS Japanese TTS system with various sample texts.
It checks for installation, initializes the model, and generates audio files.

Installation:
    git clone https://github.com/myshell-ai/MeloTTS.git
    cd MeloTTS
    pip install -e .
    python -m unidic download
"""

import os
import sys
import argparse
from pathlib import Path

def check_melotts_installation():
    """Check if MeloTTS is installed and available."""
    try:
        from melo.api import TTS
        return True, TTS
    except ImportError as e:
        return False, None

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

def test_japanese_melotts(
    text: str = None,
    output_dir: str = "test_melotts_output",
    speed: float = 1.0,
    device: str = "auto",
    speaker_id: str = None
):
    """
    Test Japanese MeloTTS with given text.
    
    Args:
        text: Japanese text to synthesize. If None, uses default test texts.
        output_dir: Directory to save output audio files.
        speed: Speech speed (1.0 = normal, >1.0 = faster, <1.0 = slower).
        device: Device to use ('auto', 'cpu', 'cuda:0', etc.).
        speaker_id: Specific speaker ID to use. If None, uses default Japanese speaker.
    """
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
        print("3. Download Japanese dictionary:")
        print("   python -m unidic download")
        print("\nAlternatively, install via pip:")
        print("   pip install melotts")
        return False
    
    print("✓ MeloTTS is installed")
    
    # Determine device (avoid MPS on macOS)
    device = check_and_setup_device(device)
    print(f"✓ Using device: {device}")
    
    # Set environment variable to prevent MPS usage in transformers/BERT
    if device == "cpu":
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        # Force CPU for transformers models
        os.environ['TRANSFORMERS_NO_MPS'] = '1'
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    print(f"✓ Output directory: {output_path.absolute()}")
    
    # Initialize model
    try:
        print("\n🔄 Initializing MeloTTS Japanese model...")
        model = TTS(language='JP', device=device)
        print("✓ Model initialized successfully")
        
        # Get available speakers
        try:
            speaker_ids = model.hps.data.spk2id
            print(f"✓ Available speakers: {list(speaker_ids.keys())}")
            
            # Use specified speaker or default Japanese speaker
            if speaker_id is None:
                # Try to find a Japanese speaker
                if 'JP' in speaker_ids:
                    selected_speaker = speaker_ids['JP']
                    speaker_name = 'JP'
                elif 'Japanese' in speaker_ids:
                    selected_speaker = speaker_ids['Japanese']
                    speaker_name = 'Japanese'
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
        print("2. Download Japanese dictionary: python -m unidic download")
        print("3. Check if you have sufficient disk space for model downloads")
        return False
    
    # Test texts (if not provided)
    test_texts = [
        "アメリカ合衆国：国家レベルでは法定の公用語はないものの国の起こりがイギリスの植民地であったことから英語が事実上の公用語である。また州レベルでは公用語が規定されている場合がある。ニューメキシコ州のスペイン語、ハワイ州のハワイ語など、州によっては別の言語が英語と併せて公用語指定を受けている。アメリカ全域においては、ATMなど公共の場でスペイン語が併記されていることが多く、スペイン語学習者も多いことから、スペイン語が事実上アメリカ国内における第二言語（英語の母語話者にとっては第一外国語）と化している。これは、近年増加しているヒスパニックの影響と推定される。コモンウェルスであるプエルトリコは、1902年のフォラカー法によりスペイン語と英語が公用語となっているが、住人の大多数は英語はほとんど使わず、スペイン語しか話さない。"
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
            output_file = output_path / f"japanese_melotts_test_{i:02d}.wav"
            
            # Synthesize speech
            print(f"   Speed: {speed}x, Speaker: {speaker_name}")
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
            else:
                print(f"   ❌ Error generating audio: {error_msg}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"   ❌ Error generating audio: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {success_count}/{len(texts_to_test)} successful")
    print(f"📁 Output directory: {output_path.absolute()}")
    
    if success_count > 0:
        print("\n✓ Japanese MeloTTS test completed successfully!")
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
        description="Test Japanese MeloTTS (MyShell AI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default sample texts
  python test_melotts_japanese.py
  
  # Test with custom text
  python test_melotts_japanese.py --text "こんにちは、元気ですか？"
  
  # Test with custom speed and output directory
  python test_melotts_japanese.py --speed 1.2 --output-dir my_output
  
  # Force CPU (recommended on macOS to avoid MPS issues)
  python test_melotts_japanese.py --device cpu
  
  # Test with GPU (Linux/Windows)
  python test_melotts_japanese.py --device cuda:0
  
Note: On macOS, CPU is automatically selected to avoid MPS device issues with BERT models.
        """
    )
    
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Japanese text to synthesize (if not provided, uses default test texts)"
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
        default=1.0,
        help="Speech speed multiplier (default: 1.0, range: 0.5-2.0)"
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
        help="Specific speaker ID to use (default: auto-select Japanese speaker)"
    )
    
    args = parser.parse_args()
    
    # Validate speed
    if args.speed < 0.5 or args.speed > 2.0:
        print("⚠ Warning: Speed should be between 0.5 and 2.0. Clamping to valid range.")
        args.speed = max(0.5, min(2.0, args.speed))
    
    print("=" * 60)
    print("Japanese MeloTTS Test Script")
    print("=" * 60)
    print()
    
    success = test_japanese_melotts(
        text=args.text,
        output_dir=args.output_dir,
        speed=args.speed,
        device=args.device,
        speaker_id=args.speaker
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
