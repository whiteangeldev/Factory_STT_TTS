#!/usr/bin/env python3
"""
Quick script to check if Fish Speech is properly set up for testing.
"""

import os
import sys
from pathlib import Path

def check_pykokoro():
    """Check if PyKokoro is available"""
    print("Checking PyKokoro...")
    try:
        from pykokoro import build_pipeline
        print("  ✓ PyKokoro is installed")
        
        # Check spaCy models
        try:
            import spacy
            nlp_en = spacy.load("en_core_web_sm")
            print("  ✓ English spaCy model (en_core_web_sm) is available")
        except:
            print("  ✗ English spaCy model (en_core_web_sm) not found")
            print("    Install with: python -m spacy download en_core_web_sm")
        
        try:
            import spacy
            nlp_zh = spacy.load("zh_core_web_sm")
            print("  ✓ Chinese spaCy model (zh_core_web_sm) is available")
        except:
            print("  ✗ Chinese spaCy model (zh_core_web_sm) not found")
            print("    Install with: python -m spacy download zh_core_web_sm")
        
        try:
            import spacy
            nlp_ja = spacy.load("ja_core_news_sm")
            print("  ✓ Japanese spaCy model (ja_core_news_sm) is available")
        except:
            print("  ✗ Japanese spaCy model (ja_core_news_sm) not found")
            print("    Install with: python -m spacy download ja_core_news_sm")
        
        return True
    except ImportError:
        print("  ✗ PyKokoro is not installed")
        print("    Install with: pip install pykokoro spacy")
        return False

def check_fish_speech():
    """Check if Fish Speech is available"""
    print("\nChecking Fish Speech...")
    
    # Try multiple import paths
    import_paths = [
        "fish_speech",
        "fishespeech",
        "FishSpeech",
        "fish_speech.api",
    ]
    
    possible_paths = [
        os.path.join(os.path.dirname(__file__), 'FishSpeech'),
        os.path.join(os.path.dirname(__file__), '..', 'FishSpeech'),
        'FishSpeech',
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    
    fish_speech_available = False
    for import_path in import_paths:
        try:
            __import__(import_path)
            print(f"  ✓ Fish Speech found (imported from: {import_path})")
            fish_speech_available = True
            break
        except ImportError:
            continue
    
    if not fish_speech_available:
        print("  ✗ Fish Speech is not installed")
        print("    Install from: https://github.com/fishaudio/fish-speech")
        print("    Or clone: git clone https://github.com/fishaudio/fish-speech.git")
        return False
    
    # Check model directories
    print("\nChecking Fish Speech model directories...")
    
    languages = ["english", "chinese", "japanese"]
    for lang in languages:
        lang_upper = lang.upper()
        model_dir = os.getenv(f"FISH_SPEECH_{lang_upper}_MODEL_DIR", "")
        
        if model_dir:
            print(f"  {lang.capitalize()} model dir: {model_dir}")
            if os.path.exists(model_dir):
                print("    ✓ Directory exists")
                # Check for common model files
                common_files = ["model.pth", "config.json", "vocoder.pth"]
                found_files = [f for f in common_files if os.path.exists(os.path.join(model_dir, f))]
                if found_files:
                    print(f"    ✓ Found model files: {', '.join(found_files)}")
                else:
                    print("    ⚠ No common model files found (may use different structure)")
            else:
                print("    ✗ Directory does not exist")
        else:
            print(f"  ✗ FISH_SPEECH_{lang_upper}_MODEL_DIR not set")
            print(f"    Set with: export FISH_SPEECH_{lang_upper}_MODEL_DIR=/path/to/models/{lang}")
    
    # Check API URL
    api_url = os.getenv("FISH_SPEECH_API_URL", "")
    if api_url:
        print(f"\n  API URL: {api_url}")
        print("    ✓ Using API-based Fish Speech")
    else:
        print("\n  ✗ FISH_SPEECH_API_URL not set")
        print("    Set with: export FISH_SPEECH_API_URL=http://localhost:8000")
        print("    (Optional - only if using API-based Fish Speech)")
    
    return fish_speech_available

def main():
    print("="*60)
    print("Fish Speech Setup Checker")
    print("="*60)
    
    pykokoro_ok = check_pykokoro()
    fish_speech_ok = check_fish_speech()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if pykokoro_ok:
        print("✓ PyKokoro is ready for testing")
    else:
        print("✗ PyKokoro needs setup")
    
    if fish_speech_ok:
        print("✓ Fish Speech is available")
        languages = ["english", "chinese", "japanese"]
        for lang in languages:
            lang_upper = lang.upper()
            model_dir = os.getenv(f"FISH_SPEECH_{lang_upper}_MODEL_DIR", "")
            api_url = os.getenv("FISH_SPEECH_API_URL", "")
            
            if model_dir and os.path.exists(model_dir):
                print(f"✓ {lang.capitalize()} models configured")
            elif api_url:
                print(f"✓ {lang.capitalize()} using API")
            else:
                print(f"✗ {lang.capitalize()} models not configured")
    else:
        print("✗ Fish Speech needs setup")
    
    print("\nNext steps:")
    if not pykokoro_ok:
        print("  1. Install PyKokoro: pip install pykokoro spacy")
        print("  2. Download spaCy models:")
        print("     python -m spacy download en_core_web_sm")
        print("     python -m spacy download zh_core_web_sm")
        print("     python -m spacy download ja_core_news_sm")
    
    if not fish_speech_ok:
        print("  1. Clone Fish Speech: git clone https://github.com/fishaudio/fish-speech.git")
        print("  2. Install dependencies: cd FishSpeech && pip install -r requirements.txt")
        print("  3. Download or train models for each language")
    
    languages = ["english", "chinese", "japanese"]
    for lang in languages:
        lang_upper = lang.upper()
        model_dir = os.getenv(f"FISH_SPEECH_{lang_upper}_MODEL_DIR", "")
        if not model_dir or not os.path.exists(model_dir):
            api_url = os.getenv("FISH_SPEECH_API_URL", "")
            if not api_url:
                print(f"  4. Configure {lang} models:")
                print(f"     export FISH_SPEECH_{lang_upper}_MODEL_DIR=/path/to/models/{lang}")
                print(f"     OR set up API: export FISH_SPEECH_API_URL=http://localhost:8000")
    
    if pykokoro_ok and (fish_speech_ok or True):  # Can test with just PyKokoro
        print("\n✓ Ready to run tests!")
        print("  Run: python test_fish_speech.py")

if __name__ == "__main__":
    main()
