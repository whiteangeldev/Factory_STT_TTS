#!/usr/bin/env python3
"""
Quick script to check if GPT-SoVITS is properly set up for testing.
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

def check_gpt_sovits():
    """Check if GPT-SoVITS is available"""
    print("\nChecking GPT-SoVITS...")
    
    # Try multiple import paths
    import_paths = [
        "GPT_SoVITS.inference.infer_tool",
        "inference.infer_tool",
        "GPT_SoVITS.inference",
    ]
    
    gpt_sovits_available = False
    for import_path in import_paths:
        try:
            # Try to add potential paths
            possible_paths = [
                os.path.join(os.path.dirname(__file__), 'GPT-SoVITS'),
                os.path.join(os.path.dirname(__file__), '..', 'GPT-SoVITS'),
                'GPT-SoVITS',
            ]
            
            for path in possible_paths:
                if os.path.exists(path) and path not in sys.path:
                    sys.path.insert(0, path)
            
            __import__(import_path)
            print(f"  ✓ GPT-SoVITS found (imported from: {import_path})")
            gpt_sovits_available = True
            break
        except ImportError:
            continue
    
    if not gpt_sovits_available:
        print("  ✗ GPT-SoVITS is not installed")
        print("    Install from: https://github.com/RVC-Boss/GPT-SoVITS")
        print("    Or clone: git clone https://github.com/RVC-Boss/GPT-SoVITS.git")
        return False
    
    # Check model directories
    print("\nChecking GPT-SoVITS model directories...")
    
    chinese_dir = os.getenv("GPT_SOVITS_CHINESE_MODEL_DIR", "")
    japanese_dir = os.getenv("GPT_SOVITS_JAPANESE_MODEL_DIR", "")
    
    if chinese_dir:
        print(f"  Chinese model dir: {chinese_dir}")
        if os.path.exists(chinese_dir):
            required_files = ["gpt_model.pth", "sovits_model.pth", "config.json", "reference.wav"]
            missing = []
            for file in required_files:
                if not os.path.exists(os.path.join(chinese_dir, file)):
                    missing.append(file)
            if missing:
                print(f"    ✗ Missing files: {', '.join(missing)}")
            else:
                print("    ✓ All required files present")
        else:
            print("    ✗ Directory does not exist")
    else:
        print("  ✗ GPT_SOVITS_CHINESE_MODEL_DIR not set")
        print("    Set with: export GPT_SOVITS_CHINESE_MODEL_DIR=/path/to/models/chinese")
    
    if japanese_dir:
        print(f"  Japanese model dir: {japanese_dir}")
        if os.path.exists(japanese_dir):
            required_files = ["gpt_model.pth", "sovits_model.pth", "config.json", "reference.wav"]
            missing = []
            for file in required_files:
                if not os.path.exists(os.path.join(japanese_dir, file)):
                    missing.append(file)
            if missing:
                print(f"    ✗ Missing files: {', '.join(missing)}")
            else:
                print("    ✓ All required files present")
        else:
            print("    ✗ Directory does not exist")
    else:
        print("  ✗ GPT_SOVITS_JAPANESE_MODEL_DIR not set")
        print("    Set with: export GPT_SOVITS_JAPANESE_MODEL_DIR=/path/to/models/japanese")
    
    return gpt_sovits_available

def main():
    print("="*60)
    print("GPT-SoVITS Setup Checker")
    print("="*60)
    
    pykokoro_ok = check_pykokoro()
    gpt_sovits_ok = check_gpt_sovits()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    if pykokoro_ok:
        print("✓ PyKokoro is ready for testing")
    else:
        print("✗ PyKokoro needs setup")
    
    if gpt_sovits_ok:
        print("✓ GPT-SoVITS is available")
        chinese_dir = os.getenv("GPT_SOVITS_CHINESE_MODEL_DIR", "")
        japanese_dir = os.getenv("GPT_SOVITS_JAPANESE_MODEL_DIR", "")
        
        if chinese_dir and os.path.exists(chinese_dir):
            print("✓ Chinese models configured")
        else:
            print("✗ Chinese models not configured")
        
        if japanese_dir and os.path.exists(japanese_dir):
            print("✓ Japanese models configured")
        else:
            print("✗ Japanese models not configured")
    else:
        print("✗ GPT-SoVITS needs setup")
    
    print("\nNext steps:")
    if not pykokoro_ok:
        print("  1. Install PyKokoro: pip install pykokoro spacy")
        print("  2. Download spaCy models: python -m spacy download zh_core_web_sm ja_core_news_sm")
    
    if not gpt_sovits_ok:
        print("  1. Clone GPT-SoVITS: git clone https://github.com/RVC-Boss/GPT-SoVITS.git")
        print("  2. Install dependencies: cd GPT-SoVITS && pip install -r requirements.txt")
    
    chinese_dir = os.getenv("GPT_SOVITS_CHINESE_MODEL_DIR", "")
    japanese_dir = os.getenv("GPT_SOVITS_JAPANESE_MODEL_DIR", "")
    if not chinese_dir or not os.path.exists(chinese_dir):
        print("  3. Train or download Chinese GPT-SoVITS models")
        print("  4. Set GPT_SOVITS_CHINESE_MODEL_DIR environment variable")
    if not japanese_dir or not os.path.exists(japanese_dir):
        print("  5. Train or download Japanese GPT-SoVITS models")
        print("  6. Set GPT_SOVITS_JAPANESE_MODEL_DIR environment variable")
    
    if pykokoro_ok and (gpt_sovits_ok or True):  # Can test with just PyKokoro
        print("\n✓ Ready to run tests!")
        print("  Run: python test_gpt_sovits.py")

if __name__ == "__main__":
    main()
