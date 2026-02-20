#!/usr/bin/env python3
"""
Helper script to find available CosyVoice2 models
"""
import sys

print("Searching for CosyVoice2 models...")
print("=" * 60)

# Try FunASR
try:
    from funasr import AutoModel
    print("✓ FunASR is installed")
    
    # Try to list or search for models
    print("\nTrying common CosyVoice2 model names:")
    test_models = [
        "iic/CosyVoice-300M",
        "iic/cosyvoice2",
        "iic/CosyVoice2",
        "iic/CosyVoice",
        "funasr/cosyvoice2",
        "FunASR/CosyVoice2",
    ]
    
    for model_name in test_models:
        try:
            print(f"  Testing: {model_name}...", end=" ")
            # Just try to initialize (don't fully load)
            # This will fail if model doesn't exist, but give us info
            model = AutoModel(model=model_name, device="cpu")
            print("✓ FOUND!")
            print(f"    Use this model: export COSYVOICE2_MODEL_PATH={model_name}")
            sys.exit(0)
        except Exception as e:
            error_msg = str(e)
            if "does not exist" in error_msg or "not found" in error_msg.lower():
                print("✗ Not found")
            else:
                print(f"⚠ Error: {error_msg[:50]}...")
                # Might be a different error, model might exist
                print(f"    Try this model: export COSYVOICE2_MODEL_PATH={model_name}")
                sys.exit(0)
    
except ImportError:
    print("✗ FunASR not installed")
    print("  Install with: pip install funasr")

# Try ModelScope
try:
    from modelscope import AutoModel
    print("\n✓ ModelScope is installed")
    print("\nTrying common CosyVoice2 model names via ModelScope:")
    
    for model_name in test_models:
        try:
            print(f"  Testing: {model_name}...", end=" ")
            model = AutoModel(model=model_name, device="cpu")
            print("✓ FOUND!")
            print(f"    Use this model: export COSYVOICE2_MODEL_PATH={model_name}")
            sys.exit(0)
        except Exception as e:
            error_msg = str(e)
            if "does not exist" in error_msg or "not found" in error_msg.lower():
                print("✗ Not found")
            else:
                print(f"⚠ Error: {error_msg[:50]}...")
                print(f"    Try this model: export COSYVOICE2_MODEL_PATH={model_name}")
                sys.exit(0)
                
except ImportError:
    print("✗ ModelScope not installed")
    print("  Install with: pip install modelscope")

print("\n" + "=" * 60)
print("No working model found automatically.")
print("\nManual steps:")
print("1. Visit https://modelscope.cn")
print("2. Search for 'CosyVoice' or 'CosyVoice2'")
print("3. Find a model and note its full path (e.g., 'iic/CosyVoice-xxx')")
print("4. Set it: export COSYVOICE2_MODEL_PATH=the/model/path/you/found")
print("5. Restart your server")
