# CosyVoice2 Setup Guide

## Installation

```bash
pip install funasr modelscope
```

## Finding the Correct Model

If the default model doesn't work, you can find available CosyVoice2 models by:

### Method 1: Check ModelScope Website
Visit https://modelscope.cn and search for "CosyVoice" or "CosyVoice2"

### Method 2: List Models Programmatically
```python
from modelscope import snapshot_download
# Or check FunASR documentation for available models
```

### Method 3: Common CosyVoice2 Model Names
Try these model paths:
- `iic/CosyVoice-300M`
- `iic/cosyvoice2`
- `iic/CosyVoice2`
- `funasr/cosyvoice2`

## Setting Custom Model Path

If you find the correct model name, set it as an environment variable:

```bash
export COSYVOICE2_MODEL_PATH=your/model/path
```

Then restart the server.

## Troubleshooting

1. **Model not found error**: 
   - Check that the model name is correct
   - Verify FunASR/ModelScope is properly installed
   - Try setting COSYVOICE2_MODEL_PATH to a known working model

2. **'NoneType' object is not callable**:
   - This usually means the model path is incorrect
   - Try a different model name from the list above
   - Check server logs for the exact error

3. **Installation issues**:
   - Ensure you have: `pip install funasr modelscope`
   - Check that torch is installed: `pip install torch`
