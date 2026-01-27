#!/usr/bin/env python3
"""Quick test to verify BlackHole is receiving audio"""
import sounddevice as sd
import numpy as np
import time

print("🔍 Testing BlackHole audio capture...")
print("=" * 50)

# Find BlackHole
devices = sd.query_devices()
blackhole_idx = None
for i, d in enumerate(devices):
    if 'blackhole' in d['name'].lower() and d['max_input_channels'] > 0:
        blackhole_idx = i
        print(f"✓ Found BlackHole: {d['name']} (index {i})")
        print(f"  Channels: {d['max_input_channels']}")
        print(f"  Sample rate: {d['default_samplerate']}")
        break

if blackhole_idx is None:
    print("❌ BlackHole not found!")
    print("\nAvailable input devices:")
    for i, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            print(f"  [{i}] {d['name']}")
    exit(1)

print("\n🎵 Now play some audio (YouTube, music, system sound, etc.)")
print("   Listening for 10 seconds...\n")

max_level_seen = 0.0
chunks_with_audio = 0
total_chunks = 0

def callback(indata, frames, time, status):
    global max_level_seen, chunks_with_audio, total_chunks
    total_chunks += 1
    
    # Check raw stereo audio
    raw_max = np.abs(indata).max()
    if raw_max > max_level_seen:
        max_level_seen = raw_max
    
    # Convert to mono
    if len(indata.shape) > 1 and indata.shape[1] >= 2:
        mono = np.mean(indata, axis=1)
    else:
        mono = indata.flatten()
    
    mono_max = np.abs(mono).max()
    
    if raw_max > 0.001:
        chunks_with_audio += 1
        db = 20 * np.log10(raw_max + 1e-10)
        print(f"  ✓ Audio detected! Level: {raw_max:.6f} ({db:.1f} dB)")

try:
    with sd.InputStream(
        device=blackhole_idx,
        channels=2,  # Open with 2 channels
        samplerate=16000,
        blocksize=480,
        callback=callback,
        dtype=np.float32
    ):
        time.sleep(10)
except KeyboardInterrupt:
    pass

print("\n" + "=" * 50)
print(f"Results:")
print(f"  Total chunks: {total_chunks}")
print(f"  Chunks with audio: {chunks_with_audio}")
print(f"  Max level seen: {max_level_seen:.6f}")

if max_level_seen < 0.001:
    print("\n❌ NO AUDIO DETECTED!")
    print("\nTroubleshooting:")
    print("1. Is Multi-Output Device selected in System Settings → Sound → Output?")
    print("2. In Audio MIDI Setup:")
    print("   - Is your Multi-Output Device selected?")
    print("   - Are BOTH your speakers AND BlackHole 2ch checked?")
    print("3. Did you play audio during the test?")
    print("4. Try: afplay /System/Library/Sounds/Glass.aiff")
    print("\n💡 Make sure audio is actually playing (you should hear it)")
else:
    print(f"\n✓ SUCCESS! BlackHole is receiving audio.")
    print(f"  Max level: {max_level_seen:.6f} ({20 * np.log10(max_level_seen + 1e-10):.1f} dB)")
