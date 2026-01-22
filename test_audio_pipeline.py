"""Test script for audio pipeline"""
import numpy as np
import time
import logging
from backend.audio.capture import AudioCapture
from backend.audio.pipeline import AudioPipeline
from backend.config import AudioConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_audio_capture():
    """Test audio capture"""
    print("Testing Audio Capture...")
    print("=" * 50)
    
    config = AudioConfig()
    capture = AudioCapture(
        sample_rate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        chunk_size=config.CHUNK_SIZE
    )
    
    # List devices
    devices = capture.list_devices()
    print(f"\nAvailable audio devices: {len(devices)}")
    for device in devices:
        print(f"  [{device['index']}] {device['name']}")
    
    # Start recording
    print("\nRecording for 5 seconds... (speak into microphone)")
    capture.start_stream()
    
    chunks = []
    start_time = time.time()
    
    while time.time() - start_time < 5:
        chunk = capture.read_chunk()
        if chunk is not None:
            chunks.append(chunk)
        time.sleep(0.01)
    
    capture.stop_stream()
    capture.cleanup()
    
    if chunks:
        audio = np.concatenate(chunks)
        print(f"Captured {len(audio) / config.SAMPLE_RATE:.2f} seconds of audio")
        print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")
        print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
        return audio
    else:
        print("No audio captured!")
        return None

def test_pipeline(audio: np.ndarray):
    """Test audio pipeline"""
    print("\n\nTesting Audio Pipeline...")
    print("=" * 50)
    
    config = AudioConfig()
    pipeline = AudioPipeline(config)
    
    # Process audio
    print("Processing audio through pipeline...")
    start_time = time.time()
    
    processed = pipeline.process_stream(audio)
    
    processing_time = time.time() - start_time
    
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Original length: {len(audio) / config.SAMPLE_RATE:.2f} seconds")
    print(f"Processed length: {len(processed) / config.SAMPLE_RATE:.2f} seconds")
    
    if len(processed) > 0:
        print(f"Processed audio range: [{processed.min():.3f}, {processed.max():.3f}]")
        print("✓ Pipeline working correctly!")
    else:
        print("⚠ No speech detected in audio")
    
    return processed

def main():
    """Main test function"""
    print("Audio Pipeline Test Suite")
    print("=" * 50)
    
    # Test 1: Audio Capture
    audio = test_audio_capture()
    
    if audio is not None:
        # Test 2: Audio Pipeline
        processed = test_pipeline(audio)
        
        print("\n" + "=" * 50)
        print("Test Complete!")
        print("=" * 50)
    else:
        print("Failed to capture audio. Check microphone connection.")

if __name__ == "__main__":
    main()