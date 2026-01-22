"""Audio capture module for microphone input"""
import pyaudio
import numpy as np
from typing import Callable, Optional
import logging

logger = logging.getLogger(__name__)

class AudioCapture:
    """Handles real-time audio capture from microphone"""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 format: int = pyaudio.paInt16):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.format = format
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
    def list_devices(self):
        """List available audio input devices"""
        devices = []
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate'])
                })
        return devices
    
    def start_stream(self, callback: Optional[Callable] = None):
        """Start audio stream with optional callback"""
        try:
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=callback
            )
            self.stream.start_stream()
            self.is_recording = True
            logger.info("Audio stream started")
            return True
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            return False
    
    def read_chunk(self) -> Optional[np.ndarray]:
        """Read a single chunk of audio data"""
        if not self.stream or not self.is_recording:
            return None
        
        try:
            data = self.stream.read(self.chunk_size, exception_on_overflow=False)
            # Convert bytes to numpy array
            audio_array = np.frombuffer(data, dtype=np.int16)
            # Normalize to float32 [-1, 1]
            audio_float = audio_array.astype(np.float32) / 32768.0
            return audio_float
        except Exception as e:
            logger.error(f"Error reading audio chunk: {e}")
            return None
    
    def stop_stream(self):
        """Stop audio stream"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        self.is_recording = False
        logger.info("Audio stream stopped")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_stream()
        if self.audio:
            self.audio.terminate()