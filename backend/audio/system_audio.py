"""Server-side system audio capture (bypasses browser limitations)"""
import numpy as np
import logging
import threading
import queue
from typing import Optional, Callable

logger = logging.getLogger(__name__)

class SystemAudioCapture:
    """Capture system audio directly on the server (no browser screen sharing needed)"""
    
    def __init__(self, sample_rate=16000, chunk_size=480, on_audio: Optional[Callable] = None):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.on_audio = on_audio
        self.is_recording = False
        self.stream = None
        self.thread = None
        self.audio_queue = queue.Queue()
        
    def _detect_backend(self):
        """Detect available audio backend for system audio - prefer pyaudio (works better in Flask/eventlet)"""
        import platform
        system = platform.system()
        
        # Try pyaudio first (works better in Flask/eventlet context, matches AssemblyAI)
        try:
            import pyaudio
            p = pyaudio.PyAudio()
            keywords = ['loopback', 'stereo mix', 'what u hear', 'monitor', 'virtual', 'blackhole', 'soundflower']
            
            logger.info("Checking for system audio devices with pyaudio...")
            device_count = p.get_device_count()
            logger.info(f"pyaudio found {device_count} total devices")
            for i in range(device_count):
                try:
                    info = p.get_device_info_by_index(i)
                    name = info['name'].lower()
                    if info['maxInputChannels'] > 0:
                        # Log all input devices for debugging
                        if any(kw in name for kw in keywords):
                            logger.info(f"Found system audio device (pyaudio): {info['name']} (index {i}, {info['maxInputChannels']}ch)")
                            p.terminate()
                            return 'pyaudio', i
                except Exception as e:
                    logger.debug(f"Error checking pyaudio device {i}: {e}")
            p.terminate()
            logger.info("pyaudio: No system audio device found, falling back to sounddevice")
        except ImportError:
            logger.warning("pyaudio not available, falling back to sounddevice")
        except Exception as e:
            logger.warning(f"Error detecting pyaudio: {e}, falling back to sounddevice")
        
        # Fallback to sounddevice
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            default_input = sd.default.device[0]  # Get default input device index
            
            # Look for system audio/loopback devices
            keywords = ['loopback', 'stereo mix', 'what u hear', 'monitor', 'virtual', 'blackhole', 'soundflower']
            
            for i, device in enumerate(devices):
                name_lower = device['name'].lower()
                # Check if it's a system audio device
                if device['max_input_channels'] > 0:
                    if any(keyword in name_lower for keyword in keywords):
                        logger.info(f"Found system audio device: {device['name']} (index {i})")
                        return 'sounddevice', i
                    # On macOS, look for devices with "Monitor" in name
                    if system == 'Darwin' and 'monitor' in name_lower:
                        logger.info(f"Found macOS monitor device: {device['name']} (index {i})")
                        return 'sounddevice', i
            
            # List all available devices for debugging
            logger.info("Available audio input devices:")
            has_input = False
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    has_input = True
                    marker = " (default)" if i == default_input else ""
                    logger.info(f"  [{i}] {device['name']} - {device['max_input_channels']} channels{marker}")
            
            if not has_input:
                logger.warning("No audio input devices found!")
                logger.info("💡 On macOS, you may need to:")
                logger.info("   1. Install BlackHole: https://github.com/ExistentialAudio/BlackHole")
                logger.info("   2. Or enable 'Monitor' devices in Audio MIDI Setup")
                return None, None
            
            # Try default input device (might work on some systems)
            logger.info(f"Using default input device: {devices[default_input]['name']} (index {default_input})")
            return 'sounddevice', default_input
        except ImportError:
            logger.warning("sounddevice not installed. Install with: pip install sounddevice")
        except Exception as e:
            logger.error(f"Error detecting sounddevice: {e}")
        
        return None, None
    
    def start(self):
        """Start system audio capture"""
        if self.is_recording:
            return True
        
        backend, device = self._detect_backend()
        
        if backend == 'pyaudio':
            return self._start_pyaudio(device)
        elif backend == 'sounddevice':
            return self._start_sounddevice(device)
        else:
            logger.error("No system audio backend available. Install: pip install pyaudio (preferred) or pip install sounddevice")
            return False
    
    def _start_sounddevice(self, device_index):
        """Start capture using sounddevice - matches working AssemblyAI approach"""
        try:
            import sounddevice as sd
            
            # Get device info to determine channel count
            device_info = sd.query_devices(device_index)
            max_channels = device_info['max_input_channels']
            
            # Use the device's actual channel count (usually 2 for BlackHole)
            # Match AssemblyAI: open with device's channels, convert to mono
            capture_channels = min(max_channels, 2)  # Use 1 or 2 channels
            
            # IMPORTANT: Use target sample rate (16000), NOT device's native rate
            # This matches the working AssemblyAI implementation and test script
            logger.info(f"Opening device: {capture_channels}ch @ {self.sample_rate}Hz (target rate)")
            
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio callback status: {status}")
                if not self.is_recording:
                    return
                
                # Check raw audio level BEFORE conversion (for debugging)
                raw_max = np.abs(indata).max()
                
                # Convert to mono if stereo (matches AssemblyAI approach)
                if capture_channels == 2 and len(indata.shape) > 1:
                    # Average both channels to mono
                    audio = np.mean(indata, axis=1)
                else:
                    audio = indata.flatten()
                
                audio_float = audio.astype(np.float32)
                mono_max = np.abs(audio_float).max()
                
                # Debug first few callbacks - show both raw and mono levels
                if not hasattr(audio_callback, '_debug_count'):
                    audio_callback._debug_count = 0
                if audio_callback._debug_count < 5:
                    audio_callback._debug_count += 1
                    logger.info(f"[SystemAudio Debug {audio_callback._debug_count}] shape={indata.shape}, raw_max={raw_max:.6f}, mono_max={mono_max:.6f}, samples={len(audio_float)}")
                    # Also log a sample of actual values to see if they're truly zero
                    if audio_callback._debug_count == 1:
                        logger.info(f"[SystemAudio Sample] First 10 raw values: {indata[:10, 0] if len(indata.shape) > 1 else indata[:10]}")
                
                if self.on_audio:
                    self.on_audio(audio_float)
                self.audio_queue.put(audio_float.copy())
            
            # Open at target sample rate (16000), not device's native rate
            # This matches the working test script and AssemblyAI implementation
            self.stream = sd.InputStream(
                device=device_index,
                channels=capture_channels,  # Open with device's channel count
                samplerate=self.sample_rate,  # Use target rate (16000), not device's native rate
                blocksize=self.chunk_size,
                callback=audio_callback,
                dtype=np.float32
            )
            self.stream.start()
            self.is_recording = True
            logger.info(f"System audio capture started (sounddevice, device={device_index}, {capture_channels}ch @ {self.sample_rate}Hz)")
            return True
        except Exception as e:
            logger.error(f"Failed to start sounddevice capture: {e}")
            logger.info("💡 To enable system audio capture:")
            logger.info("   1. Install: pip install sounddevice")
            logger.info("   2. On macOS: Install BlackHole (https://github.com/ExistentialAudio/BlackHole)")
            logger.info("      Or enable 'Monitor of [device]' in Audio MIDI Setup")
            logger.info("   3. On Windows: Enable 'Stereo Mix' in Sound settings (Recording devices)")
            logger.info("   4. On Linux: Configure PulseAudio loopback module")
            return False
    
    def _start_pyaudio(self, device_index):
        """Start capture using pyaudio - matches working AssemblyAI implementation"""
        try:
            import pyaudio
            import threading
            import time
            
            self.audio = pyaudio.PyAudio()
            
            # Get device info to determine channel count and sample rate (like AssemblyAI)
            try:
                device_info = self.audio.get_device_info_by_index(device_index)
                device_name = device_info['name']
                max_channels = device_info['maxInputChannels']
                device_sample_rate = int(device_info['defaultSampleRate'])
                
                # Use device's channel count (usually 2 for BlackHole)
                capture_channels = min(max_channels, 2) if max_channels >= 2 else 1
                needs_stereo_to_mono = (capture_channels == 2)
                
                # Use device's native sample rate (usually 48000 for BlackHole)
                capture_sample_rate = device_sample_rate
                needs_resample = (capture_sample_rate != self.sample_rate)
                
                logger.info(f"Opening device with pyaudio: {device_name} ({capture_channels} channels -> mono @ {capture_sample_rate}Hz, resample: {needs_resample})")
            except OSError:
                raise RuntimeError(f"Audio device index {device_index} does not exist")
            
            def audio_callback(in_data, frame_count, time_info, status):
                if not self.is_recording:
                    return (None, pyaudio.paContinue)
                
                # Check raw int16 values BEFORE conversion (critical debug)
                audio_int16 = np.frombuffer(in_data, dtype=np.int16)
                raw_max_int16 = np.abs(audio_int16).max()
                
                # Convert int16 bytes to float32 (matches AssemblyAI)
                audio_float = audio_int16.astype(np.float32) / 32768.0
                
                # Convert stereo to mono if needed (matches AssemblyAI)
                if needs_stereo_to_mono and len(audio_float) >= 2:
                    num_samples = len(audio_float) // capture_channels
                    audio_reshaped = audio_float[:num_samples * capture_channels].reshape(num_samples, capture_channels)
                    audio_float = np.mean(audio_reshaped, axis=1)
                
                # Resample if needed (device is at different rate than target)
                if needs_resample:
                    try:
                        from scipy import signal
                        num_samples = int(len(audio_float) * self.sample_rate / capture_sample_rate)
                        audio_float = signal.resample(audio_float, num_samples).astype(np.float32)
                    except ImportError:
                        # Fallback: simple linear interpolation
                        ratio = self.sample_rate / capture_sample_rate
                        indices = np.linspace(0, len(audio_float) - 1, int(len(audio_float) * ratio))
                        audio_float = np.interp(indices, np.arange(len(audio_float)), audio_float).astype(np.float32)
                
                # Debug first few callbacks - show RAW int16 values
                if not hasattr(audio_callback, '_debug_count'):
                    audio_callback._debug_count = 0
                if audio_callback._debug_count < 5:
                    audio_callback._debug_count += 1
                    max_level = np.abs(audio_float).max()
                    logger.info(f"[SystemAudio Debug {audio_callback._debug_count}] pyaudio: raw_int16_max={raw_max_int16}, float_max={max_level:.6f}, samples={len(audio_float)}")
                    if audio_callback._debug_count == 1:
                        logger.info(f"[SystemAudio Sample] First 10 raw int16: {audio_int16[:10]}")
                        logger.info(f"[SystemAudio Sample] First 10 float32: {audio_float[:10]}")
                
                if self.on_audio:
                    self.on_audio(audio_float)
                self.audio_queue.put(audio_float.copy())
                
                return (None, pyaudio.paContinue)
            
            # Calculate frames_per_buffer for device's sample rate
            device_chunk_size = int(self.chunk_size * capture_sample_rate / self.sample_rate)
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=capture_channels,
                rate=capture_sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=device_chunk_size,
                stream_callback=audio_callback
            )
            
            # CRITICAL: Start stream in a real thread to avoid eventlet greenlet issues
            # This matches how AssemblyAI handles it - pyaudio callbacks need real OS threads
            def start_stream_in_thread():
                try:
                    self.stream.start_stream()
                    # Keep the thread alive while recording
                    while self.is_recording:
                        time.sleep(0.1)
                except Exception as e:
                    logger.error(f"Error in pyaudio stream thread: {e}")
            
            self.stream_thread = threading.Thread(target=start_stream_in_thread, daemon=True)
            self.stream_thread.start()
            
            # Wait a moment for stream to start
            time.sleep(0.2)
            
            if not self.stream.is_active():
                raise RuntimeError(f"Audio stream failed to start for device: {device_name}")
            
            self.is_recording = True
            logger.info(f"System audio capture started (pyaudio, device={device_index}, {capture_channels}ch @ {capture_sample_rate}Hz -> {self.sample_rate}Hz)")
            return True
        except Exception as e:
            logger.error(f"Failed to start pyaudio capture: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if hasattr(self, 'audio'):
                try:
                    self.audio.terminate()
                except:
                    pass
            return False
    
    def stop(self):
        """Stop system audio capture"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream:
            try:
                if hasattr(self.stream, 'stop_stream'):
                    self.stream.stop_stream()
                if hasattr(self.stream, 'close'):
                    self.stream.close()
            except:
                pass
            self.stream = None
        
        # Wait for stream thread to finish
        if hasattr(self, 'stream_thread') and self.stream_thread:
            self.stream_thread.join(timeout=1.0)
        
        if hasattr(self, 'audio'):
            try:
                self.audio.terminate()
            except:
                pass
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        
        logger.info("System audio capture stopped")
    
    def read_chunk(self) -> Optional[np.ndarray]:
        """Read a chunk of audio (non-blocking)"""
        try:
            return self.audio_queue.get_nowait()
        except:
            return None
