"""Audio processing pipeline with VAD, RNNoise, and offline Whisper STT"""
import numpy as np
import logging
from typing import Optional, Callable, Dict, Any
import time

from .vad import VAD
from .rnnoise import RNNoise
from .stt_whisper_offline import WhisperOfflineSTT
from ..config import AudioConfig

logger = logging.getLogger(__name__)

class AudioPipeline:
    def __init__(self, config: AudioConfig, event_callback: Optional[Callable] = None):
        self.config = config
        self.event_callback = event_callback
        
        self.vad = VAD(config.VAD_AGGRESSIVENESS, config.VAD_FRAME_MS, config.SAMPLE_RATE)
        self.rnnoise = RNNoise(sample_rate=config.SAMPLE_RATE)
        self.streaming_stt = WhisperOfflineSTT(
            model="base",  # Whisper model: "tiny", "base", "small", "medium", "large"
            sample_rate=16000,  # Whisper uses 16kHz
            on_transcript=self._on_transcript
        )
        
        self.stt_available = self.streaming_stt.is_available
        if not self.stt_available:
            logger.warning("Whisper STT not available (install with: pip install openai-whisper)")
        else:
            logger.info("Whisper offline STT available")
        
        self.is_speaking = False
        self.speech_start_time = None
        self.stt_start_failed = False
        self.stt_stop_time = None
        self.stt_cooldown_seconds = 0.5
        
        self.min_speech_chunks = max(1, int(config.MIN_SPEECH_DURATION_MS / config.VAD_FRAME_MS))
        self.hangover_chunks = int(config.SPEECH_HANGOVER_MS / config.VAD_FRAME_MS)
        self.speech_chunk_count = 0
        self.silence_chunk_count = 0
        self.speech_pre_buffer = []  # Buffer audio chunks before STT starts to capture beginning
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        if self.event_callback:
            try:
                self.event_callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    def _on_transcript(self, text: str, is_final: bool, language: str, confidence: float):
        if not text or not text.strip():
            logger.debug(f"[Pipeline STT] Empty transcription received (is_final={is_final})")
            return
        
        # Log transcription
        status = "FINAL" if is_final else "INTERIM"
        logger.info(f"[Pipeline STT {status}] '{text[:80]}...'")
        
        # Emit to frontend with timestamp for latency tracking
        event_type = "transcription" if is_final else "transcription_interim"
        self._emit_event(event_type, {
            "text": text.strip(),
            "language": language or "en",
            "confidence": float(confidence) if confidence else (1.0 if is_final else 0.8),
            "timestamp": time.time(),
            "is_final": is_final
        })
    
    def _resample_to_stt_rate(self, audio: np.ndarray) -> np.ndarray:
        """Resample audio to STT target rate (16kHz for Whisper)"""
        target_rate = 16000  # Whisper uses 16kHz
        
        if self.config.SAMPLE_RATE == target_rate:
            return audio
        
        ratio = target_rate / self.config.SAMPLE_RATE
        new_length = int(len(audio) * ratio)
        try:
            from scipy import signal
            return signal.resample(audio, new_length).astype(np.float32)
        except ImportError:
            indices = np.linspace(0, len(audio) - 1, new_length)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    
    def process_chunk(self, audio: np.ndarray, reference_audio: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        if len(audio) == 0:
            return None
        
        denoised_audio = self.rnnoise.reduce_noise(audio)
        is_speech = self.vad.is_speech(denoised_audio)
        
        if is_speech:
            self.speech_chunk_count += 1
            self.silence_chunk_count = 0
            
            if not self.is_speaking:
                # Buffer audio chunks before STT starts to capture the beginning of speech
                audio_16k = self._resample_to_stt_rate(denoised_audio)
                self.speech_pre_buffer.append(audio_16k)
                # Keep only recent chunks (last ~1 second worth)
                max_pre_buffer_samples = int(16000 * 1.0)  # 1 second at 16kHz
                total_samples = sum(len(chunk) for chunk in self.speech_pre_buffer)
                while total_samples > max_pre_buffer_samples and len(self.speech_pre_buffer) > 1:
                    removed = self.speech_pre_buffer.pop(0)
                    total_samples -= len(removed)
                
                if self.speech_chunk_count >= self.min_speech_chunks:
                    if self.stt_stop_time and (time.time() - self.stt_stop_time) < self.stt_cooldown_seconds:
                        return denoised_audio
                    
                    if not self.stt_available:
                        if not self.stt_start_failed:
                            logger.warning("Skipping transcription - API key not configured")
                            self.stt_start_failed = True
                        return denoised_audio
                    
                    logger.info(f"Speech detected ({self.speech_chunk_count} chunks) - starting STT")
                    self.is_speaking = True
                    self.speech_start_time = time.time()
                    self.stt_stop_time = None
                    
                    if self.streaming_stt.start_stream():
                        logger.info("STT stream started")
                        self._emit_event("speech_start", {"timestamp": self.speech_start_time})
                        self.stt_start_failed = False
                        self.stt_stop_time = None  # Clear cooldown timer on successful start
                        # Wait a brief moment for STT connection to be fully ready
                        time.sleep(0.1)
                        # Reset send counter
                        self._stt_send_count = 0
                        
                        # Send pre-buffered audio chunks first to capture the beginning
                        if len(self.speech_pre_buffer) > 0:
                            logger.info(f"Sending {len(self.speech_pre_buffer)} pre-buffered audio chunks to capture speech beginning")
                            for pre_audio in self.speech_pre_buffer:
                                self.streaming_stt.send_audio(pre_audio)
                            self.speech_pre_buffer = []  # Clear after sending
                    else:
                        logger.error("Failed to start STT stream")
                        if not self.stt_start_failed:
                            self.stt_start_failed = True
                        self.is_speaking = False
                        self.speech_pre_buffer = []  # Clear on failure
                        return denoised_audio
            else:
                # Already speaking - send audio normally
                # Resample to 16kHz for Whisper STT
                audio_16k = self._resample_to_stt_rate(denoised_audio)
                # Send audio to STT (it will buffer and process internally)
                # Log first few sends to debug
                if not hasattr(self, '_stt_send_count'):
                    self._stt_send_count = 0
                self._stt_send_count += 1
                if self._stt_send_count <= 5:
                    logger.debug(f"[Pipeline] Sending audio chunk {self._stt_send_count} to STT (len={len(audio_16k)}, max={np.abs(audio_16k).max():.6f})")
                self.streaming_stt.send_audio(audio_16k)
        else:
            self.silence_chunk_count += 1
            if not self.is_speaking:
                self.speech_chunk_count = 0
                self.speech_pre_buffer = []  # Clear pre-buffer if speech not confirmed
            
            if self.is_speaking:
                if self.silence_chunk_count >= self.hangover_chunks:
                    # Send any remaining audio before stopping (important for capturing end of speech)
                    audio_16k = self._resample_to_stt_rate(denoised_audio)
                    self.streaming_stt.send_audio(audio_16k)
                    
                    # Mark that we're about to stop, but continue sending a few more chunks
                    # to ensure we capture the very end of speech
                    if not hasattr(self, '_stopping_stt'):
                        self._stopping_stt = True
                        self._stopping_chunks = 0
                        logger.info("Starting extended stopping period to capture end of speech")
                    
                    self._stopping_chunks += 1
                    
                    # Send a few more chunks after hangover to capture trailing speech
                    if self._stopping_chunks <= 8:  # Send 8 more chunks after hangover (increased from 5)
                        # Continue sending during this extended period - already sent above
                        pass
                    else:
                        # Now actually stop - set is_speaking to False FIRST so processed_audio events reflect silence
                        self.is_speaking = False
                        
                        # Longer delay to ensure last audio chunks are queued and processed
                        time.sleep(0.5)  # Increased from 0.3
                        
                        duration = time.time() - self.speech_start_time if self.speech_start_time else 0
                        logger.info(f"Speech ended - stopping STT, duration={duration:.2f}s")
                        # stop_stream will handle waiting for final transcription
                        self.streaming_stt.stop_stream()
                        self.stt_stop_time = time.time()
                        self._emit_event("speech_end", {"timestamp": time.time(), "duration": duration})
                        self.speech_chunk_count = 0
                        self.silence_chunk_count = 0
                        self._stopping_stt = False
                        if hasattr(self, '_stopping_chunks'):
                            delattr(self, '_stopping_chunks')
                else:
                    # Continue sending audio during hangover period
                    audio_16k = self._resample_to_stt_rate(denoised_audio)
                    self.streaming_stt.send_audio(audio_16k)
        
        return denoised_audio
    
    def reset(self):
        if self.is_speaking:
            self.streaming_stt.stop_stream()
        self.is_speaking = False
        self.speech_start_time = None
        self.speech_chunk_count = 0
        self.silence_chunk_count = 0
        self.speech_pre_buffer = []
        self.stt_start_failed = False
        self.stt_stop_time = None
