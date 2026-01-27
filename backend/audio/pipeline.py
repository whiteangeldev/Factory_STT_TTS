"""Audio processing pipeline with VAD, RNNoise, and OpenAI Realtime STT"""
import numpy as np
import logging
from typing import Optional, Callable, Dict, Any
import time

from .vad import VAD
from .rnnoise import RNNoise
from .stt_openai_realtime import OpenAIRealtimeSTT
from ..config import AudioConfig

logger = logging.getLogger(__name__)

class AudioPipeline:
    def __init__(self, config: AudioConfig, event_callback: Optional[Callable] = None):
        self.config = config
        self.event_callback = event_callback
        
        self.vad = VAD(config.VAD_AGGRESSIVENESS, config.VAD_FRAME_MS, config.SAMPLE_RATE)
        self.rnnoise = RNNoise(sample_rate=config.SAMPLE_RATE)
        self.streaming_stt = OpenAIRealtimeSTT(
            model="gpt-4o-realtime-preview-2024-10-01",
            sample_rate=24000,
            on_transcript=self._on_transcript
        )
        
        self.stt_available = self.streaming_stt.api_key is not None
        if not self.stt_available:
            logger.warning("OpenAI Realtime STT not available (API key missing)")
        else:
            logger.info("OpenAI Realtime STT available")
        
        self.is_speaking = False
        self.speech_start_time = None
        self.stt_start_failed = False
        self.stt_stop_time = None
        self.stt_cooldown_seconds = 0.5
        
        self.min_speech_chunks = max(1, int(config.MIN_SPEECH_DURATION_MS / config.VAD_FRAME_MS))
        self.hangover_chunks = int(config.SPEECH_HANGOVER_MS / config.VAD_FRAME_MS)
        self.speech_chunk_count = 0
        self.silence_chunk_count = 0
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        if self.event_callback:
            try:
                self.event_callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event callback: {e}")
    
    def _on_transcript(self, text: str, is_final: bool, language: str, confidence: float):
        if not text or not text.strip():
            return
        
        # Log transcription
        status = "FINAL" if is_final else "INTERIM"
        logger.info(f"[Pipeline STT {status}] '{text[:80]}...'")
        
        # Emit to frontend
        event_type = "transcription" if is_final else "transcription_interim"
        self._emit_event(event_type, {
            "text": text.strip(),
            "language": language or "en",
            "confidence": float(confidence) if confidence else (1.0 if is_final else 0.8)
        })
    
    def _resample_to_24k(self, audio: np.ndarray) -> np.ndarray:
        if self.config.SAMPLE_RATE == 24000:
            return audio
        ratio = 24000 / self.config.SAMPLE_RATE
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
                    else:
                        logger.error("Failed to start STT stream")
                        if not self.stt_start_failed:
                            self.stt_start_failed = True
                        self.is_speaking = False
                        return denoised_audio
            
            if self.is_speaking:
                # Resample to 24kHz for STT
                audio_24k = self._resample_to_24k(denoised_audio)
                # Send audio to STT (it will skip silence internally)
                self.streaming_stt.send_audio(audio_24k)
        else:
            self.silence_chunk_count += 1
            if not self.is_speaking:
                self.speech_chunk_count = 0
            
            if self.is_speaking:
                if self.silence_chunk_count >= self.hangover_chunks:
                    self.is_speaking = False
                    duration = time.time() - self.speech_start_time if self.speech_start_time else 0
                    logger.info(f"Speech ended - stopping STT, duration={duration:.2f}s")
                    self.streaming_stt.stop_stream()
                    self.stt_stop_time = time.time()
                    self._emit_event("speech_end", {"timestamp": time.time(), "duration": duration})
                    self.speech_chunk_count = 0
                    self.silence_chunk_count = 0
                else:
                    # Continue sending audio during hangover period
                    audio_24k = self._resample_to_24k(denoised_audio)
                    self.streaming_stt.send_audio(audio_24k)
        
        return denoised_audio
    
    def reset(self):
        if self.is_speaking:
            self.streaming_stt.stop_stream()
        self.is_speaking = False
        self.speech_start_time = None
        self.speech_chunk_count = 0
        self.silence_chunk_count = 0
        self.stt_start_failed = False
        self.stt_stop_time = None
