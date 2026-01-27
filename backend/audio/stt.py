"""Speech-to-Text using OpenAI Whisper"""
import whisper
import numpy as np
import logging

logger = logging.getLogger(__name__)

class STT:
    """Whisper-based Speech-to-Text"""
    
    def __init__(self, model_name="base", language=None):
        self.model_name = model_name
        self.language = language
        logger.info(f"Loading Whisper model: {model_name}")
        self.model = whisper.load_model(model_name)
        logger.info("Whisper model loaded successfully")
    
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """Transcribe audio to text"""
        try:
            # Ensure audio is float32 and in correct range
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            # Whisper expects audio in range [-1, 1]
            audio = np.clip(audio, -1.0, 1.0)
            
            # Transcribe with optimized settings for minimal latency and high accuracy
            result = self.model.transcribe(
                audio,
                language=self.language,
                task="transcribe",
                fp16=False,  # Use fp32 for compatibility
                verbose=False,
                # Optimized for speed and accuracy
                condition_on_previous_text=True,  # Better accuracy with context
                initial_prompt=None,  # No prompt for faster processing
                temperature=0.0,  # Deterministic for consistency and speed
                compression_ratio_threshold=2.4,  # Default
                logprob_threshold=-1.0,  # Default
                no_speech_threshold=0.6,  # Default
                best_of=1,  # No beam search for speed (use 1 for fastest)
                beam_size=1  # Single beam for minimal latency
            )
            
            text = result.get("text", "").strip()
            language = result.get("language", self.language)
            
            # Calculate confidence from segments if available
            confidence = 1.0
            if "segments" in result and result["segments"]:
                # Average confidence from segments
                confidences = [s.get("no_speech_prob", 0.0) for s in result["segments"]]
                if confidences:
                    # Convert no_speech_prob to confidence (inverse)
                    confidence = 1.0 - np.mean(confidences)
            
            return {
                "text": text,
                "language": language,
                "confidence": float(confidence)
            }
        except Exception as e:
            logger.error(f"STT transcription error: {e}")
            return {
                "text": "",
                "language": self.language,
                "confidence": 0.0
            }
