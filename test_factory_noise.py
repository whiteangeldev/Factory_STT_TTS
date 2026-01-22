"""Factory noise testing script for 85dB+ noise environments"""
import numpy as np
import soundfile as sf
import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import time

from backend.audio.pipeline import AudioPipeline, SpeechState
from backend.audio.vad import VoiceActivityDetector
from backend.audio.noise_suppression import NoiseSuppressor
from backend.config import AudioConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FactoryNoiseTester:
    """Test audio pipeline with factory noise scenarios"""
    
    def __init__(self):
        self.config = AudioConfig()
        self.speech_events = []  # Track speech events
        
        def event_callback(event_type, data):
            self.speech_events.append({"type": event_type, "data": data})
        
        self.pipeline = AudioPipeline(self.config, event_callback=event_callback)
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
    
    def calculate_snr(self, signal: np.ndarray, noise: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio in dB"""
        signal_power = np.mean(signal ** 2)
        noise_power = np.mean(noise ** 2)
        if noise_power == 0:
            return float('inf')
        snr_linear = signal_power / noise_power
        return 10 * np.log10(snr_linear) if snr_linear > 0 else -float('inf')
    
    def calculate_rms(self, audio: np.ndarray) -> float:
        """Calculate RMS (Root Mean Square) level in dB"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms == 0:
            return -float('inf')
        return 20 * np.log10(rms)
    
    def detect_clipping(self, audio: np.ndarray) -> dict:
        """Detect audio clipping"""
        clipped_samples = np.sum(np.abs(audio) > 1.0)
        total_samples = len(audio)
        clipping_percentage = (clipped_samples / total_samples) * 100 if total_samples > 0 else 0
        max_level = np.max(np.abs(audio))
        return {
            "clipped_samples": clipped_samples,
            "total_samples": total_samples,
            "clipping_percentage": clipping_percentage,
            "max_level": max_level,
            "has_clipping": clipping_percentage > 0.1  # More than 0.1% clipped
        }
    
    def measure_latency(self, process_func, *args, **kwargs):
        """Measure processing latency"""
        import time
        start_time = time.perf_counter()
        result = process_func(*args, **kwargs)
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        return result, latency_ms
    
    def test_false_triggers(self, noise_file: str, duration_sec: float = 10.0) -> dict:
        """
        Test false trigger rate (noise detected as speech)
        
        Args:
            noise_file: Path to noise-only audio file
            duration_sec: Duration to test (seconds)
            
        Returns:
            Dictionary with false trigger metrics
        """
        logger.info(f"Testing false triggers with {noise_file}")
        
        try:
            noise, sr = sf.read(noise_file)
            if len(noise.shape) > 1:
                noise = noise[:, 0]
            
            if sr != self.config.SAMPLE_RATE:
                try:
                    import librosa
                    noise = librosa.resample(noise, orig_sr=sr, target_sr=self.config.SAMPLE_RATE)
                except ImportError:
                    logger.warning("librosa not available for resampling")
            
            noise = noise.astype(np.float32)
            if noise.max() > 1.0 or noise.min() < -1.0:
                noise = noise / np.max(np.abs(noise))
            
            # Limit to test duration
            max_samples = int(self.config.SAMPLE_RATE * duration_sec)
            noise = noise[:max_samples]
            
            # Reset pipeline state
            self.pipeline.speech_state = SpeechState.SILENCE
            self.speech_events = []
            
            # Process noise in chunks
            chunk_size = int(self.config.SAMPLE_RATE * 0.1)  # 100ms chunks
            total_chunks = 0
            speech_detected_chunks = 0
            false_triggers = 0
            
            for i in range(0, len(noise), chunk_size):
                chunk = noise[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                
                total_chunks += 1
                processed = self.pipeline.process_chunk(chunk)
                
                if processed is not None:
                    speech_detected_chunks += 1
                
                # Count speech_start events (false triggers)
                for event in self.speech_events:
                    if event["type"] == "speech_start":
                        false_triggers += 1
            
            false_trigger_rate = (false_triggers / (len(noise) / self.config.SAMPLE_RATE)) if len(noise) > 0 else 0
            speech_detection_rate = (speech_detected_chunks / total_chunks) * 100 if total_chunks > 0 else 0
            
            results = {
                "false_triggers": false_triggers,
                "false_trigger_rate_per_sec": false_trigger_rate,
                "speech_detected_chunks": speech_detected_chunks,
                "total_chunks": total_chunks,
                "speech_detection_rate_percent": speech_detection_rate,
                "test_duration_sec": len(noise) / self.config.SAMPLE_RATE
            }
            
            logger.info(f"False triggers: {false_triggers} ({false_trigger_rate:.2f}/sec)")
            logger.info(f"Speech detection rate: {speech_detection_rate:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in false trigger test: {e}")
            return {"error": str(e)}
    
    def mix_speech_noise(self, speech: np.ndarray, noise: np.ndarray, target_snr_db: float) -> Tuple[np.ndarray, float]:
        """
        Mix speech and noise at target SNR
        
        Args:
            speech: Clean speech signal
            noise: Noise signal
            target_snr_db: Target SNR in dB
            
        Returns:
            Mixed audio and actual SNR
        """
        # Ensure same length
        min_len = min(len(speech), len(noise))
        speech = speech[:min_len]
        noise = noise[:min_len]
        
        # Calculate signal and noise power
        speech_power = np.mean(speech ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0:
            return speech, float('inf')
        
        # Calculate scaling factor for noise to achieve target SNR
        target_snr_linear = 10 ** (target_snr_db / 10)
        noise_scale = np.sqrt(speech_power / (noise_power * target_snr_linear))
        
        # Mix signals
        mixed = speech + (noise * noise_scale)
        
        # Calculate actual SNR
        actual_snr = self.calculate_snr(speech, noise * noise_scale)
        
        return mixed, actual_snr
    
    def test_with_noise_file(self, speech_path: str, noise_path: str, 
                            snr_db: float, output_name: str) -> dict:
        """
        Test pipeline with speech + noise file
        
        Args:
            speech_path: Path to clean speech audio file
            noise_path: Path to noise-only audio file
            snr_db: Target SNR in dB
            output_name: Name for output files
            
        Returns:
            Dictionary with test results
        """
        logger.info(f"Testing: {output_name} at {snr_db}dB SNR")
        
        try:
            # Load audio files
            speech, sr_speech = sf.read(speech_path)
            noise, sr_noise = sf.read(noise_path)
            
            # Ensure mono and same sample rate
            if len(speech.shape) > 1:
                speech = speech[:, 0]
            if len(noise.shape) > 1:
                noise = noise[:, 0]
            
            # Resample to 16kHz if needed
            try:
                import librosa
                if sr_speech != self.config.SAMPLE_RATE:
                    speech = librosa.resample(speech, orig_sr=sr_speech, target_sr=self.config.SAMPLE_RATE)
                if sr_noise != self.config.SAMPLE_RATE:
                    noise = librosa.resample(noise, orig_sr=sr_noise, target_sr=self.config.SAMPLE_RATE)
            except ImportError:
                if sr_speech != self.config.SAMPLE_RATE or sr_noise != self.config.SAMPLE_RATE:
                    logger.warning("librosa not available. Audio must be 16kHz. Skipping resampling.")
                    raise ValueError("Audio sample rate must be 16kHz when librosa is not available")
            
            # Normalize to [-1, 1]
            speech = speech.astype(np.float32)
            if speech.max() > 1.0 or speech.min() < -1.0:
                speech = speech / np.max(np.abs(speech))
            
            noise = noise.astype(np.float32)
            if noise.max() > 1.0 or noise.min() < -1.0:
                noise = noise / np.max(np.abs(noise))
            
            # Mix speech and noise
            mixed, actual_snr = self.mix_speech_noise(speech, noise, snr_db)
            
            # Calculate input metrics
            input_rms = self.calculate_rms(mixed)
            input_snr = self.calculate_snr(speech, mixed - speech)
            
            # Process through pipeline with latency measurement
            processed_chunks = []
            chunk_size = int(self.config.SAMPLE_RATE * 0.1)  # 100ms chunks
            latencies = []
            
            # Reset pipeline state
            self.pipeline.speech_state = SpeechState.SILENCE
            self.speech_events = []
            
            for i in range(0, len(mixed), chunk_size):
                chunk = mixed[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                
                # Measure latency per chunk
                processed, latency_ms = self.measure_latency(
                    self.pipeline.process_chunk, chunk
                )
                latencies.append(latency_ms)
                
                if processed is not None:
                    processed_chunks.append(processed)
            
            processing_time = sum(latencies) / 1000.0  # Total time in seconds
            avg_latency_ms = np.mean(latencies) if latencies else 0
            max_latency_ms = np.max(latencies) if latencies else 0
            
            if processed_chunks:
                processed = np.concatenate(processed_chunks)
            else:
                processed = np.array([], dtype=np.float32)
            
            # Calculate output metrics
            output_rms = self.calculate_rms(processed) if len(processed) > 0 else -float('inf')
            if len(processed) > 0 and len(speech) > 0:
                min_len = min(len(processed), len(speech))
                output_snr = self.calculate_snr(speech[:min_len], processed[:min_len] - speech[:min_len])
                snr_improvement = output_snr - input_snr
            else:
                output_snr = -float('inf')
                snr_improvement = 0
            
            # Check for clipping
            clipping_info = self.detect_clipping(processed) if len(processed) > 0 else {"has_clipping": False, "clipping_percentage": 0}
            
            # Count speech events
            speech_start_events = sum(1 for e in self.speech_events if e["type"] == "speech_start")
            speech_end_events = sum(1 for e in self.speech_events if e["type"] == "speech_end")
            
            # Save audio files for comparison
            output_dir = self.results_dir / output_name
            output_dir.mkdir(exist_ok=True)
            
            sf.write(output_dir / "01_original_speech.wav", speech, self.config.SAMPLE_RATE)
            sf.write(output_dir / "02_noise.wav", noise, self.config.SAMPLE_RATE)
            sf.write(output_dir / "03_mixed_input.wav", mixed, self.config.SAMPLE_RATE)
            if len(processed) > 0:
                sf.write(output_dir / "04_processed_output.wav", processed, self.config.SAMPLE_RATE)
            
            # Test VAD accuracy
            vad = VoiceActivityDetector(
                threshold=self.config.VAD_THRESHOLD,
                sample_rate=self.config.SAMPLE_RATE
            )
            
            # Test VAD on clean speech (should detect speech)
            speech_detected_clean = vad.is_speech(speech[:int(self.config.SAMPLE_RATE * 0.5)])
            
            # Test VAD on mixed audio (should detect speech)
            speech_detected_mixed = vad.is_speech(mixed[:int(self.config.SAMPLE_RATE * 0.5)])
            
            # Test VAD on noise only (should NOT detect speech)
            speech_detected_noise = vad.is_speech(noise[:int(self.config.SAMPLE_RATE * 0.5)])
            
            results = {
                "test_name": output_name,
                "target_snr_db": snr_db,
                "actual_snr_db": actual_snr,
                "input_rms_db": input_rms,
                "output_rms_db": output_rms,
                "input_snr_db": input_snr,
                "output_snr_db": output_snr,
                "snr_improvement_db": snr_improvement,
                "processing_time_sec": processing_time,
                "processing_speed_x_realtime": len(mixed) / self.config.SAMPLE_RATE / processing_time if processing_time > 0 else 0,
                "avg_latency_ms": avg_latency_ms,
                "max_latency_ms": max_latency_ms,
                "vad_speech_detected_clean": speech_detected_clean,
                "vad_speech_detected_mixed": speech_detected_mixed,
                "vad_speech_detected_noise": speech_detected_noise,
                "vad_accuracy": "✓" if speech_detected_clean and speech_detected_mixed and not speech_detected_noise else "✗",
                "clipping_detected": clipping_info["has_clipping"],
                "clipping_percentage": clipping_info["clipping_percentage"],
                "max_audio_level": clipping_info.get("max_level", 0),
                "speech_start_events": speech_start_events,
                "speech_end_events": speech_end_events,
                "output_files": str(output_dir)
            }
            
            logger.info(f"Results: SNR improvement = {snr_improvement:.2f}dB, "
                        f"Processing = {results['processing_speed_x_realtime']:.2f}x realtime, "
                        f"Latency = {avg_latency_ms:.2f}ms, "
                        f"Clipping = {clipping_info['clipping_percentage']:.2f}%")
            
            return results
            
        except Exception as e:
            logger.error(f"Error testing {output_name}: {e}")
            return {"test_name": output_name, "error": str(e)}
    
    def test_synthetic_noise(self, speech_path: str, noise_level_db: float = 85.0) -> dict:
        """
        Test with synthetic factory noise (white/pink noise at specified level)
        
        Args:
            speech_path: Path to clean speech
            noise_level_db: Noise level in dB (e.g., 85 for 85dB factory noise)
            
        Returns:
            Test results
        """
        logger.info(f"Testing with synthetic {noise_level_db}dB noise")
        
        try:
            # Load speech
            speech, sr = sf.read(speech_path)
            if len(speech.shape) > 1:
                speech = speech[:, 0]
            
            if sr != self.config.SAMPLE_RATE:
                import librosa
                speech = librosa.resample(speech, orig_sr=sr, target_sr=self.config.SAMPLE_RATE)
            
            speech = speech.astype(np.float32)
            if speech.max() > 1.0 or speech.min() < -1.0:
                speech = speech / np.max(np.abs(speech))
            
            # Generate synthetic noise (white noise)
            noise = np.random.normal(0, 1, len(speech)).astype(np.float32)
            
            # Scale noise to target dB level
            # Convert dB to linear scale
            noise_linear = 10 ** (noise_level_db / 20)
            # Normalize and scale
            noise = (noise / np.max(np.abs(noise))) * (noise_linear / 10.0)  # Approximate scaling
            
            # Mix at low SNR (simulating speech in noisy factory)
            mixed, actual_snr = self.mix_speech_noise(speech, noise, target_snr_db=5.0)
            
            # Process
            processed_chunks = []
            chunk_size = int(self.config.SAMPLE_RATE * 0.1)
            
            for i in range(0, len(mixed), chunk_size):
                chunk = mixed[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')
                
                processed = self.pipeline.process_chunk(chunk)
                if processed is not None:
                    processed_chunks.append(processed)
            
            processed = np.concatenate(processed_chunks) if processed_chunks else np.array([], dtype=np.float32)
            
            # Calculate metrics
            input_rms = self.calculate_rms(mixed)
            output_rms = self.calculate_rms(processed) if len(processed) > 0 else -float('inf')
            input_snr = self.calculate_snr(speech, mixed - speech)
            if len(processed) > 0:
                min_len = min(len(processed), len(speech))
                output_snr = self.calculate_snr(speech[:min_len], processed[:min_len] - speech[:min_len])
                snr_improvement = output_snr - input_snr
            else:
                output_snr = -float('inf')
                snr_improvement = 0
            
            # Save results
            output_dir = self.results_dir / f"synthetic_{noise_level_db}dB"
            output_dir.mkdir(exist_ok=True)
            
            sf.write(output_dir / "01_speech.wav", speech, self.config.SAMPLE_RATE)
            sf.write(output_dir / "02_noise.wav", noise, self.config.SAMPLE_RATE)
            sf.write(output_dir / "03_mixed.wav", mixed, self.config.SAMPLE_RATE)
            if len(processed) > 0:
                sf.write(output_dir / "04_processed.wav", processed, self.config.SAMPLE_RATE)
            
            results = {
                "test_name": f"synthetic_{noise_level_db}dB",
                "noise_level_db": noise_level_db,
                "input_rms_db": input_rms,
                "output_rms_db": output_rms,
                "input_snr_db": input_snr,
                "output_snr_db": output_snr,
                "snr_improvement_db": snr_improvement,
                "output_files": str(output_dir)
            }
            
            logger.info(f"Synthetic test: SNR improvement = {snr_improvement:.2f}dB")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in synthetic test: {e}")
            return {"test_name": f"synthetic_{noise_level_db}dB", "error": str(e)}
    
    def run_test_suite(self, speech_file: Optional[str] = None, noise_file: Optional[str] = None):
        """
        Run comprehensive test suite
        
        Args:
            speech_file: Path to clean speech file (optional, will use synthetic if not provided)
            noise_file: Path to noise file (optional, will use synthetic if not provided)
        """
        logger.info("=" * 60)
        logger.info("Factory Noise Test Suite - 85dB+ Noise Testing")
        logger.info("=" * 60)
        
        all_results = []
        
        # Test 0: False trigger test (if noise file provided)
        if noise_file and os.path.exists(noise_file):
            logger.info("\n[Test 0] False Trigger Rate Test")
            false_trigger_results = self.test_false_triggers(noise_file, duration_sec=10.0)
            all_results.append({"test_type": "false_triggers", **false_trigger_results})
        
        # Test 1: Synthetic 85dB noise
        logger.info("\n[Test 1] Synthetic 85dB Factory Noise")
        result1 = self.test_synthetic_noise(
            speech_path=speech_file or "test_audio/speech_sample.wav",
            noise_level_db=85.0
        )
        all_results.append(result1)
        
        # Test 2: Synthetic 90dB noise (extreme)
        logger.info("\n[Test 2] Synthetic 90dB Factory Noise (Extreme)")
        result2 = self.test_synthetic_noise(
            speech_path=speech_file or "test_audio/speech_sample.wav",
            noise_level_db=90.0
        )
        all_results.append(result2)
        
        # Test 3: Real noise file if provided
        if noise_file and os.path.exists(noise_file):
            logger.info("\n[Test 3] Real Factory Noise File")
            if speech_file and os.path.exists(speech_file):
                for snr in [0, 5, 10]:
                    result = self.test_with_noise_file(
                        speech_path=speech_file,
                        noise_path=noise_file,
                        snr_db=snr,
                        output_name=f"real_noise_snr{snr}db"
                    )
                    all_results.append(result)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        for result in all_results:
            if "error" not in result:
                test_name = result.get('test_name', result.get('test_type', 'Unknown'))
                logger.info(f"\n{test_name}:")
                
                if result.get('test_type') == 'false_triggers':
                    logger.info(f"  False Triggers: {result.get('false_triggers', 0)}")
                    logger.info(f"  False Trigger Rate: {result.get('false_trigger_rate_per_sec', 0):.2f}/sec")
                    logger.info(f"  Speech Detection Rate: {result.get('speech_detection_rate_percent', 0):.2f}%")
                else:
                    logger.info(f"  Input SNR: {result.get('input_snr_db', 'N/A'):.2f}dB")
                    logger.info(f"  Output SNR: {result.get('output_snr_db', 'N/A'):.2f}dB")
                    logger.info(f"  SNR Improvement: {result.get('snr_improvement_db', 0):.2f}dB")
                    logger.info(f"  Avg Latency: {result.get('avg_latency_ms', 'N/A'):.2f}ms")
                    logger.info(f"  Max Latency: {result.get('max_latency_ms', 'N/A'):.2f}ms")
                    logger.info(f"  Clipping: {result.get('clipping_percentage', 0):.2f}%")
                    logger.info(f"  VAD Accuracy: {result.get('vad_accuracy', 'N/A')}")
                    logger.info(f"  Speech Events: {result.get('speech_start_events', 0)} start, {result.get('speech_end_events', 0)} end")
                    logger.info(f"  Output files: {result.get('output_files', 'N/A')}")
            else:
                logger.error(f"{result.get('test_name', 'Unknown')}: ERROR - {result['error']}")
        
        # Save results to file
        import json
        results_file = self.results_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"\nFull results saved to: {results_file}")
        logger.info("=" * 60)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Factory Noise Test Suite')
    parser.add_argument('--speech', type=str, help='Path to clean speech audio file')
    parser.add_argument('--noise', type=str, help='Path to factory noise audio file')
    parser.add_argument('--snr', type=float, nargs='+', default=[0, 5, 10], 
                       help='SNR levels to test (default: 0, 5, 10)')
    
    args = parser.parse_args()
    
    tester = FactoryNoiseTester()
    
    if args.speech and args.noise:
        # Test with provided files
        for snr in args.snr:
            result = tester.test_with_noise_file(
                speech_path=args.speech,
                noise_path=args.noise,
                snr_db=snr,
                output_name=f"custom_test_snr{snr}db"
            )
    else:
        # Run default test suite
        tester.run_test_suite(
            speech_file=args.speech,
            noise_file=args.noise
        )

if __name__ == "__main__":
    main()
