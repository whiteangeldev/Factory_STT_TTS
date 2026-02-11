#!/usr/bin/env python3
"""
Test script to evaluate alternative TTS models for English TTS.
Compares MMS-TTS (current), Coqui TTS, Piper TTS, and Bark.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Test texts with various scenarios
TEST_TEXTS = {
    "numbers": "The GDP increased by 3.7% in 1924. TTS and AI are important technologies.",
}

# Output directory for test results
OUTPUT_DIR = Path("test_tts_models")
OUTPUT_DIR.mkdir(exist_ok=True)
AUDIO_DIR = OUTPUT_DIR / "audio_samples"
AUDIO_DIR.mkdir(exist_ok=True)

class TTSModelTester:
    """Base class for TTS model testing"""
    
    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.model = None
        self.load_time = 0
        self.sample_rate = None
        
    def is_available(self) -> bool:
        """Check if model dependencies are available"""
        return self.available
    
    def load_model(self) -> Tuple[bool, str]:
        """Load the TTS model. Returns (success, error_message)"""
        raise NotImplementedError
    
    def synthesize(self, text: str) -> Tuple[Optional[np.ndarray], Optional[int], float]:
        """
        Synthesize speech from text.
        Returns (audio_array, sample_rate, synthesis_time)
        """
        raise NotImplementedError
    
    def cleanup(self):
        """Clean up model resources"""
        pass


class MMS_TTS_Tester(TTSModelTester):
    """Test MMS-TTS (current model)"""
    
    def __init__(self):
        super().__init__("MMS-TTS")
        try:
            import torch
            from transformers import AutoProcessor, VitsModel
            self.torch = torch
            self.VitsModel = VitsModel
            self.AutoProcessor = AutoProcessor
            self.available = True
        except ImportError:
            logger.warning("MMS-TTS dependencies not available (torch, transformers)")
            self.available = False
    
    def load_model(self) -> Tuple[bool, str]:
        if not self.available:
            return False, "Dependencies not available"
        
        try:
            start_time = time.time()
            model_id = "facebook/mms-tts-eng"
            
            # Set device
            if self.torch.cuda.is_available():
                device = self.torch.device("cuda")
            elif self.torch.backends.mps.is_available():
                device = self.torch.device("mps")
            else:
                device = self.torch.device("cpu")
            
            logger.info(f"Loading MMS-TTS model on {device}...")
            self.model = self.VitsModel.from_pretrained(model_id).to(device)
            self.processor = self.AutoProcessor.from_pretrained(model_id)
            self.device = device
            self.sample_rate = getattr(self.model.config, "sampling_rate", 16000)
            
            self.load_time = time.time() - start_time
            logger.info(f"MMS-TTS loaded in {self.load_time:.2f}s")
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def synthesize(self, text: str) -> Tuple[Optional[np.ndarray], Optional[int], float]:
        try:
            start_time = time.time()
            inputs = self.processor(text=text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with self.torch.no_grad():
                outputs = self.model(**inputs)
            
            audio = outputs.waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)
            synthesis_time = time.time() - start_time
            
            return audio, self.sample_rate, synthesis_time
        except Exception as e:
            logger.error(f"MMS-TTS synthesis error: {e}")
            return None, None, 0.0
    
    def cleanup(self):
        if self.model:
            del self.model
            del self.processor
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()


class Coqui_TTS_Tester(TTSModelTester):
    """Test Coqui TTS (XTTS-v2)"""
    
    def __init__(self):
        super().__init__("Coqui TTS (XTTS-v2)")
        try:
            from TTS.api import TTS
            self.TTS = TTS
            self.available = True
        except ImportError:
            logger.warning("Coqui TTS not available. Install with: pip install TTS")
            self.available = False
    
    def load_model(self) -> Tuple[bool, str]:
        if not self.available:
            return False, "Dependencies not available"
        
        try:
            start_time = time.time()
            logger.info("Loading Coqui TTS XTTS-v2 model...")
            # XTTS-v2 is multilingual and high quality
            self.model = self.TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
            self.sample_rate = 24000  # XTTS-v2 uses 24kHz
            self.load_time = time.time() - start_time
            logger.info(f"Coqui TTS loaded in {self.load_time:.2f}s")
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def synthesize(self, text: str) -> Tuple[Optional[np.ndarray], Optional[int], float]:
        try:
            import soundfile as sf
            import tempfile
            
            start_time = time.time()
            
            # Coqui TTS writes to file, so we use a temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            try:
                self.model.tts_to_file(
                    text=text,
                    file_path=tmp_path,
                    language="en"
                )
                
                # Read the generated audio
                audio, sr = sf.read(tmp_path)
                synthesis_time = time.time() - start_time
                
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                # Normalize to float32
                audio = audio.astype(np.float32)
                
                return audio, sr, synthesis_time
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception as e:
            logger.error(f"Coqui TTS synthesis error: {e}")
            return None, None, 0.0
    
    def cleanup(self):
        if self.model:
            del self.model


class Piper_TTS_Tester(TTSModelTester):
    """Test Piper TTS"""
    
    def __init__(self):
        super().__init__("Piper TTS")
        try:
            from piper import PiperVoice
            from piper.download import ensure_voice_exists, find_voice
            self.PiperVoice = PiperVoice
            self.ensure_voice_exists = ensure_voice_exists
            self.find_voice = find_voice
            self.available = True
        except ImportError:
            logger.warning("Piper TTS not available. Install with: pip install piper-tts")
            self.available = False
    
    def load_model(self) -> Tuple[bool, str]:
        if not self.available:
            return False, "Dependencies not available"
        
        try:
            start_time = time.time()
            logger.info("Loading Piper TTS model...")
            
            # Use a high-quality English voice
            voice_name = "en_US-lessac-medium"
            self.ensure_voice_exists(voice_name, [])
            voice_path = self.find_voice(voice_name)
            
            self.model = self.PiperVoice.load(voice_path)
            self.sample_rate = 22050  # Piper typically uses 22.05kHz
            self.load_time = time.time() - start_time
            logger.info(f"Piper TTS loaded in {self.load_time:.2f}s")
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def synthesize(self, text: str) -> Tuple[Optional[np.ndarray], Optional[int], float]:
        try:
            start_time = time.time()
            audio_bytes = self.model.synthesize(text)
            synthesis_time = time.time() - start_time
            
            # Convert bytes to numpy array
            import wave
            import io
            
            audio_io = io.BytesIO(audio_bytes)
            with wave.open(audio_io, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                audio_bytes = wav_file.readframes(frames)
                audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            return audio, sample_rate, synthesis_time
        except Exception as e:
            logger.error(f"Piper TTS synthesis error: {e}")
            return None, None, 0.0
    
    def cleanup(self):
        if self.model:
            del self.model


class Bark_TTS_Tester(TTSModelTester):
    """Test Bark TTS (for natural prosody)"""
    
    def __init__(self):
        super().__init__("Bark TTS")
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            self.SAMPLE_RATE = SAMPLE_RATE
            self.generate_audio = generate_audio
            self.preload_models = preload_models
            self.available = True
        except ImportError:
            logger.warning("Bark TTS not available. Install with: pip install bark")
            self.available = False
    
    def load_model(self) -> Tuple[bool, str]:
        if not self.available:
            return False, "Dependencies not available"
        
        try:
            start_time = time.time()
            logger.info("Loading Bark TTS model (this may take a while)...")
            self.preload_models()
            self.sample_rate = self.SAMPLE_RATE
            self.load_time = time.time() - start_time
            logger.info(f"Bark TTS loaded in {self.load_time:.2f}s")
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def synthesize(self, text: str) -> Tuple[Optional[np.ndarray], Optional[int], float]:
        try:
            start_time = time.time()
            # Bark uses special prompt format
            prompt = f"[speaker] {text}"
            audio_array = self.generate_audio(prompt)
            synthesis_time = time.time() - start_time
            
            # Convert to float32
            audio = audio_array.astype(np.float32)
            
            return audio, self.sample_rate, synthesis_time
        except Exception as e:
            logger.error(f"Bark TTS synthesis error: {e}")
            return None, None, 0.0
    
    def cleanup(self):
        # Bark doesn't need explicit cleanup
        pass


def save_audio(audio: np.ndarray, sample_rate: int, filename: str):
    """Save audio to WAV file"""
    try:
        import soundfile as sf
        sf.write(filename, audio, sample_rate)
    except Exception as e:
        logger.error(f"Error saving audio to {filename}: {e}")


def run_tests():
    """Run comprehensive TTS model tests"""
    
    # Initialize testers - Coqui and Piper before Bark
    testers = [
        MMS_TTS_Tester(),
        Coqui_TTS_Tester(),
        Piper_TTS_Tester(),
        # Bark_TTS_Tester(),  # Test Bark last
    ]
    
    # Filter to only available models
    available_testers = [tester for tester in testers if tester.is_available()]
    
    if not available_testers:
        logger.error("No TTS models available for testing!")
        logger.info("Install at least one:")
        logger.info("  - MMS-TTS: pip install torch transformers")
        logger.info("  - Coqui TTS: pip install TTS")
        logger.info("  - Piper TTS: pip install piper-tts")
        logger.info("  - Bark: pip install bark")
        return
    
    logger.info(f"Testing {len(available_testers)} TTS models...")
    
    results = {}
    
    # Test each model
    for tester in available_testers:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {tester.name}")
        logger.info(f"{'='*60}")
        
        # Load model
        success, error = tester.load_model()
        if not success:
            logger.error(f"Failed to load {tester.name}: {error}")
            results[tester.name] = {"status": "failed", "error": error}
            # Generate immediate report for failed model
            generate_immediate_report(tester.name, {"status": "failed", "error": error})
            continue
        
        model_results = {
            "status": "success",
            "load_time": tester.load_time,
            "sample_rate": tester.sample_rate,
            "tests": {}
        }
        
        # Test each text sample
        for test_name, test_text in TEST_TEXTS.items():
            logger.info(f"\nTest: {test_name}")
            logger.info(f"Text: {test_text[:60]}...")
            
            audio, sr, synth_time = tester.synthesize(test_text)
            
            if audio is not None:
                # Calculate metrics
                duration = len(audio) / sr if sr else 0
                real_time_factor = duration / synth_time if synth_time > 0 else 0
                
                # Save audio sample
                audio_filename = AUDIO_DIR / f"{tester.name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}_{test_name}.wav"
                save_audio(audio, sr, str(audio_filename))
                
                model_results["tests"][test_name] = {
                    "text": test_text,
                    "synthesis_time": synth_time,
                    "audio_duration": duration,
                    "real_time_factor": real_time_factor,
                    "audio_file": str(audio_filename),
                    "success": True
                }
                
                logger.info(f"  ✓ Synthesized in {synth_time:.3f}s")
                logger.info(f"  ✓ Audio duration: {duration:.3f}s")
                logger.info(f"  ✓ Real-time factor: {real_time_factor:.2f}x")
            else:
                model_results["tests"][test_name] = {
                    "success": False,
                    "error": "Synthesis failed"
                }
                logger.error(f"  ✗ Synthesis failed")
        
        results[tester.name] = model_results
        
        # Generate immediate report for this model
        generate_immediate_report(tester.name, model_results)
        
        # Cleanup
        tester.cleanup()
    
    # Generate final comparison report
    generate_report(results)
    
    logger.info(f"\n{'='*60}")
    logger.info("Testing complete!")
    logger.info(f"Results saved to: {OUTPUT_DIR}")
    logger.info(f"Audio samples saved to: {AUDIO_DIR}")


def generate_immediate_report(model_name: str, model_data: Dict):
    """Generate immediate report for a single model"""
    logger.info(f"\n{'='*60}")
    logger.info(f"IMMEDIATE RESULTS: {model_name}")
    logger.info(f"{'='*60}")
    
    if model_data.get("status") == "failed":
        logger.error(f"  ✗ Failed: {model_data.get('error', 'Unknown error')}")
        return
    
    logger.info(f"  Load time: {model_data.get('load_time', 0):.2f}s")
    logger.info(f"  Sample rate: {model_data.get('sample_rate', 0)} Hz")
    
    tests = model_data.get("tests", {})
    for test_name, test_data in tests.items():
        if test_data.get("success"):
            logger.info(f"\n  Test: {test_name}")
            logger.info(f"    Synthesis time: {test_data['synthesis_time']:.3f}s")
            logger.info(f"    Audio duration: {test_data['audio_duration']:.3f}s")
            logger.info(f"    Real-time factor: {test_data['real_time_factor']:.2f}x")
            logger.info(f"    Audio file: {test_data['audio_file']}")
        else:
            logger.error(f"  Test {test_name}: Failed - {test_data.get('error', 'Unknown error')}")
    
    logger.info(f"{'='*60}\n")


def generate_immediate_report(model_name: str, model_data: Dict):
    """Generate immediate report for a single model"""
    logger.info(f"\n{'='*60}")
    logger.info(f"IMMEDIATE RESULTS: {model_name}")
    logger.info(f"{'='*60}")
    
    if model_data.get("status") == "failed":
        logger.error(f"  ✗ Failed: {model_data.get('error', 'Unknown error')}")
        return
    
    logger.info(f"  Load time: {model_data.get('load_time', 0):.2f}s")
    logger.info(f"  Sample rate: {model_data.get('sample_rate', 0)} Hz")
    
    tests = model_data.get("tests", {})
    for test_name, test_data in tests.items():
        if test_data.get("success"):
            logger.info(f"\n  Test: {test_name}")
            logger.info(f"    Synthesis time: {test_data['synthesis_time']:.3f}s")
            logger.info(f"    Audio duration: {test_data['audio_duration']:.3f}s")
            logger.info(f"    Real-time factor: {test_data['real_time_factor']:.2f}x")
            logger.info(f"    Audio file: {test_data['audio_file']}")
        else:
            logger.error(f"  Test {test_name}: Failed - {test_data.get('error', 'Unknown error')}")
    
    logger.info(f"{'='*60}\n")


def generate_report(results: Dict):
    """Generate comparison report"""
    
    report = {
        "summary": {},
        "detailed_results": results,
        "recommendations": []
    }
    
    # Calculate summary statistics
    for model_name, model_data in results.items():
        if model_data.get("status") != "success":
            continue
        
        tests = model_data.get("tests", {})
        successful_tests = [t for t in tests.values() if t.get("success")]
        
        if successful_tests:
            avg_synth_time = np.mean([t["synthesis_time"] for t in successful_tests])
            avg_rtf = np.mean([t["real_time_factor"] for t in successful_tests])
            
            report["summary"][model_name] = {
                "load_time": model_data["load_time"],
                "avg_synthesis_time": avg_synth_time,
                "avg_real_time_factor": avg_rtf,
                "sample_rate": model_data["sample_rate"],
                "successful_tests": len(successful_tests),
                "total_tests": len(tests)
            }
    
    # Generate recommendations
    if len(report["summary"]) > 1:
        # Find fastest model
        fastest = min(report["summary"].items(), 
                     key=lambda x: x[1]["avg_synthesis_time"])
        report["recommendations"].append(
            f"Fastest model: {fastest[0]} ({fastest[1]['avg_synthesis_time']:.3f}s avg)"
        )
        
        # Find best RTF
        best_rtf = max(report["summary"].items(),
                      key=lambda x: x[1]["avg_real_time_factor"])
        report["recommendations"].append(
            f"Best real-time factor: {best_rtf[0]} ({best_rtf[1]['avg_real_time_factor']:.2f}x)"
        )
    
    # Save report
    report_file = OUTPUT_DIR / "test_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for model_name, summary in report["summary"].items():
        logger.info(f"\n{model_name}:")
        logger.info(f"  Load time: {summary['load_time']:.2f}s")
        logger.info(f"  Avg synthesis time: {summary['avg_synthesis_time']:.3f}s")
        logger.info(f"  Avg real-time factor: {summary['avg_real_time_factor']:.2f}x")
        logger.info(f"  Sample rate: {summary['sample_rate']} Hz")
        logger.info(f"  Success rate: {summary['successful_tests']}/{summary['total_tests']}")
    
    if report["recommendations"]:
        logger.info("\nRecommendations:")
        for rec in report["recommendations"]:
            logger.info(f"  - {rec}")
    
    logger.info(f"\nFull report saved to: {report_file}")


if __name__ == "__main__":
    logger.info("TTS Model Comparison Test")
    logger.info("="*60)
    logger.info("This script will test multiple TTS models and compare:")
    logger.info("  - Synthesis speed")
    logger.info("  - Audio quality")
    logger.info("  - Real-time factor")
    logger.info("  - Model loading time")
    logger.info("")
    logger.info("Test text:")
    for name, text in TEST_TEXTS.items():
        logger.info(f"  - {name}: {text}")
    logger.info("")
    
    input("Press Enter to start testing...")
    
    run_tests()
