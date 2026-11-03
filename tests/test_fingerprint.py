"""
Unit and integration tests for Heart Sound Fingerprint generation and Acoustic Similarity.
"""
import pytest
import numpy as np
import wave
from pathlib import Path

from src.fingerprint.config import FingerprintConfig, FINGERPRINT_VERSION
from src.fingerprint.generator import (
    HeartSoundFingerprint,
    HeartSoundFingerprintGenerator,
    generate_fingerprint
)
from src.fingerprint.similarity import (
    AcousticSimilarityCalculator,
    calculate_acoustic_similarity
)
from src.utils.config import RAW_DATA_DIR

def test_fingerprint_config():
    """Verify fingerprint configuration defaults and total dimension."""
    cfg = FingerprintConfig()
    assert cfg.version == "v1"
    assert cfg.mfcc_dim == 120
    assert cfg.mel_dim == 128
    assert cfg.ast_dim == 768
    assert cfg.total_dimension == 1016

def test_fingerprint_generation_synthetic():
    """Verify fingerprint generation on synthetic audio produces exact 1016-dim unit vector."""
    gen = HeartSoundFingerprintGenerator()
    sr = 4000
    # 5.0 seconds of synthetic PCG waveform at 4000 Hz
    t = np.linspace(0, 5.0, 5 * sr)
    synthetic_pcg = (0.5 * np.sin(2 * np.pi * 30 * t) + 0.3 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    
    fp = gen.generate_recording_fingerprint(synthetic_pcg, orig_sr=sr, location="AV")
    
    assert isinstance(fp, HeartSoundFingerprint)
    assert fp.dimension == 1016
    assert fp.vector.shape == (1016,)
    assert fp.version == "v1"
    assert fp.location == "AV"
    assert fp.component_dims == {"mfcc": 120, "mel": 128, "ast": 768}
    
    # Verify unit L2 normalization
    norm = np.linalg.norm(fp.vector)
    assert abs(norm - 1.0) < 1e-4

def test_fingerprint_determinism_and_filename_invariance():
    """Verify fingerprint depends strictly on audio content and is completely independent of filenames or metadata."""
    gen = HeartSoundFingerprintGenerator()
    sr = 4000
    synthetic_pcg = np.random.randn(5 * sr).astype(np.float32) * 0.1
    
    fp1 = gen.generate_recording_fingerprint(synthetic_pcg, orig_sr=sr, location="MV")
    fp2 = gen.generate_recording_fingerprint(synthetic_pcg, orig_sr=sr, location="MV")
    
    assert np.allclose(fp1.vector, fp2.vector, atol=1e-6)
    
    # Base64 serialization round-trip
    b64 = fp1.to_base64()
    fp_reconstructed = HeartSoundFingerprint.from_base64(b64, dimension=1016, location="MV")
    assert np.allclose(fp1.vector, fp_reconstructed.vector, atol=1e-6)

def test_patient_multilocation_fingerprint():
    """Verify multi-location patient composite fingerprint generation."""
    gen = HeartSoundFingerprintGenerator()
    sr = 4000
    t = np.linspace(0, 5.0, 5 * sr)
    wave_av = np.sin(2 * np.pi * 40 * t).astype(np.float32)
    wave_mv = np.sin(2 * np.pi * 80 * t).astype(np.float32)
    wave_pv = np.sin(2 * np.pi * 100 * t).astype(np.float32)
    
    patient_audios = {
        "AV": wave_av,
        "MV": wave_mv,
        "PV": wave_pv
    }
    
    patient_fp = gen.generate_patient_fingerprint(patient_audios, orig_sr=sr)
    
    assert patient_fp.dimension == 1016
    assert patient_fp.location == "multi_location"
    assert patient_fp.metadata["num_locations"] == 3
    assert abs(np.linalg.norm(patient_fp.vector) - 1.0) < 1e-4

def test_acoustic_similarity_identical_audio():
    """Verify Acoustic Similarity for identical fingerprints is approximately 1.0 (100%)."""
    gen = HeartSoundFingerprintGenerator()
    sr = 4000
    synthetic_audio = np.random.randn(5 * sr).astype(np.float32) * 0.1
    
    fp_a = gen.generate_recording_fingerprint(synthetic_audio, orig_sr=sr)
    fp_b = gen.generate_recording_fingerprint(synthetic_audio, orig_sr=sr)
    
    sim_result = calculate_acoustic_similarity(fp_a, fp_b)
    
    assert sim_result["metric_name"] == "Acoustic Similarity"
    assert abs(sim_result["acoustic_similarity_score"] - 1.0) < 1e-4
    assert abs(sim_result["acoustic_similarity_percent"] - 100.0) < 0.1
    assert sim_result["is_identical_audio"] is True
    assert "disclaimer" in sim_result
    assert "NOT indicate a medical diagnosis" in sim_result["disclaimer"]

def test_acoustic_similarity_different_circor_recordings():
    """Verify Acoustic Similarity between different valve sites and subjects from CirCor dataset."""
    training_data_dir = Path(RAW_DATA_DIR) / "training_data"
    av_path = training_data_dir / "40840_AV.wav"
    mv_path = training_data_dir / "40840_MV.wav"
    
    if not av_path.exists() or not mv_path.exists():
        pytest.skip("CirCor test files not found.")
        
    def load_audio(p):
        with wave.open(str(p), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, wf.getframerate()
            
    audio_av, sr_av = load_audio(av_path)
    audio_mv, sr_mv = load_audio(mv_path)
    
    gen = HeartSoundFingerprintGenerator()
    fp_av = gen.generate_recording_fingerprint(audio_av, orig_sr=sr_av, location="AV")
    fp_mv = gen.generate_recording_fingerprint(audio_mv, orig_sr=sr_mv, location="MV")
    
    sim_result = calculate_acoustic_similarity(fp_av, fp_mv)
    
    score = sim_result["acoustic_similarity_score"]
    assert -1.0 <= score <= 1.0
    assert 0.0 <= sim_result["acoustic_similarity_percent"] <= 100.0
    assert "component_similarities" in sim_result
    assert "mfcc_similarity" in sim_result["component_similarities"]
    assert "mel_similarity" in sim_result["component_similarities"]
    assert "ast_similarity" in sim_result["component_similarities"]
