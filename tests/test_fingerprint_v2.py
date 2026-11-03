"""
Unit tests for Heart Sound Fingerprint V2 (Temporal-Aware).
Verifies configuration, dimensions, determinism, normalization, numerical stability,
real audio extraction, and checks that Fingerprint V1 remains completely intact.
"""
import numpy as np
import pytest

from src.fingerprint import (
    FingerprintConfig as FingerprintConfigV1,
    HeartSoundFingerprintGenerator as GeneratorV1,
    calculate_acoustic_similarity as calculate_similarity_v1
)
from src.fingerprint_v2.config import FingerprintV2Config, FINGERPRINT_V2_VERSION
from src.fingerprint_v2.fingerprint_v2_generator import (
    HeartSoundFingerprintV2,
    HeartSoundFingerprintV2Generator,
    generate_fingerprint_v2
)
from src.fingerprint_v2.fingerprint_v2_similarity import (
    AcousticSimilarityCalculatorV2,
    calculate_acoustic_similarity_v2
)
from src.data.dataset import CirCorDataset


def test_v2_config_dimensions():
    """Verifies V2 default configuration dimensionality (1,264 dims)."""
    cfg = FingerprintV2Config()
    assert cfg.version == "v2"
    assert cfg.temporal_stats == ("mean", "std")
    assert cfg.mfcc_dim == 240
    assert cfg.mel_dim == 256
    assert cfg.ast_dim == 768
    assert cfg.total_dimension == 1264


def test_v2_synthetic_generation_and_normalization():
    """Tests V2 fingerprint generation on synthetic signals with norm and NaN/Inf checks."""
    sr = 4000
    t = np.linspace(0, 5.0, 5 * sr, endpoint=False)
    # Synthetic heart-like sound (S1/S2 pulses)
    audio = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 120 * t)
    audio = audio.astype(np.float32)

    gen = HeartSoundFingerprintV2Generator()
    fp = gen.generate_recording_fingerprint(audio, orig_sr=sr, location="AV")

    assert isinstance(fp, HeartSoundFingerprintV2)
    assert fp.vector.shape == (1264,)
    assert fp.dimension == 1264
    assert fp.version == "v2"
    assert fp.location == "AV"
    assert not np.isnan(fp.vector).any()
    assert not np.isinf(fp.vector).any()

    # Verify L2 norm is ~1.0
    norm = float(np.linalg.norm(fp.vector))
    assert pytest.approx(norm, rel=1e-5) == 1.0


def test_v2_determinism_and_self_similarity():
    """Verifies that V2 extraction is 100% deterministic and produces exact 1.0 self-similarity."""
    sr = 4000
    t = np.linspace(0, 4.0, 4 * sr, endpoint=False)
    audio = (0.6 * np.sin(2 * np.pi * 70 * t)).astype(np.float32)

    gen = HeartSoundFingerprintV2Generator()
    fp1 = gen.generate_recording_fingerprint(audio, orig_sr=sr, location="MV")
    fp2 = gen.generate_recording_fingerprint(audio, orig_sr=sr, location="MV")

    np.testing.assert_allclose(fp1.vector, fp2.vector, rtol=1e-5, atol=1e-6)

    sim = calculate_acoustic_similarity_v2(fp1, fp2)
    assert sim["acoustic_similarity_score"] == 1.0
    assert sim["acoustic_similarity_percent"] == 100.0
    assert sim["is_identical_audio"] is True
    assert sim["component_similarities"]["mfcc_similarity"] == 1.0
    assert sim["component_similarities"]["mel_similarity"] == 1.0
    assert sim["component_similarities"]["ast_similarity"] == 1.0


def test_v2_real_circor_recording_extraction():
    """Tests V2 fingerprint generation on a real CirCor dataset recording."""
    dataset = CirCorDataset()
    rec_df = dataset.get_recordings_df()
    sample_row = rec_df.iloc[0]

    audio, sr = dataset.load_audio(sample_row["wav_path"])
    gen = HeartSoundFingerprintV2Generator()
    fp = gen.generate_recording_fingerprint(audio, orig_sr=sr, location=sample_row["location"])

    assert fp.dimension == 1264
    assert fp.vector.shape == (1264,)
    assert not np.isnan(fp.vector).any()
    assert not np.isinf(fp.vector).any()
    assert pytest.approx(float(np.linalg.norm(fp.vector)), rel=1e-5) == 1.0


def test_v1_remains_intact_and_unmodified():
    """Ensures that V1 generator and similarity behavior remain completely intact."""
    cfg_v1 = FingerprintConfigV1()
    assert cfg_v1.version == "v1"
    assert cfg_v1.total_dimension == 1016

    sr = 4000
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 80 * t)).astype(np.float32)

    gen_v1 = GeneratorV1()
    fp_v1 = gen_v1.generate_recording_fingerprint(audio, orig_sr=sr, location="PV")

    assert fp_v1.dimension == 1016
    assert fp_v1.vector.shape == (1016,)
    assert fp_v1.version == "v1"

    sim_v1 = calculate_similarity_v1(fp_v1, fp_v1)
    assert sim_v1["acoustic_similarity_score"] == 1.0
