"""
Unit tests for PCG feature extraction routines.
"""
import pytest
import numpy as np
import pandas as pd

from src.audio.features import (
    extract_time_domain_features,
    extract_frequency_domain_features,
    extract_mfcc_features,
    extract_segment_murmur_features,
    extract_full_pcg_features
)
from src.utils.config import SAMPLE_RATE

@pytest.fixture
def sample_audio():
    sr = 4000
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    # Synthetic heart sound simulation
    audio = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.random.randn(len(t))
    return audio.astype(np.float32), sr

@pytest.fixture
def mock_segmentation():
    # Construct 2 mockup cardiac cycles
    # Cycle 1: S1 (0.0-0.1s), Systole (0.1-0.3s), S2 (0.3-0.4s), Diastole (0.4-0.8s)
    # Cycle 2: S1 (0.8-0.9s), Systole (0.9-1.1s), S2 (1.1-1.2s), Diastole (1.2-1.6s)
    rows = [
        {"start": 0.0, "end": 0.1, "state": 1},
        {"start": 0.1, "end": 0.3, "state": 2},
        {"start": 0.3, "end": 0.4, "state": 3},
        {"start": 0.4, "end": 0.8, "state": 4},
        {"start": 0.8, "end": 0.9, "state": 1},
        {"start": 0.9, "end": 1.1, "state": 2},
        {"start": 1.1, "end": 1.2, "state": 3},
        {"start": 1.2, "end": 1.6, "state": 4},
    ]
    df = pd.DataFrame(rows)
    df["duration"] = df["end"] - df["start"]
    return df

def test_time_domain_features(sample_audio):
    audio, _ = sample_audio
    feats = extract_time_domain_features(audio)
    
    assert isinstance(feats, dict)
    assert "time_rms" in feats
    assert "time_zcr" in feats
    assert "time_skewness" in feats
    assert "shannon_mean" in feats
    
    for k, v in feats.items():
        assert not np.isnan(v), f"Feature {k} is NaN"
        assert not np.isinf(v), f"Feature {k} is Inf"

def test_frequency_domain_features(sample_audio):
    audio, sr = sample_audio
    feats = extract_frequency_domain_features(audio, sr=sr)
    
    assert isinstance(feats, dict)
    assert "spec_total_power" in feats
    assert "spec_band1_rel" in feats
    assert "spec_band3_rel" in feats
    assert "spec_centroid" in feats
    assert "spec_flatness" in feats
    
    for k, v in feats.items():
        assert not np.isnan(v), f"Feature {k} is NaN"

def test_mfcc_features(sample_audio):
    audio, sr = sample_audio
    feats = extract_mfcc_features(audio, sr=sr, n_mfcc=13)
    
    assert isinstance(feats, dict)
    assert "mfcc_1_mean" in feats
    assert "mfcc_13_mean" in feats
    assert "delta_mfcc_1_mean" in feats
    
    for k, v in feats.items():
        assert not np.isnan(v), f"Feature {k} is NaN"

def test_segment_features(sample_audio, mock_segmentation):
    audio, sr = sample_audio
    feats = extract_segment_murmur_features(audio, mock_segmentation, sr=sr)
    
    assert isinstance(feats, dict)
    assert "seg_systole_energy_ratio" in feats
    assert "seg_sys_s1_rms_ratio" in feats
    assert "seg_num_cycles" in feats
    assert feats["seg_num_cycles"] == 2.0
    
    for k, v in feats.items():
        assert not np.isnan(v), f"Feature {k} is NaN"

def test_full_features_pipeline(sample_audio, mock_segmentation):
    audio, sr = sample_audio
    full_feats = extract_full_pcg_features(audio, seg_df=mock_segmentation, location="MV", sr=sr)
    
    assert isinstance(full_feats, dict)
    assert len(full_feats) > 40
    assert full_feats["loc_is_MV"] == 1.0
    assert full_feats["loc_is_AV"] == 0.0
    
    for k, v in full_feats.items():
        assert not np.isnan(v), f"Feature {k} is NaN"
