"""
Unit Tests for Module 2 MFCC, Delta, and Delta-Delta Extraction.
"""
import pytest
import numpy as np

from src.features.config import FeatureConfig
from src.features.mfcc import MFCCExtractor

@pytest.fixture
def test_config():
    return FeatureConfig(
        sample_rate=2000,
        waveform_samples=10000,
        n_mfcc=40,
        n_mels=128,
        frame_length_ms=30.0,
        hop_length_ms=10.0,
        f_min=25.0,
        f_max=400.0,
        n_fft=512
    )

@pytest.fixture
def sample_waveform():
    sr = 2000
    t = np.linspace(0, 5.0, 10000, endpoint=False)
    # Synthetic heart sound simulation
    return (0.8 * np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)

def test_mfcc_shapes(test_config, sample_waveform):
    extractor = MFCCExtractor(test_config)
    mfcc, delta, delta2, combined = extractor.extract_all(sample_waveform)

    T = test_config.num_time_frames  # 501
    assert mfcc.shape == (40, T)
    assert delta.shape == (40, T)
    assert delta2.shape == (40, T)
    assert combined.shape == (120, T)
    assert not np.isnan(combined).any()
    assert not np.isinf(combined).any()

def test_mfcc_silent_signal(test_config):
    extractor = MFCCExtractor(test_config)
    zeros = np.zeros(10000, dtype=np.float32)
    mfcc, delta, delta2, combined = extractor.extract_all(zeros)

    T = test_config.num_time_frames
    assert combined.shape == (120, T)
    assert not np.isnan(combined).any()
    assert not np.isinf(combined).any()

def test_mfcc_near_zero_signal(test_config):
    extractor = MFCCExtractor(test_config)
    near_zero = np.full(10000, 1e-7, dtype=np.float32)
    mfcc, delta, delta2, combined = extractor.extract_all(near_zero)

    T = test_config.num_time_frames
    assert combined.shape == (120, T)
    assert not np.isnan(combined).any()

def test_mfcc_nan_inf_sanitization(test_config):
    extractor = MFCCExtractor(test_config)
    dirty_waveform = np.random.randn(10000).astype(np.float32)
    dirty_waveform[100:150] = np.nan
    dirty_waveform[500:550] = np.inf

    _, _, _, combined = extractor.extract_all(dirty_waveform)
    assert not np.isnan(combined).any()
    assert not np.isinf(combined).any()
