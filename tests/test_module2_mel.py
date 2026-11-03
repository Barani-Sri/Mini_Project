"""
Unit Tests for Module 2 Mel-Spectrogram Extraction.
"""
import pytest
import numpy as np

from src.features.config import FeatureConfig
from src.features.melspectrogram import MelSpectrogramExtractor

@pytest.fixture
def test_config():
    return FeatureConfig(
        sample_rate=2000,
        waveform_samples=10000,
        n_mels=128,
        frame_length_ms=30.0,
        hop_length_ms=10.0,
        f_min=25.0,
        f_max=400.0,
        n_fft=512
    )

def test_mel_shape_and_values(test_config):
    extractor = MelSpectrogramExtractor(test_config)
    waveform = np.random.randn(10000).astype(np.float32)
    
    log_mel = extractor.extract_log_mel(waveform)
    T = test_config.num_time_frames  # 501
    
    assert log_mel.shape == (128, T)
    assert not np.isnan(log_mel).any()
    assert not np.isinf(log_mel).any()

def test_mel_silent_signal(test_config):
    extractor = MelSpectrogramExtractor(test_config)
    zeros = np.zeros(10000, dtype=np.float32)
    
    log_mel = extractor.extract_log_mel(zeros)
    assert log_mel.shape == (128, test_config.num_time_frames)
    assert not np.isnan(log_mel).any()
    assert not np.isinf(log_mel).any()

def test_mel_standardization(test_config):
    extractor = MelSpectrogramExtractor(test_config)
    waveform = np.random.randn(10000).astype(np.float32)
    
    log_mel = extractor.extract_log_mel(waveform, standardize=True)
    assert np.isclose(np.mean(log_mel), 0.0, atol=1e-3)
    assert np.isclose(np.std(log_mel), 1.0, atol=1e-3)
