"""
Unit Tests for Module 2 Feature Quality Assessment.
"""
import pytest
import numpy as np

from src.features.config import FeatureConfig
from src.features.quality import FeatureQualityAssessment

@pytest.fixture
def quality_assessor():
    config = FeatureConfig()
    return FeatureQualityAssessment(config)

def test_clean_signal_quality(quality_assessor):
    # Good synthetic heart sound
    t = np.linspace(0, 5.0, 10000)
    waveform = (0.5 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    
    metrics = quality_assessor.assess(waveform)
    assert 0.0 <= metrics.quality_score <= 1.0
    assert metrics.quality_flag in ["GOOD", "QUESTIONABLE", "POOR"]
    assert metrics.has_nan_inf is False
    assert metrics.clipping_ratio == 0.0

def test_clipped_signal_quality(quality_assessor):
    # Severely clipped signal
    clipped_wave = np.ones(10000, dtype=np.float32) * 0.95
    metrics = quality_assessor.assess(clipped_wave)
    
    assert metrics.clipping_ratio > 0.90
    assert metrics.quality_score < 0.90

def test_silent_signal_quality(quality_assessor):
    zeros = np.zeros(10000, dtype=np.float32)
    metrics = quality_assessor.assess(zeros)
    
    assert metrics.near_zero_ratio == 1.0
    assert metrics.quality_flag == "POOR"
    assert metrics.quality_score < 0.50

def test_nan_signal_quality(quality_assessor):
    dirty = np.random.randn(10000).astype(np.float32)
    dirty[500] = np.nan
    metrics = quality_assessor.assess(dirty)
    
    assert metrics.has_nan_inf is True
    assert metrics.quality_score <= 0.50
