"""
Unit tests for HFrEF ECG and PCG signal processing and feature extraction.
"""
import pytest
import numpy as np

from src.hfref.config import (
    HFREF_ECG_SAMPLE_RATE,
    HFREF_PCG_TARGET_SR,
    ECG_FEATURE_DIM,
    PCG_FEATURE_PER_LOC_DIM
)
from src.hfref.ecg_processing import (
    preprocess_ecg,
    detect_r_peaks,
    extract_ecg_features
)
from src.hfref.pcg_processing import (
    preprocess_hfref_pcg,
    compute_shannon_energy_envelope,
    extract_pcg_acoustic_features
)

def test_ecg_preprocessing_and_peak_detection():
    """Verify ECG preprocessing and R-peak detection on synthetic heartbeat signal."""
    fs = HFREF_ECG_SAMPLE_RATE # 500 Hz
    duration = 5.0 # 5 seconds
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Create synthetic ECG with pulses every 0.8s (75 BPM)
    ecg = np.zeros_like(t)
    heart_rate_sec = 0.8
    for beat_t in np.arange(0.2, duration, heart_rate_sec):
        idx = int(beat_t * fs)
        # R-peak pulse
        ecg[idx-5:idx+5] += np.hanning(10) * 2.0
    
    # Add small 50Hz noise and baseline drift
    ecg += 0.05 * np.sin(2 * np.pi * 50 * t)
    ecg += 0.2 * np.sin(2 * np.pi * 0.2 * t)

    proc = preprocess_ecg(ecg, orig_sr=fs, target_sr=fs)
    assert len(proc) == len(ecg)
    assert not np.isnan(proc).any()
    assert not np.isinf(proc).any()

    # Peak detection
    peaks = detect_r_peaks(proc, fs=fs)
    assert len(peaks) >= 4 # Expect ~5-6 peaks in 5 seconds at 75 BPM

def test_ecg_feature_extraction():
    """Verify 24-dimensional ECG feature extraction vector."""
    fs = HFREF_ECG_SAMPLE_RATE
    # 10s synthetic signal
    synthetic_ecg = np.random.randn(int(10.0 * fs)).astype(np.float32) * 0.1
    # Add regular pulses
    for i in range(1, 10):
        synthetic_ecg[i * fs] = 2.0

    feats = extract_ecg_features(synthetic_ecg, fs=fs)
    assert "feature_vector" in feats
    assert len(feats["feature_vector"]) == ECG_FEATURE_DIM
    assert not np.isnan(feats["feature_vector"]).any()
    assert feats["heart_rate_bpm"] >= 0.0
    assert feats["r_peaks_count"] >= 0

def test_pcg_preprocessing():
    """Verify PCG preprocessing, bandpass filtering, and peak normalization."""
    sr = 4000
    target_sr = HFREF_PCG_TARGET_SR # 2000
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False)
    # 50 Hz heart sound component + 1000 Hz high frequency noise
    pcg = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 1000 * t)

    proc = preprocess_hfref_pcg(pcg, orig_sr=sr, target_sr=target_sr, target_peak=0.95)
    assert len(proc) == int(3.0 * target_sr)
    assert np.max(np.abs(proc)) <= 0.9501
    assert not np.isnan(proc).any()

def test_pcg_shannon_envelope():
    """Verify Shannon energy envelope computation."""
    data = np.sin(np.linspace(0, 10, 1000))
    env = compute_shannon_energy_envelope(data, window_len=20)
    assert len(env) == len(data)
    assert (env >= 0.0).all()
    assert not np.isnan(env).any()

def test_pcg_feature_extraction():
    """Verify 48-dimensional PCG acoustic feature extraction vector."""
    sr = HFREF_PCG_TARGET_SR
    # 3s synthetic heart sound signal
    t = np.linspace(0, 3.0, 3 * sr, endpoint=False)
    synthetic_pcg = (0.5 * np.sin(2 * np.pi * 60 * t) + 0.1 * np.random.randn(len(t))).astype(np.float32)

    feats = extract_pcg_acoustic_features(synthetic_pcg, sr=sr)
    assert "feature_vector" in feats
    assert len(feats["feature_vector"]) == PCG_FEATURE_PER_LOC_DIM
    assert not np.isnan(feats["feature_vector"]).any()
    assert feats["rms_energy"] > 0.0
    assert feats["spectral_centroid_hz"] >= 0.0
