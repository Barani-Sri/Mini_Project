"""
Unit tests for PCG audio signal processing and preprocessing routines.
"""
import pytest
import numpy as np

from src.audio.preprocessing import (
    butter_bandpass_filter,
    remove_spikes,
    normalize_audio,
    resample_pcg,
    compute_shannon_energy,
    compute_hilbert_envelope,
    preprocess_pcg_recording
)

@pytest.fixture
def synthetic_pcg():
    sr = 4000
    duration = 2.0  # 2 seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # 5 Hz baseline wander + 50 Hz heart sound + 150 Hz murmur + 1000 Hz noise
    signal = (
        0.5 * np.sin(2 * np.pi * 5 * t) +       # Baseline drift
        1.0 * np.sin(2 * np.pi * 50 * t) +      # Heart sound fundamental
        0.4 * np.sin(2 * np.pi * 150 * t) +     # Murmur frequency
        0.3 * np.sin(2 * np.pi * 1000 * t)      # High frequency noise
    ).astype(np.float32)
    return signal, sr

def test_butter_bandpass_filter(synthetic_pcg):
    audio, sr = synthetic_pcg
    filtered = butter_bandpass_filter(audio, sr=sr, lowcut=25.0, highcut=400.0)
    
    assert isinstance(filtered, np.ndarray)
    assert len(filtered) == len(audio)
    assert not np.isnan(filtered).any()
    
    # In frequency domain, 5 Hz baseline and 1000 Hz noise should be significantly attenuated
    fft_orig = np.abs(np.fft.rfft(audio))
    fft_filt = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(audio), d=1.0/sr)
    
    # 5 Hz attenuation
    idx_5hz = np.argmin(np.abs(freqs - 5))
    assert fft_filt[idx_5hz] < 0.1 * fft_orig[idx_5hz]
    
    # 1000 Hz attenuation
    idx_1000hz = np.argmin(np.abs(freqs - 1000))
    assert fft_filt[idx_1000hz] < 0.1 * fft_orig[idx_1000hz]
    
    # 150 Hz passband preservation
    idx_150hz = np.argmin(np.abs(freqs - 150))
    assert fft_filt[idx_150hz] > 0.8 * fft_orig[idx_150hz]

def test_remove_spikes():
    sr = 4000
    audio = np.random.randn(sr).astype(np.float32) * 0.1
    # Add huge motion artifact spike
    audio[1000] = 50.0
    audio[1001] = -40.0
    
    cleaned = remove_spikes(audio, sr=sr)
    assert np.max(np.abs(cleaned)) < 10.0
    assert not np.isnan(cleaned).any()

def test_normalize_audio(synthetic_pcg):
    audio, _ = synthetic_pcg
    
    # Z-score normalization
    z_norm = normalize_audio(audio, method="zscore")
    assert np.isclose(np.mean(z_norm), 0.0, atol=1e-3)
    assert np.isclose(np.std(z_norm), 1.0, atol=1e-3)
    
    # Peak normalization
    p_norm = normalize_audio(audio, method="peak", target_peak=1.0)
    assert np.isclose(np.max(np.abs(p_norm)), 1.0, atol=1e-4)
    
    p_norm_default = normalize_audio(audio, method="peak")
    assert np.isclose(np.max(np.abs(p_norm_default)), 0.95, atol=1e-4)

def test_resample_pcg(synthetic_pcg):
    audio, orig_sr = synthetic_pcg
    target_sr = 2000
    resampled, new_sr = resample_pcg(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    assert new_sr == target_sr
    expected_len = int(len(audio) * (target_sr / orig_sr))
    assert np.abs(len(resampled) - expected_len) <= 2

def test_shannon_and_hilbert_envelopes(synthetic_pcg):
    audio, sr = synthetic_pcg
    
    shannon = compute_shannon_energy(audio, sr=sr)
    assert len(shannon) == len(audio)
    assert not np.isnan(shannon).any()
    
    hilbert = compute_hilbert_envelope(audio)
    assert len(hilbert) == len(audio)
    assert (hilbert >= 0).all()
