"""
Digital signal processing and preprocessing routines for PCG audio signals.
Provides validation, resampling, Butterworth bandpass filtering (25-400Hz, order 5),
optional notch filtering (50Hz/60Hz), peak/z-score normalization, and fixed-length windowing.
"""
from typing import Optional, Tuple, Union, List
import numpy as np
import scipy.signal as signal

from src.utils.config import (
    SAMPLE_RATE,
    MODULE1_TARGET_SAMPLE_RATE,
    MODULE1_BANDPASS_LOW_HZ,
    MODULE1_BANDPASS_HIGH_HZ,
    MODULE1_BANDPASS_ORDER,
    MODULE1_NOTCH_ENABLED,
    MODULE1_NOTCH_FREQ,
    MODULE1_NOTCH_Q,
    MODULE1_NORM_METHOD,
    MODULE1_TARGET_PEAK,
    MODULE1_WINDOW_SEC,
    MODULE1_HOP_SEC,
    MODULE1_WINDOW_SAMPLES,
    MODULE1_HOP_SAMPLES
)

def resample_pcg(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = MODULE1_TARGET_SAMPLE_RATE
) -> Tuple[np.ndarray, int]:
    """
    Resamples PCG signal to target sampling rate using polyphase filtering.
    Does NOT resample if target_sr == orig_sr.
    """
    if orig_sr == target_sr or len(audio) == 0:
        return audio.astype(np.float32), orig_sr
        
    from math import gcd
    g = gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    
    resampled = signal.resample_poly(audio, up, down)
    return resampled.astype(np.float32), target_sr

def butter_bandpass_filter(
    audio: np.ndarray,
    sr: int = MODULE1_TARGET_SAMPLE_RATE,
    lowcut: float = MODULE1_BANDPASS_LOW_HZ,
    highcut: float = MODULE1_BANDPASS_HIGH_HZ,
    order: int = MODULE1_BANDPASS_ORDER
) -> np.ndarray:
    """
    Applies a zero-phase Butterworth bandpass filter (25-400Hz, default order 5) using SOS.
    Removes low-frequency baseline drift (<25Hz) and high-frequency noise (>400Hz).
    Preserves exact waveform length and safely handles short signals.
    """
    if len(audio) == 0:
        return audio
        
    nyquist = 0.5 * sr
    low = max(0.001, lowcut / nyquist)
    high = min(0.999, highcut / nyquist)
    
    sos = signal.butter(order, [low, high], btype="bandpass", output="sos")
    
    # Handle short signals where padlen might exceed signal length
    min_len = 3 * (2 * order + 1)
    if len(audio) < min_len:
        pad_width = min_len - len(audio)
        padded = np.pad(audio, (0, pad_width), mode="reflect")
        filtered_padded = signal.sosfiltfilt(sos, padded)
        return filtered_padded[:len(audio)].astype(np.float32)
        
    filtered = signal.sosfiltfilt(sos, audio)
    return filtered.astype(np.float32)

def apply_notch_filter(
    audio: np.ndarray,
    sr: int = MODULE1_TARGET_SAMPLE_RATE,
    freq: float = MODULE1_NOTCH_FREQ,
    q: float = MODULE1_NOTCH_Q
) -> np.ndarray:
    """
    Applies an optional notch filter (e.g. 50Hz or 60Hz power-line frequency).
    """
    if len(audio) == 0:
        return audio
        
    nyquist = 0.5 * sr
    if freq >= nyquist or freq <= 0:
        return audio
        
    w0 = freq / nyquist
    b, a = signal.iirnotch(w0, q)
    
    min_len = max(len(b), len(a)) * 3
    if len(audio) < min_len:
        pad_width = min_len - len(audio)
        padded = np.pad(audio, (0, pad_width), mode="reflect")
        filtered_padded = signal.filtfilt(b, a, padded)
        return filtered_padded[:len(audio)].astype(np.float32)
        
    filtered = signal.filtfilt(b, a, audio)
    return filtered.astype(np.float32)

def remove_spikes(
    audio: np.ndarray,
    sr: int = MODULE1_TARGET_SAMPLE_RATE,
    window_sec: float = 0.5,
    threshold_factor: float = 3.0
) -> np.ndarray:
    """
    Attenuates sharp acoustic motion spikes/artifacts (e.g. stethoscope rubbing)
    using windowed energy thresholding.
    """
    window_len = int(window_sec * sr)
    if window_len < 2 or len(audio) < window_len:
        return audio
        
    abs_audio = np.abs(audio)
    median_val = np.median(abs_audio)
    std_val = np.std(abs_audio)
    spike_threshold = median_val + threshold_factor * std_val
    
    cleaned = np.clip(audio, -spike_threshold, spike_threshold)
    return cleaned.astype(np.float32)

def normalize_audio(
    audio: np.ndarray,
    method: str = MODULE1_NORM_METHOD,
    target_peak: float = MODULE1_TARGET_PEAK,
    eps: float = 1e-8
) -> np.ndarray:
    """
    Normalizes audio signal.
    Methods:
    - 'peak' (default): scales maximum absolute amplitude to target_peak (0.95)
    - 'zscore': zero mean and unit variance
    - 'minmax': scales to [-1.0, 1.0]
    Safely handles zero-valued signals without division-by-zero.
    """
    if len(audio) == 0:
        return audio
        
    peak_val = np.max(np.abs(audio))
    if peak_val < eps:
        # Zero-valued or near-zero signal
        return np.zeros_like(audio, dtype=np.float32)
        
    if method == "peak":
        norm = (audio / (peak_val + eps)) * target_peak
        return norm.astype(np.float32)
    elif method == "zscore":
        mean = np.mean(audio)
        std = np.std(audio)
        if std < eps:
            return np.zeros_like(audio, dtype=np.float32)
        return ((audio - mean) / (std + eps)).astype(np.float32)
    elif method == "minmax":
        min_v, max_v = np.min(audio), np.max(audio)
        if max_v - min_v > eps:
            return (2.0 * (audio - min_v) / (max_v - min_v) - 1.0).astype(np.float32)
        return np.zeros_like(audio, dtype=np.float32)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def segment_and_pad_waveform(
    audio: np.ndarray,
    sr: int = MODULE1_TARGET_SAMPLE_RATE,
    window_sec: float = MODULE1_WINDOW_SEC,
    hop_sec: float = MODULE1_HOP_SEC
) -> np.ndarray:
    """
    Generates fixed-length waveform windows of shape (N, window_samples).
    - If recording < window_sec (e.g. <5s): zero-pad right to exactly window_samples.
    - If recording >= window_sec: sliding window with hop_sec step.
    - If trailing segment exists: zero-pad right to window_samples.
    Returns:
    - windows: np.ndarray of shape (N_windows, window_samples)
    """
    win_samples = int(window_sec * sr)
    hop_samples = int(hop_sec * sr)
    
    if len(audio) == 0:
        return np.empty((0, win_samples), dtype=np.float32)
        
    if len(audio) < win_samples:
        padded = np.zeros(win_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        return np.expand_dims(padded, axis=0)
        
    windows = []
    start = 0
    while start + win_samples <= len(audio):
        w = audio[start : start + win_samples]
        windows.append(w.astype(np.float32))
        start += hop_samples
        
    # Check if there is remaining unsegmented tail
    if start < len(audio) and (start + hop_samples < len(audio) or len(windows) == 0):
        tail = audio[start:]
        padded_tail = np.zeros(win_samples, dtype=np.float32)
        padded_tail[:len(tail)] = tail
        windows.append(padded_tail)
        
    return np.array(windows, dtype=np.float32)

def compute_shannon_energy(
    audio: np.ndarray,
    sr: int = MODULE1_TARGET_SAMPLE_RATE,
    smooth_ms: float = 20.0,
    eps: float = 1e-12
) -> np.ndarray:
    """Computes normalized average Shannon energy envelope."""
    peak = np.max(np.abs(audio))
    x = audio / (peak + 1e-8) if peak > 1e-8 else audio
    x_sq = x ** 2
    shannon = -x_sq * np.log(x_sq + eps)
    
    win_size = int((smooth_ms / 1000.0) * sr)
    if win_size > 1:
        window = np.ones(win_size) / win_size
        shannon = np.convolve(shannon, window, mode="same")
        
    std_s = np.std(shannon)
    if std_s > 1e-8:
        shannon = (shannon - np.mean(shannon)) / std_s
    return shannon.astype(np.float32)

def compute_hilbert_envelope(audio: np.ndarray) -> np.ndarray:
    """Extracts instantaneous amplitude envelope via Hilbert transform."""
    if len(audio) == 0:
        return audio
    analytic = signal.hilbert(audio)
    return np.abs(analytic).astype(np.float32)

def preprocess_module1_audio(
    audio: np.ndarray,
    orig_sr: int = SAMPLE_RATE,
    target_sr: int = MODULE1_TARGET_SAMPLE_RATE,
    bandpass_low: float = MODULE1_BANDPASS_LOW_HZ,
    bandpass_high: float = MODULE1_BANDPASS_HIGH_HZ,
    bandpass_order: int = MODULE1_BANDPASS_ORDER,
    notch_enabled: bool = MODULE1_NOTCH_ENABLED,
    notch_freq: float = MODULE1_NOTCH_FREQ,
    notch_q: float = MODULE1_NOTCH_Q,
    norm_method: str = MODULE1_NORM_METHOD,
    target_peak: float = MODULE1_TARGET_PEAK
) -> Tuple[np.ndarray, int]:
    """
    Standard Module 1 PCG preprocessing pipeline:
    1. Resample (4000Hz -> 2000Hz)
    2. Zero-phase Butterworth Bandpass filter (25-400Hz, order 5)
    3. Optional Notch filter (50Hz/60Hz, if enabled)
    4. Spike / artifact attenuation
    5. Normalization (Peak norm to 0.95 by default)
    """
    # 1. Resampling
    processed, current_sr = resample_pcg(audio, orig_sr=orig_sr, target_sr=target_sr)
    
    # 2. Bandpass filtering
    processed = butter_bandpass_filter(
        processed,
        sr=current_sr,
        lowcut=bandpass_low,
        highcut=bandpass_high,
        order=bandpass_order
    )
    
    # 3. Optional Notch filtering
    if notch_enabled:
        processed = apply_notch_filter(
            processed,
            sr=current_sr,
            freq=notch_freq,
            q=notch_q
        )
        
    # 4. Spike removal
    processed = remove_spikes(processed, sr=current_sr)
    
    # 5. Normalization
    processed = normalize_audio(
        processed,
        method=norm_method,
        target_peak=target_peak
    )
    
    return processed, current_sr

# Backward compatibility alias for existing features code
preprocess_pcg_recording = preprocess_module1_audio
