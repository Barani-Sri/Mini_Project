"""
ECG Preprocessing and Signal Feature Extraction.
Processes 500 Hz EKO DUO single/multi-lead ECG waveforms.
Applies bandpass filtering, baseline stabilization, and extracts reproducible
morphological, rhythm, and spectral features.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
from scipy import signal

from src.hfref.config import (
    HFREF_ECG_SAMPLE_RATE,
    ECG_BANDPASS_LOW_HZ,
    ECG_BANDPASS_HIGH_HZ,
    ECG_FILTER_ORDER,
    ECG_FEATURE_DIM
)

def butter_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: int,
    order: int = 3
) -> np.ndarray:
    """Applies a zero-phase Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    if low >= high:
        return data
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def preprocess_ecg(
    ecg_signal: np.ndarray,
    orig_sr: int = HFREF_ECG_SAMPLE_RATE,
    target_sr: int = HFREF_ECG_SAMPLE_RATE
) -> np.ndarray:
    """
    Standardizes and filters raw ECG signal:
    1. Resamples to target_sr if necessary
    2. 0.5 - 45 Hz bandpass filtering (Butterworth)
    3. Baseline wander removal via median subtraction
    4. Z-score amplitude normalization
    """
    if len(ecg_signal) == 0:
        return np.array([], dtype=np.float32)

    data = ecg_signal.astype(np.float32)

    # Resample if needed
    if orig_sr != target_sr and orig_sr > 0:
        num_samples = int(round(len(data) * float(target_sr) / orig_sr))
        data = signal.resample(data, num_samples)

    # Zero-phase bandpass filter
    filtered = butter_bandpass_filter(
        data,
        lowcut=ECG_BANDPASS_LOW_HZ,
        highcut=ECG_BANDPASS_HIGH_HZ,
        fs=target_sr,
        order=ECG_FILTER_ORDER
    )

    # Baseline centering
    baseline = np.median(filtered)
    centered = filtered - baseline

    # Robust normalization
    std = np.std(centered)
    if std > 1e-6:
        normalized = centered / std
    else:
        normalized = centered

    return normalized.astype(np.float32)


def detect_r_peaks(
    filtered_ecg: np.ndarray,
    fs: int = HFREF_ECG_SAMPLE_RATE
) -> np.ndarray:
    """
    Detects R-peaks using Pan-Tompkins style derivative, squaring,
    and adaptive prominence peak picking.
    """
    if len(filtered_ecg) < fs:
        return np.array([], dtype=int)

    # Differentiate to highlight slope of QRS complex
    diff = np.diff(filtered_ecg)
    squared = diff ** 2

    # Moving window integration (150ms window)
    win_len = int(0.15 * fs)
    if win_len > 1:
        kernel = np.ones(win_len) / win_len
        integrated = np.convolve(squared, kernel, mode='same')
    else:
        integrated = squared

    # Minimum distance between peaks: 300 ms (corresponds to max 200 BPM)
    min_dist = int(0.30 * fs)
    prominence = np.std(integrated) * 0.5
    peaks, _ = signal.find_peaks(integrated, distance=min_dist, prominence=prominence)

    # Refine peaks back to local maxima in the original filtered ECG
    refined_peaks = []
    search_radius = int(0.08 * fs)  # 80ms radius
    for p in peaks:
        start = max(0, p - search_radius)
        end = min(len(filtered_ecg), p + search_radius + 1)
        if start < end:
            local_max = start + np.argmax(filtered_ecg[start:end])
            refined_peaks.append(local_max)

    return np.unique(refined_peaks)


def extract_ecg_features(
    ecg_signal: np.ndarray,
    fs: int = HFREF_ECG_SAMPLE_RATE
) -> Dict[str, Any]:
    """
    Extracts 24 reproducible ECG features including rhythm, variability,
    morphology, and spectral power.
    Returns both feature dictionary and normalized feature vector.
    """
    if len(ecg_signal) == 0:
        return {
            "feature_vector": np.zeros(ECG_FEATURE_DIM, dtype=np.float32),
            "heart_rate_bpm": 0.0,
            "rr_mean_ms": 0.0,
            "sdnn_ms": 0.0,
            "rmssd_ms": 0.0,
            "r_peaks_count": 0,
            "energy": 0.0
        }

    proc = preprocess_ecg(ecg_signal, orig_sr=fs, target_sr=fs)
    r_peaks = detect_r_peaks(proc, fs=fs)

    # 1. Statistical & amplitude metrics
    mean_val = float(np.mean(proc))
    std_val = float(np.std(proc))
    rms_val = float(np.sqrt(np.mean(proc ** 2)))
    var_val = float(np.var(proc))
    p2p_val = float(np.ptp(proc)) if len(proc) > 0 else 0.0
    energy_val = float(np.sum(proc ** 2) / len(proc)) if len(proc) > 0 else 0.0
    
    # Skewness and Kurtosis
    if std_val > 1e-6:
        skew_val = float(np.mean(((proc - mean_val) / std_val) ** 3))
        kurt_val = float(np.mean(((proc - mean_val) / std_val) ** 4) - 3.0)
    else:
        skew_val = 0.0
        kurt_val = 0.0

    # 2. Rhythm and R-R Interval Heart Rate Variability (HRV) metrics
    if len(r_peaks) >= 2:
        rr_intervals = np.diff(r_peaks) / float(fs) * 1000.0  # in ms
        rr_mean = float(np.mean(rr_intervals))
        hr_bpm = float(60000.0 / rr_mean) if rr_mean > 0 else 0.0
        sdnn = float(np.std(rr_intervals))
        rr_diffs = np.diff(rr_intervals)
        rmssd = float(np.sqrt(np.mean(rr_diffs ** 2))) if len(rr_diffs) > 0 else 0.0
        pnn50 = float(np.sum(np.abs(rr_diffs) > 50.0) / len(rr_diffs) * 100.0) if len(rr_diffs) > 0 else 0.0
        rr_min = float(np.min(rr_intervals))
        rr_max = float(np.max(rr_intervals))
    else:
        rr_mean = 0.0
        hr_bpm = 0.0
        sdnn = 0.0
        rmssd = 0.0
        pnn50 = 0.0
        rr_min = 0.0
        rr_max = 0.0

    # 3. Spectral Frequency Band Powers (Welch PSD)
    freqs, psd = signal.welch(proc, fs=fs, nperseg=min(len(proc), fs * 2))
    total_power = np.sum(psd) + 1e-9

    def _band_power(f_low: float, f_high: float) -> float:
        mask = (freqs >= f_low) & (freqs < f_high)
        return float(np.sum(psd[mask]) / total_power)

    p_delta = _band_power(0.5, 4.0)
    p_theta = _band_power(4.0, 8.0)
    p_alpha = _band_power(8.0, 15.0)
    p_beta = _band_power(15.0, 30.0)
    p_gamma = _band_power(30.0, 45.0)

    # Dominant peak frequency and spectral centroid
    dom_freq = float(freqs[np.argmax(psd)]) if len(psd) > 0 else 0.0
    spectral_centroid = float(np.sum(freqs * psd) / total_power)

    # Assemble 24-dimensional feature vector
    feat_vec = np.array([
        mean_val,
        std_val,
        rms_val,
        var_val,
        p2p_val,
        energy_val,
        skew_val,
        kurt_val,
        hr_bpm / 100.0,            # Normalized scale ~0.6-1.5
        rr_mean / 1000.0,          # Normalized scale ~0.6-1.2
        sdnn / 100.0,
        rmssd / 100.0,
        pnn50 / 100.0,
        rr_min / 1000.0,
        rr_max / 1000.0,
        len(r_peaks) / 30.0,       # Peaks per second
        p_delta,
        p_theta,
        p_alpha,
        p_beta,
        p_gamma,
        dom_freq / 20.0,
        spectral_centroid / 20.0,
        float(len(proc)) / (30.0 * fs) # Duration ratio
    ], dtype=np.float32)

    # Guard against NaN/Inf
    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)

    return {
        "feature_vector": feat_vec,
        "heart_rate_bpm": round(hr_bpm, 1),
        "rr_mean_ms": round(rr_mean, 1),
        "sdnn_ms": round(sdnn, 1),
        "rmssd_ms": round(rmssd, 1),
        "pnn50_percent": round(pnn50, 1),
        "r_peaks_count": int(len(r_peaks)),
        "dominant_frequency_hz": round(dom_freq, 2),
        "spectral_centroid_hz": round(spectral_centroid, 2),
        "energy": round(energy_val, 4)
    }
