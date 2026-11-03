"""
PCG Preprocessing and Multi-Location Acoustic Feature Extraction.
Processes 4000 Hz heart sounds recorded at the four standard auscultation valves:
- Aortic (RUSB / AV)
- Pulmonary (LUSB / PV)
- Tricuspid (LLSB / TV)
- Mitral (APEX / MV)
Extracts 48 acoustic features per location across spectral, cepstral, and envelope domains.
"""
from typing import Dict, Any, Tuple, Optional
import numpy as np
import librosa
from scipy import signal

from src.hfref.config import (
    HFREF_PCG_SAMPLE_RATE,
    HFREF_PCG_TARGET_SR,
    PCG_BANDPASS_LOW_HZ,
    PCG_BANDPASS_HIGH_HZ,
    PCG_FILTER_ORDER,
    PCG_NORM_METHOD,
    PCG_TARGET_PEAK,
    PCG_FEATURE_PER_LOC_DIM
)

def butter_bandpass_pcg(
    data: np.ndarray,
    lowcut: float = PCG_BANDPASS_LOW_HZ,
    highcut: float = PCG_BANDPASS_HIGH_HZ,
    fs: int = HFREF_PCG_TARGET_SR,
    order: int = PCG_FILTER_ORDER
) -> np.ndarray:
    """Applies a zero-phase Butterworth bandpass filter to PCG signal."""
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    if low >= high:
        return data
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, data)

def preprocess_hfref_pcg(
    audio_signal: np.ndarray,
    orig_sr: int = HFREF_PCG_SAMPLE_RATE,
    target_sr: int = HFREF_PCG_TARGET_SR,
    target_peak: float = PCG_TARGET_PEAK
) -> np.ndarray:
    """
    Standardizes raw heart sound audio:
    1. Resamples from native 4000 Hz to target 2000 Hz
    2. 25 - 400 Hz zero-phase Butterworth bandpass filter
    3. Peak amplitude normalization to target_peak (0.95)
    """
    if len(audio_signal) == 0:
        return np.array([], dtype=np.float32)

    data = audio_signal.astype(np.float32)

    # Resample if needed
    if orig_sr != target_sr and orig_sr > 0:
        num_samples = int(round(len(data) * float(target_sr) / orig_sr))
        data = signal.resample(data, num_samples)

    # Bandpass filter (cardiac acoustic range)
    filtered = butter_bandpass_pcg(data, fs=target_sr)

    # Peak normalization
    max_abs = np.max(np.abs(filtered))
    if max_abs > 1e-6:
        normalized = (filtered / max_abs) * target_peak
    else:
        normalized = filtered

    return normalized.astype(np.float32)


def compute_shannon_energy_envelope(
    pcg: np.ndarray,
    window_len: int = 40
) -> np.ndarray:
    """Computes normalized Shannon energy envelope of heart sound."""
    if len(pcg) == 0:
        return np.array([], dtype=np.float32)

    # Normalize to [-1, 1]
    max_val = np.max(np.abs(pcg)) + 1e-9
    norm_pcg = pcg / max_val

    # Shannon energy: -x^2 * log(x^2)
    x_sq = norm_pcg ** 2
    # Guard against log(0)
    x_sq = np.clip(x_sq, 1e-12, 1.0)
    shannon_en = -x_sq * np.log(x_sq)

    # Smooth with moving average window
    kernel = np.ones(window_len) / window_len
    smoothed = np.convolve(shannon_en, kernel, mode='same')
    return smoothed.astype(np.float32)


def extract_pcg_acoustic_features(
    pcg_signal: np.ndarray,
    sr: int = HFREF_PCG_TARGET_SR
) -> Dict[str, Any]:
    """
    Extracts 48 acoustic features from a PCG recording:
    - 13 MFCC means
    - 13 Delta-MFCC means
    - 8 Log-Mel frequency band energies (25-400Hz)
    - 4 Spectral features (centroid, bandwidth, roll-off, flatness)
    - 10 Time-domain & envelope features (Shannon envelope, RMS, ZCR, crest factor, kurtosis, etc.)
    """
    if len(pcg_signal) == 0:
        return {
            "feature_vector": np.zeros(PCG_FEATURE_PER_LOC_DIM, dtype=np.float32),
            "rms_energy": 0.0,
            "spectral_centroid_hz": 0.0,
            "zero_crossing_rate": 0.0,
            "shannon_energy_mean": 0.0
        }

    proc = preprocess_hfref_pcg(pcg_signal, orig_sr=sr, target_sr=sr)

    # 1. MFCC & Delta features (13 + 13 = 26)
    # n_fft=512 at 2000Hz = 256ms, hop=128 (64ms)
    n_fft = min(512, len(proc))
    hop_length = min(128, n_fft // 2)

    try:
        mfcc = librosa.feature.mfcc(
            y=proc,
            sr=sr,
            n_mfcc=13,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=PCG_BANDPASS_LOW_HZ,
            fmax=PCG_BANDPASS_HIGH_HZ
        )
        mfcc_mean = np.mean(mfcc, axis=1) # (13,)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta_mean = np.mean(delta_mfcc, axis=1) # (13,)
    except Exception:
        mfcc_mean = np.zeros(13, dtype=np.float32)
        delta_mean = np.zeros(13, dtype=np.float32)

    # 2. Log-Mel Filterbank energies (8 sub-bands)
    try:
        melspec = librosa.feature.melspectrogram(
            y=proc,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=8,
            fmin=PCG_BANDPASS_LOW_HZ,
            fmax=PCG_BANDPASS_HIGH_HZ
        )
        log_mel = librosa.power_to_db(melspec, ref=np.max)
        log_mel_mean = np.mean(log_mel, axis=1) # (8,)
    except Exception:
        log_mel_mean = np.zeros(8, dtype=np.float32)

    # 3. Spectral Features (4)
    try:
        sc = float(np.mean(librosa.feature.spectral_centroid(y=proc, sr=sr, n_fft=n_fft, hop_length=hop_length)))
        sb = float(np.mean(librosa.feature.spectral_bandwidth(y=proc, sr=sr, n_fft=n_fft, hop_length=hop_length)))
        sro = float(np.mean(librosa.feature.spectral_rolloff(y=proc, sr=sr, n_fft=n_fft, hop_length=hop_length)))
        sf = float(np.mean(librosa.feature.spectral_flatness(y=proc, n_fft=n_fft, hop_length=hop_length)))
    except Exception:
        sc, sb, sro, sf = 0.0, 0.0, 0.0, 0.0

    # 4. Temporal & Envelope Features (10)
    rms_val = float(np.sqrt(np.mean(proc ** 2)))
    zcr_val = float(np.mean(librosa.feature.zero_crossing_rate(proc)))
    mean_val = float(np.mean(proc))
    std_val = float(np.std(proc))
    var_val = float(np.var(proc))
    p2p_val = float(np.ptp(proc)) if len(proc) > 0 else 0.0

    if std_val > 1e-6:
        skew_val = float(np.mean(((proc - mean_val) / std_val) ** 3))
        kurt_val = float(np.mean(((proc - mean_val) / std_val) ** 4) - 3.0)
    else:
        skew_val, kurt_val = 0.0, 0.0

    peak_amp = float(np.max(np.abs(proc)))
    crest_factor = float(peak_amp / rms_val) if rms_val > 1e-6 else 0.0

    shannon_env = compute_shannon_energy_envelope(proc)
    shannon_mean = float(np.mean(shannon_env)) if len(shannon_env) > 0 else 0.0

    temporal_feats = np.array([
        rms_val,
        zcr_val,
        mean_val,
        std_val,
        var_val,
        p2p_val,
        skew_val,
        kurt_val,
        crest_factor,
        shannon_mean
    ], dtype=np.float32)

    # Concatenate all 48 features
    feat_vec = np.concatenate([
        mfcc_mean,       # 13
        delta_mean,      # 13
        log_mel_mean,    # 8
        np.array([sc / 500.0, sb / 500.0, sro / 500.0, sf], dtype=np.float32), # 4
        temporal_feats   # 10
    ]).astype(np.float32)

    feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=1.0, neginf=-1.0)

    return {
        "feature_vector": feat_vec,
        "rms_energy": round(rms_val, 4),
        "spectral_centroid_hz": round(sc, 1),
        "spectral_bandwidth_hz": round(sb, 1),
        "zero_crossing_rate": round(zcr_val, 4),
        "shannon_energy_mean": round(shannon_mean, 4)
    }
