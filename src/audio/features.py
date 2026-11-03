"""
Comprehensive Feature Extraction Engine for PCG Heart Sound Audio.
Extracts time-domain, frequency-domain, spectral, cepstral (MFCC),
and segment-level physiological murmur features.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import scipy.fft as fft
import pandas as pd

from src.utils.config import (
    SAMPLE_RATE,
    N_MFCC,
    FRAME_LENGTH_MS,
    FRAME_STEP_MS,
    STATE_S1,
    STATE_SYSTOLE,
    STATE_S2,
    STATE_DIASTOLE
)
from src.audio.preprocessing import (
    butter_bandpass_filter,
    compute_shannon_energy,
    compute_hilbert_envelope
)
from src.audio.segmentation import (
    extract_cardiac_cycles,
    compute_cycle_timing_features
)

def extract_time_domain_features(audio: np.ndarray) -> Dict[str, float]:
    """
    Extracts statistical and temporal waveform metrics.
    """
    if len(audio) == 0:
        return {}
        
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    crest_factor = float(peak / (rms + 1e-8))
    
    # Zero Crossing Rate
    zero_crossings = np.sum(np.diff(audio > 0) != 0)
    zcr = float(zero_crossings / max(1, len(audio) - 1))
    
    # Statistical moments
    variance = float(np.var(audio))
    skewness = float(stats.skew(audio))
    kurt = float(stats.kurtosis(audio))
    
    # Envelopes
    shannon_env = compute_shannon_energy(audio)
    shannon_mean = float(np.mean(shannon_env))
    shannon_std = float(np.std(shannon_env))
    shannon_var = float(np.var(shannon_env))
    
    hilbert_env = compute_hilbert_envelope(audio)
    hilbert_mean = float(np.mean(hilbert_env))
    hilbert_std = float(np.std(hilbert_env))
    
    return {
        "time_rms": rms,
        "time_peak": peak,
        "time_crest_factor": crest_factor,
        "time_zcr": zcr,
        "time_variance": variance,
        "time_skewness": skewness,
        "time_kurtosis": kurt,
        "shannon_mean": shannon_mean,
        "shannon_std": shannon_std,
        "shannon_var": shannon_var,
        "hilbert_mean": hilbert_mean,
        "hilbert_std": hilbert_std
    }

def extract_frequency_domain_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE
) -> Dict[str, float]:
    """
    Extracts spectral features and Power Spectral Density (PSD) in clinical sub-bands.
    - Band 1 (25-50 Hz): Low-frequency fundamental S1/S2
    - Band 2 (50-100 Hz): Heart sound harmonics
    - Band 3 (100-200 Hz): Medium-frequency systolic murmurs
    - Band 4 (200-400 Hz): High-frequency turbulent murmurs
    """
    if len(audio) < 256:
        return {}
        
    nperseg = min(512, len(audio))
    freqs, psd = signal.welch(audio, fs=sr, nperseg=nperseg)
    total_power = float(np.sum(psd) + 1e-12)
    
    # Sub-band powers
    def get_band_power(f_min, f_max):
        idx = (freqs >= f_min) & (freqs < f_max)
        if not np.any(idx):
            return 0.0
        return float(np.sum(psd[idx]))
        
    p_band1 = get_band_power(25, 50)
    p_band2 = get_band_power(50, 100)
    p_band3 = get_band_power(100, 200)
    p_band4 = get_band_power(200, 400)
    
    # Relative sub-band ratios
    r_band1 = p_band1 / total_power
    r_band2 = p_band2 / total_power
    r_band3 = p_band3 / total_power
    r_band4 = p_band4 / total_power
    
    # Murmur turbulence index: high-frequency power ratio (100-400Hz vs Total)
    murmur_freq_ratio = (p_band3 + p_band4) / total_power
    
    # Spectral Centroid & Spread
    spectral_centroid = float(np.sum(freqs * psd) / total_power)
    spectral_spread = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_power))
    
    # Spectral Rolloff (85% and 95%)
    cum_power = np.cumsum(psd)
    rolloff_85_idx = np.where(cum_power >= 0.85 * total_power)[0]
    rolloff_85 = float(freqs[rolloff_85_idx[0]]) if len(rolloff_85_idx) > 0 else 0.0
    
    rolloff_95_idx = np.where(cum_power >= 0.95 * total_power)[0]
    rolloff_95 = float(freqs[rolloff_95_idx[0]]) if len(rolloff_95_idx) > 0 else 0.0
    
    # Spectral Flatness (Wiener entropy: geometric mean / arithmetic mean)
    # Higher in noisy turbulent murmurs, lower in tonal heart sounds
    geom_mean = np.exp(np.mean(np.log(psd + 1e-12)))
    arith_mean = np.mean(psd) + 1e-12
    spectral_flatness = float(geom_mean / arith_mean)
    
    return {
        "spec_total_power": total_power,
        "spec_band1_power": p_band1,
        "spec_band2_power": p_band2,
        "spec_band3_power": p_band3,
        "spec_band4_power": p_band4,
        "spec_band1_rel": r_band1,
        "spec_band2_rel": r_band2,
        "spec_band3_rel": r_band3,
        "spec_band4_rel": r_band4,
        "spec_murmur_freq_ratio": murmur_freq_ratio,
        "spec_centroid": spectral_centroid,
        "spec_spread": spectral_spread,
        "spec_rolloff_85": rolloff_85,
        "spec_rolloff_95": rolloff_95,
        "spec_flatness": spectral_flatness
    }

def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def get_mel_filterbank(
    num_filters: int = 26,
    nfft: int = 512,
    sr: int = SAMPLE_RATE,
    low_freq: float = 20.0,
    high_freq: float = 800.0
) -> np.ndarray:
    """
    Constructs triangular Mel filterbank matrix.
    """
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((nfft + 1) * hz_points / sr).astype(int)
    
    fbank = np.zeros((num_filters, int(np.floor(nfft / 2 + 1))))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        
        for k in range(f_m_minus, f_m):
            if f_m != f_m_minus:
                fbank[m - 1, k] = (k - bin_points[m - 1]) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            if f_m_plus != f_m:
                fbank[m - 1, k] = (bin_points[m + 1] - k) / (f_m_plus - f_m)
                
    return fbank

def extract_mfcc_features(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = N_MFCC,
    frame_len_ms: float = FRAME_LENGTH_MS,
    frame_step_ms: float = FRAME_STEP_MS
) -> Dict[str, float]:
    """
    Extracts Mel-Frequency Cepstral Coefficients (MFCCs) and their summary statistics.
    """
    frame_len = int((frame_len_ms / 1000.0) * sr)
    frame_step = int((frame_step_ms / 1000.0) * sr)
    
    if len(audio) < frame_len:
        # Pad with zeros if too short
        audio = np.pad(audio, (0, frame_len - len(audio)))
        
    # Framing
    num_frames = 1 + int((len(audio) - frame_len) / frame_step)
    frames = np.zeros((num_frames, frame_len))
    for i in range(num_frames):
        start = i * frame_step
        frames[i] = audio[start:start + frame_len]
        
    # Windowing
    frames *= np.hamming(frame_len)
    
    # FFT & Power Spectrum
    nfft = 512
    mag_frames = np.abs(fft.rfft(frames, n=nfft))
    pow_frames = (1.0 / nfft) * (mag_frames ** 2)
    
    # Filterbank
    num_filters = 26
    fbank = get_mel_filterbank(num_filters=num_filters, nfft=nfft, sr=sr)
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_filter_banks = np.log(filter_banks)
    
    # Discrete Cosine Transform (DCT)
    from scipy.fftpack import dct
    mfcc = dct(log_filter_banks, type=2, axis=1, norm="ortho")[:, :n_mfcc]
    
    # Delta MFCCs
    if len(mfcc) > 2:
        delta_mfcc = np.diff(mfcc, axis=0)
    else:
        delta_mfcc = np.zeros_like(mfcc)
        
    features = {}
    for c in range(n_mfcc):
        features[f"mfcc_{c+1}_mean"] = float(np.mean(mfcc[:, c]))
        features[f"mfcc_{c+1}_std"] = float(np.std(mfcc[:, c]))
        features[f"mfcc_{c+1}_max"] = float(np.max(mfcc[:, c]))
        features[f"mfcc_{c+1}_min"] = float(np.min(mfcc[:, c]))
        if c < 10 and len(delta_mfcc) > 0:
            features[f"delta_mfcc_{c+1}_mean"] = float(np.mean(delta_mfcc[:, c]))
            features[f"delta_mfcc_{c+1}_std"] = float(np.std(delta_mfcc[:, c]))
            
    return features

def extract_segment_murmur_features(
    audio: np.ndarray,
    seg_df: pd.DataFrame,
    sr: int = SAMPLE_RATE
) -> Dict[str, float]:
    """
    Extracts clinical murmur features by evaluating acoustic properties across
    cardiac states: S1, Systole (where systolic murmurs occur), S2, and Diastole.
    """
    features = {}
    
    if seg_df is None or len(seg_df) == 0:
        return {
            "seg_systole_energy_ratio": 0.25,
            "seg_diastole_energy_ratio": 0.25,
            "seg_sys_s1_rms_ratio": 0.5,
            "seg_dia_s2_rms_ratio": 0.5,
            "seg_sys_shannon_var": 0.0,
            "seg_dia_shannon_var": 0.0
        }
        
    cycles = extract_cardiac_cycles(audio, seg_df, sr=sr)
    
    if not cycles:
        # Fallback to computing across entire segmented intervals
        total_energy = float(np.sum(audio ** 2) + 1e-12)
        
        def get_state_energy_and_rms(state_id):
            state_rows = seg_df[seg_df["state"] == state_id]
            state_samples = []
            for _, r in state_rows.iterrows():
                s = max(0, int(r["start"] * sr))
                e = min(len(audio), int(r["end"] * sr))
                if e > s:
                    state_samples.append(audio[s:e])
            if state_samples:
                concat = np.concatenate(state_samples)
                energy = float(np.sum(concat ** 2))
                rms = float(np.sqrt(np.mean(concat ** 2)))
                return energy, rms, concat
            return 0.0, 0.0, np.array([])
            
        e_s1, rms_s1, _ = get_state_energy_and_rms(STATE_S1)
        e_sys, rms_sys, arr_sys = get_state_energy_and_rms(STATE_SYSTOLE)
        e_s2, rms_s2, _ = get_state_energy_and_rms(STATE_S2)
        e_dia, rms_dia, arr_dia = get_state_energy_and_rms(STATE_DIASTOLE)
        
        sys_shannon_var = float(np.var(compute_shannon_energy(arr_sys))) if len(arr_sys) > 10 else 0.0
        dia_shannon_var = float(np.var(compute_shannon_energy(arr_dia))) if len(arr_dia) > 10 else 0.0
        
        return {
            "seg_systole_energy_ratio": float(e_sys / total_energy),
            "seg_diastole_energy_ratio": float(e_dia / total_energy),
            "seg_sys_s1_rms_ratio": float(rms_sys / (rms_s1 + 1e-6)),
            "seg_dia_s2_rms_ratio": float(rms_dia / (rms_s2 + 1e-6)),
            "seg_sys_shannon_var": sys_shannon_var,
            "seg_dia_shannon_var": dia_shannon_var
        }
        
    # Aggregate over extracted cycles
    sys_ratios = []
    dia_ratios = []
    sys_s1_ratios = []
    dia_s2_ratios = []
    sys_shannon_vars = []
    
    for c in cycles:
        c_audio = c["cycle_audio"]
        s1_a = c["s1_audio"]
        sys_a = c["systole_audio"]
        s2_a = c["s2_audio"]
        dia_a = c["diastole_audio"]
        
        c_energy = np.sum(c_audio ** 2) + 1e-12
        sys_energy = np.sum(sys_a ** 2)
        dia_energy = np.sum(dia_a ** 2)
        
        sys_ratios.append(sys_energy / c_energy)
        dia_ratios.append(dia_energy / c_energy)
        
        rms_s1 = np.sqrt(np.mean(s1_a ** 2)) if len(s1_a) > 0 else 1e-6
        rms_sys = np.sqrt(np.mean(sys_a ** 2)) if len(sys_a) > 0 else 0.0
        rms_s2 = np.sqrt(np.mean(s2_a ** 2)) if len(s2_a) > 0 else 1e-6
        rms_dia = np.sqrt(np.mean(dia_a ** 2)) if len(dia_a) > 0 else 0.0
        
        sys_s1_ratios.append(rms_sys / (rms_s1 + 1e-6))
        dia_s2_ratios.append(rms_dia / (rms_s2 + 1e-6))
        
        if len(sys_a) > 10:
            sys_shannon_vars.append(np.var(compute_shannon_energy(sys_a)))
            
    features["seg_systole_energy_ratio"] = float(np.mean(sys_ratios))
    features["seg_diastole_energy_ratio"] = float(np.mean(dia_ratios))
    features["seg_sys_s1_rms_ratio"] = float(np.mean(sys_s1_ratios))
    features["seg_dia_s2_rms_ratio"] = float(np.mean(dia_s2_ratios))
    features["seg_sys_shannon_var"] = float(np.mean(sys_shannon_vars)) if sys_shannon_vars else 0.0
    features["seg_num_cycles"] = float(len(cycles))
    
    return features

def extract_full_pcg_features(
    audio: np.ndarray,
    seg_df: Optional[pd.DataFrame] = None,
    location: str = "AV",
    sr: int = SAMPLE_RATE
) -> Dict[str, float]:
    """
    Runs full feature extraction pipeline on a preprocessed PCG recording.
    Combines:
    1. Time domain features
    2. Frequency domain & Sub-band PSD features
    3. MFCC features
    4. Segmentation & Murmur interval features
    5. Location encoding
    """
    feat_dict: Dict[str, float] = {}
    
    # 1. Time domain
    feat_dict.update(extract_time_domain_features(audio))
    
    # 2. Frequency domain
    feat_dict.update(extract_frequency_domain_features(audio, sr=sr))
    
    # 3. MFCCs
    feat_dict.update(extract_mfcc_features(audio, sr=sr))
    
    # 4. Cycle timing
    if seg_df is not None:
        feat_dict.update(compute_cycle_timing_features(seg_df))
        feat_dict.update(extract_segment_murmur_features(audio, seg_df, sr=sr))
    else:
        feat_dict.update(compute_cycle_timing_features(pd.DataFrame()))
        feat_dict.update(extract_segment_murmur_features(audio, pd.DataFrame(), sr=sr))
        
    # 5. Auscultation Location One-Hot Encoding
    for loc in ["AV", "MV", "PV", "TV", "Phc"]:
        feat_dict[f"loc_is_{loc}"] = 1.0 if location == loc else 0.0
        
    return feat_dict
