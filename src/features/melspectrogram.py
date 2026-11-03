"""
Mel-Spectrogram Feature Extraction for PCG signals.
Extracts log-Mel spectrograms with numerical stability guarantees and per-window standardization.
"""
from typing import Optional, Tuple, Union
import numpy as np
import librosa

from src.features.config import FeatureConfig

class MelSpectrogramExtractor:
    """
    Extracts Log-Mel Spectrogram representations from PCG waveforms.
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config if config is not None else FeatureConfig()
        self.mel_basis = librosa.filters.mel(
            sr=self.config.sample_rate,
            n_fft=self.config.n_fft,
            n_mels=self.config.n_mels,
            fmin=self.config.f_min,
            fmax=self.config.f_max
        ).astype(np.float32)
        
    def extract_power_mel(self, waveform: np.ndarray) -> np.ndarray:
        """
        Computes raw power Mel-spectrogram of shape (n_mels, T).
        Handles edge cases (silent, short, NaN/Inf).
        """
        if len(waveform) == 0:
            return np.zeros((self.config.n_mels, self.config.num_time_frames), dtype=np.float32)
            
        # Sanitize non-finite values
        clean_waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        # Check for absolute silence
        if np.max(np.abs(clean_waveform)) < 1e-8:
            return np.zeros((self.config.n_mels, self.config.num_time_frames), dtype=np.float32)
            
        # Pad short signals to expected length if needed
        if len(clean_waveform) < self.config.waveform_samples:
            padded = np.zeros(self.config.waveform_samples, dtype=np.float32)
            padded[:len(clean_waveform)] = clean_waveform
            clean_waveform = padded
            
        D = librosa.stft(
            clean_waveform,
            n_fft=self.config.n_fft,
            win_length=self.config.frame_length_samples,
            hop_length=self.config.hop_length_samples,
            window=self.config.window_function,
            center=True
        )
        power_spec = np.abs(D) ** 2
        mel = np.dot(self.mel_basis, power_spec)
        return np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def extract_log_mel(
        self,
        waveform: np.ndarray,
        standardize: Optional[bool] = None
    ) -> np.ndarray:
        """
        Computes Log-Mel Spectrogram in dB scale with optional per-window standardization.
        Returns:
        - log_mel: np.ndarray of shape (n_mels, T) e.g. (128, 501)
        """
        power_mel = self.extract_power_mel(waveform)
        
        # If completely silent
        if np.max(power_mel) < 1e-12:
            return np.zeros((self.config.n_mels, self.config.num_time_frames), dtype=np.float32)
            
        # Numerically stable dB scaling
        log_mel = librosa.power_to_db(power_mel, ref=np.max, top_db=self.config.top_db)
        log_mel = np.nan_to_num(log_mel, nan=-self.config.top_db, posinf=0.0, neginf=-self.config.top_db).astype(np.float32)
        
        apply_norm = standardize if standardize is not None else (self.config.normalization == "per_window")
        if apply_norm:
            log_mel = self.standardize_feature_map(log_mel)
            
        return log_mel.astype(np.float32)

    @staticmethod
    def standardize_feature_map(feat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """
        Standardizes feature map (zero mean, unit variance) per window.
        Safely handles constant or zero-variance features.
        """
        mean = np.mean(feat)
        std = np.std(feat)
        if std < eps:
            return np.zeros_like(feat, dtype=np.float32)
        norm = (feat - mean) / (std + eps)
        return np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        return self.extract_log_mel(waveform)
