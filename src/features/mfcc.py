"""
MFCC, Delta, and Delta-Delta Feature Extraction for PCG signals.
Extracts 40 MFCCs, 40 Deltas, and 40 Delta-Deltas concatenated into a (120, T) acoustic representation.
"""
from typing import Optional, Tuple, Union
import numpy as np
import librosa

from src.features.config import FeatureConfig
from src.features.melspectrogram import MelSpectrogramExtractor

class MFCCExtractor:
    """
    Extracts 40 MFCCs, Delta MFCCs, and Delta-Delta MFCCs from PCG waveforms.
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config if config is not None else FeatureConfig()
        self.mel_extractor = MelSpectrogramExtractor(self.config)

    def extract_mfcc(self, waveform: np.ndarray, log_mel: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extracts 40 MFCC coefficients from audio waveform or precomputed log-Mel spectrogram.
        Returns:
        - mfcc: np.ndarray of shape (n_mfcc, T) e.g. (40, 501)
        """
        if log_mel is None:
            # Extract non-standardized log-Mel for DCT computation
            power_mel = self.mel_extractor.extract_power_mel(waveform)
            if np.max(power_mel) < 1e-12:
                return np.zeros((self.config.n_mfcc, self.config.num_time_frames), dtype=np.float32)
            log_mel = librosa.power_to_db(power_mel, ref=np.max, top_db=self.config.top_db)
            
        mfcc = librosa.feature.mfcc(
            S=log_mel,
            sr=self.config.sample_rate,
            n_mfcc=self.config.n_mfcc
        )
        return np.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    def extract_deltas(
        self,
        mfcc: np.ndarray,
        width: int = 9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes 1st order (Delta) and 2nd order (Delta-Delta) temporal derivatives.
        Returns:
        - delta_mfcc: np.ndarray of shape (n_mfcc, T)
        - delta_delta_mfcc: np.ndarray of shape (n_mfcc, T)
        """
        if mfcc.shape[1] < width:
            width = max(3, (mfcc.shape[1] // 2) * 2 + 1)
            
        if np.max(np.abs(mfcc)) < 1e-12 or mfcc.shape[1] < 3:
            zeros = np.zeros_like(mfcc, dtype=np.float32)
            return zeros, zeros
            
        delta = librosa.feature.delta(mfcc, width=width, order=1, axis=-1)
        delta2 = librosa.feature.delta(mfcc, width=width, order=2, axis=-1)
        
        delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        delta2 = np.nan_to_num(delta2, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return delta, delta2

    def extract_all(
        self,
        waveform: np.ndarray,
        log_mel: Optional[np.ndarray] = None,
        standardize: Optional[bool] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts all MFCC features for a waveform.
        Returns:
        - mfcc: (40, T)
        - delta: (40, T)
        - delta_delta: (40, T)
        - combined: (120, T) concatenated along channel axis
        """
        import scipy.fft
        
        if log_mel is None:
            power_mel = self.mel_extractor.extract_power_mel(waveform)
            if np.max(power_mel) < 1e-12:
                zeros_40 = np.zeros((self.config.n_mfcc, self.config.num_time_frames), dtype=np.float32)
                zeros_120 = np.zeros((self.config.combined_mfcc_channels, self.config.num_time_frames), dtype=np.float32)
                return zeros_40, zeros_40, zeros_40, zeros_120
            log_mel_unnorm = librosa.power_to_db(power_mel, ref=np.max, top_db=self.config.top_db)
        else:
            log_mel_unnorm = log_mel
            
        mfcc = scipy.fft.dct(log_mel_unnorm, axis=0, type=2, norm="ortho")[:self.config.n_mfcc]
        mfcc = np.nan_to_num(mfcc, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        delta, delta2 = self.extract_deltas(mfcc)
        
        # Combine along feature / channel axis (axis 0)
        combined_parts = [mfcc]
        if self.config.include_delta:
            combined_parts.append(delta)
        if self.config.include_delta_delta:
            combined_parts.append(delta2)
            
        combined = np.concatenate(combined_parts, axis=0).astype(np.float32)
        
        apply_norm = standardize if standardize is not None else (self.config.normalization == "per_window")
        if apply_norm:
            mfcc = MelSpectrogramExtractor.standardize_feature_map(mfcc)
            delta = MelSpectrogramExtractor.standardize_feature_map(delta)
            delta2 = MelSpectrogramExtractor.standardize_feature_map(delta2)
            combined = MelSpectrogramExtractor.standardize_feature_map(combined)
            
        return mfcc, delta, delta2, combined

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Returns the combined (120, T) representation by default."""
        _, _, _, combined = self.extract_all(waveform)
        return combined
