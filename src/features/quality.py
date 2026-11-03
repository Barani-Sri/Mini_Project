"""
Feature Quality Assessment for PCG signals and extracted representations.
Computes RMS energy, clipping ratio, near-zero percentage, numerical validity,
and assigns a continuous quality score [0, 1] and discrete flags (GOOD, QUESTIONABLE, POOR).
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from src.features.config import FeatureConfig

@dataclass
class QualityMetrics:
    rms_energy: float
    signal_std: float
    near_zero_ratio: float
    clipping_ratio: float
    has_nan_inf: bool
    mfcc_valid: bool
    mel_valid: bool
    quality_score: float
    quality_flag: str  # "GOOD", "QUESTIONABLE", "POOR"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rms_energy": float(self.rms_energy),
            "signal_std": float(self.signal_std),
            "near_zero_ratio": float(self.near_zero_ratio),
            "clipping_ratio": float(self.clipping_ratio),
            "has_nan_inf": bool(self.has_nan_inf),
            "mfcc_valid": bool(self.mfcc_valid),
            "mel_valid": bool(self.mel_valid),
            "quality_score": float(self.quality_score),
            "quality_flag": str(self.quality_flag)
        }

class FeatureQualityAssessment:
    """
    Evaluates PCG signal quality and feature validity.
    """
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config if config is not None else FeatureConfig()

    def assess(
        self,
        waveform: np.ndarray,
        mfcc: Optional[np.ndarray] = None,
        log_mel: Optional[np.ndarray] = None
    ) -> QualityMetrics:
        """
        Assesses PCG recording quality and numerical integrity.
        """
        if len(waveform) == 0:
            return QualityMetrics(
                rms_energy=0.0,
                signal_std=0.0,
                near_zero_ratio=1.0,
                clipping_ratio=0.0,
                has_nan_inf=False,
                mfcc_valid=False,
                mel_valid=False,
                quality_score=0.0,
                quality_flag="POOR"
            )

        # 1. Waveform Checks
        has_nan_inf = bool(np.isnan(waveform).any() or np.isinf(waveform).any())
        clean_wave = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
        
        rms_energy = float(np.sqrt(np.mean(clean_wave ** 2)))
        signal_std = float(np.std(clean_wave))
        near_zero_ratio = float(np.mean(np.abs(clean_wave) < 1e-4))
        clipping_ratio = float(np.mean(np.abs(clean_wave) >= 0.949))

        # 2. Feature Numerical Validity
        mfcc_valid = True
        if mfcc is not None:
            mfcc_valid = bool(not (np.isnan(mfcc).any() or np.isinf(mfcc).any()))

        mel_valid = True
        if log_mel is not None:
            mel_valid = bool(not (np.isnan(log_mel).any() or np.isinf(log_mel).any()))

        # 3. Quality Score Computation in [0, 1]
        score = 1.0
        
        # Penalize non-finite values heavily
        if has_nan_inf or not mfcc_valid or not mel_valid:
            score -= 0.60
            
        # Penalize excessive clipping
        if clipping_ratio > 0.01:
            score -= min(0.40, clipping_ratio * 4.0)
            
        # Penalize excessive silence / flatline
        if near_zero_ratio > 0.50:
            score -= min(0.60, (near_zero_ratio - 0.50) / 0.50 * 0.60)
            
        # Penalize extremely low energy
        if signal_std < 1e-4 or rms_energy < 1e-4:
            score -= 0.50

        # Clamp score to [0.0, 1.0]
        quality_score = max(0.0, min(1.0, score))

        # 4. Assign Quality Flag
        if quality_score >= self.config.quality_threshold_good:
            quality_flag = "GOOD"
        elif quality_score >= self.config.quality_threshold_questionable:
            quality_flag = "QUESTIONABLE"
        else:
            quality_flag = "POOR"

        return QualityMetrics(
            rms_energy=rms_energy,
            signal_std=signal_std,
            near_zero_ratio=near_zero_ratio,
            clipping_ratio=clipping_ratio,
            has_nan_inf=has_nan_inf,
            mfcc_valid=mfcc_valid,
            mel_valid=mel_valid,
            quality_score=quality_score,
            quality_flag=quality_flag
        )
