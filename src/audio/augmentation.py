"""
Training-only audio data augmentation for PCG waveforms.
Includes time shifting, low-level additive Gaussian noise, and time stretching.
Guarantees exact output shape preservation (e.g. 10,000 samples).
"""
from typing import Optional, Tuple, Union
import numpy as np
import scipy.signal as signal

from src.utils.config import (
    MODULE1_TARGET_SAMPLE_RATE,
    AUG_MAX_TIME_SHIFT_SEC,
    AUG_NOISE_STD,
    AUG_TIME_STRETCH_RANGE
)

def time_shift(
    audio: np.ndarray,
    sr: int = MODULE1_TARGET_SAMPLE_RATE,
    max_shift_sec: float = AUG_MAX_TIME_SHIFT_SEC,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Randomly shifts audio forward or backward in time using circular roll.
    Preserves exact array length.
    """
    if len(audio) == 0:
        return audio
    if rng is None:
        rng = np.random.default_rng()
        
    max_samples = int(max_shift_sec * sr)
    if max_samples <= 0:
        return audio
        
    shift = rng.integers(-max_samples, max_samples + 1)
    shifted = np.roll(audio, shift)
    return shifted.astype(np.float32)

def add_gaussian_noise(
    audio: np.ndarray,
    std: float = AUG_NOISE_STD,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Adds zero-mean additive Gaussian noise to PCG waveform.
    """
    if len(audio) == 0:
        return audio
    if rng is None:
        rng = np.random.default_rng()
        
    noise = rng.normal(0.0, std, size=audio.shape).astype(np.float32)
    noisy_audio = audio + noise
    return noisy_audio.astype(np.float32)

def time_stretch(
    audio: np.ndarray,
    stretch_range: Tuple[float, float] = AUG_TIME_STRETCH_RANGE,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Stretches/compresses signal in time by a factor within stretch_range (e.g. 0.95 - 1.05),
    then resamples/interpolates back to the exact original length.
    """
    orig_len = len(audio)
    if orig_len == 0:
        return audio
    if rng is None:
        rng = np.random.default_rng()
        
    rate = rng.uniform(stretch_range[0], stretch_range[1])
    if np.isclose(rate, 1.0, atol=1e-3):
        return audio
        
    # Interpolate across stretched time grid
    orig_indices = np.linspace(0, orig_len - 1, orig_len)
    stretched_indices = np.linspace(0, orig_len - 1, int(orig_len * rate))
    stretched_audio = np.interp(stretched_indices, orig_indices, audio)
    
    # Resample / re-interpolate back to exact orig_len
    target_indices = np.linspace(0, len(stretched_audio) - 1, orig_len)
    restored_audio = np.interp(target_indices, np.arange(len(stretched_audio)), stretched_audio)
    
    return restored_audio.astype(np.float32)

class PCGAugmentor:
    """
    Augmentation pipeline for training waveforms.
    """
    def __init__(
        self,
        max_shift_sec: float = AUG_MAX_TIME_SHIFT_SEC,
        noise_std: float = AUG_NOISE_STD,
        stretch_range: Tuple[float, float] = AUG_TIME_STRETCH_RANGE,
        p_shift: float = 0.5,
        p_noise: float = 0.5,
        p_stretch: float = 0.5,
        seed: Optional[int] = None
    ):
        self.max_shift_sec = max_shift_sec
        self.noise_std = noise_std
        self.stretch_range = stretch_range
        self.p_shift = p_shift
        self.p_noise = p_noise
        self.p_stretch = p_stretch
        self.rng = np.random.default_rng(seed)

    def __call__(self, audio: np.ndarray, sr: int = MODULE1_TARGET_SAMPLE_RATE) -> np.ndarray:
        return self.augment(audio, sr=sr)

    def augment(self, audio: np.ndarray, sr: int = MODULE1_TARGET_SAMPLE_RATE) -> np.ndarray:
        """
        Applies stochastic augmentation chain, guaranteeing exact output length.
        """
        aug = audio.copy()
        orig_len = len(aug)
        
        if self.p_shift > 0 and self.rng.random() < self.p_shift:
            aug = time_shift(aug, sr=sr, max_shift_sec=self.max_shift_sec, rng=self.rng)
            
        if self.p_stretch > 0 and self.rng.random() < self.p_stretch:
            aug = time_stretch(aug, stretch_range=self.stretch_range, rng=self.rng)
            
        if self.p_noise > 0 and self.rng.random() < self.p_noise:
            aug = add_gaussian_noise(aug, std=self.noise_std, rng=self.rng)
            
        # Guarantee exact original shape
        if len(aug) != orig_len:
            if len(aug) > orig_len:
                aug = aug[:orig_len]
            else:
                aug = np.pad(aug, (0, orig_len - len(aug)))
                
        return aug.astype(np.float32)
