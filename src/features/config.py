"""
Configuration for Module 2: Feature Extraction and Multi-Location Representation.
Defines parameters for STFT, Mel Spectrogram, MFCC, Deltas, and Multi-Location Tensors.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Dict

from src.utils.config import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
    MODULE1_H5_PATH,
    MODULE1_TARGET_SAMPLE_RATE,
    MODULE1_WINDOW_SAMPLES
)

# Output Paths for Module 2
MODULE2_H5_PATH = PROCESSED_DATA_DIR / "module2_features.h5"
MODULE2_CHECKPOINT_H5_PATH = PROCESSED_DATA_DIR / "module2_features.checkpoint.h5"
MODULE2_CHECKPOINT_META_PATH = PROCESSED_DATA_DIR / "module2_features.checkpoint.json"
MODULE2_SUMMARY_PATH = PROCESSED_DATA_DIR / "module2_summary.json"
MODULE2_ERRORS_PATH = PROCESSED_DATA_DIR / "module2_errors.csv"
MODULE2_REPORTS_DIR = BASE_DIR / "reports" / "module2"

# Deterministic Location Mapping
LOCATION_TO_ID: Dict[str, int] = {
    "AV": 0,
    "MV": 1,
    "PV": 2,
    "TV": 3,
    "Phc": 4
}
ID_TO_LOCATION: Dict[int, str] = {v: k for k, v in LOCATION_TO_ID.items()}
ORDERED_LOCATIONS: List[str] = ["AV", "MV", "PV", "TV", "Phc"]
NUM_LOCATIONS: int = len(ORDERED_LOCATIONS)  # 5

@dataclass
class FeatureConfig:
    """
    Feature extraction and representation configuration.
    """
    sample_rate: int = MODULE1_TARGET_SAMPLE_RATE  # 2000 Hz
    waveform_samples: int = MODULE1_WINDOW_SAMPLES  # 10,000 samples (5.0s)
    
    # MFCC & Mel Spectrogram Parameters
    n_mfcc: int = 40
    n_mels: int = 128
    frame_length_ms: float = 30.0
    hop_length_ms: float = 10.0
    f_min: float = 25.0
    f_max: float = 400.0
    n_fft: int = 512
    window_function: str = "hann"
    mel_power: float = 2.0
    top_db: float = 80.0
    
    # Delta Configuration
    include_delta: bool = True
    include_delta_delta: bool = True
    
    # Normalization & Quality Assessment
    normalization: str = "per_window"  # "per_window", "none"
    quality_threshold_good: float = 0.70
    quality_threshold_questionable: float = 0.30
    
    # Pipeline Processing & Checkpointing
    batch_size: int = 64
    compression: str = "gzip"
    compression_opts: int = 4
    flush_interval_batches: int = 1
    resume: bool = True
    
    # Derived Properties
    @property
    def frame_length_samples(self) -> int:
        return int((self.frame_length_ms / 1000.0) * self.sample_rate)  # 60 samples
        
    @property
    def hop_length_samples(self) -> int:
        return int((self.hop_length_ms / 1000.0) * self.sample_rate)    # 20 samples
        
    @property
    def num_time_frames(self) -> int:
        # With librosa center=True on 10,000 samples and hop=20 -> 1 + 10,000 // 20 = 501
        return 1 + (self.waveform_samples // self.hop_length_samples)
        
    @property
    def combined_mfcc_channels(self) -> int:
        multiplier = 1 + int(self.include_delta) + int(self.include_delta_delta)
        return self.n_mfcc * multiplier  # 40 * 3 = 120
