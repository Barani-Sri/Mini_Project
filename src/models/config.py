"""
Configuration for Module 3: Clinical Outcome Prediction & Heart Murmur Detection.
Defines parameters for multi-location neural network, multi-task learning, training, and evaluation.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

from src.utils.config import (
    BASE_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    MODULE1_H5_PATH,
    AUSCULTATION_LOCATIONS
)

# Output & Artifact Paths for Module 3
MODULE2_H5_PATH = PROCESSED_DATA_DIR / "module2_features.h5"
MODULE3_METADATA_CSV_PATH = PROCESSED_DATA_DIR / "circor_features.csv"
MODULE3_MODEL_PATH = MODELS_DIR / "module3_best_model.pt"
MODULE3_PROCESSED_MODEL_PATH = PROCESSED_DATA_DIR / "module3_best_model.pt"
MODULE3_PREDICTIONS_PATH = PROCESSED_DATA_DIR / "module3_patient_predictions.csv"
MODULE3_RESULTS_JSON_PATH = PROCESSED_DATA_DIR / "module3_results.json"
MODULE3_REPORT_MD_PATH = PROCESSED_DATA_DIR / "MODULE3_REPORT.md"
MODULE3_RESULTS_DIR = PROCESSED_DATA_DIR / "module3_results"

@dataclass
class Module3Config:
    """
    Module 3 Model, Training, and Evaluation Configuration.
    """
    # Multi-Location Tensor Dimensions
    num_locations: int = 5                  # AV, MV, PV, TV, Phc
    in_channels: int = 120                  # Static (40) + Delta (40) + Delta-Delta (40)
    time_frames: int = 501                  # T = 501 frames (5.0s @ 2000Hz, hop=20)
    
    # Shared Location Feature Encoder
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    latent_dim: int = 128                   # Feature representation dimension per location
    encoder_dropout: float = 0.25
    
    # Masked Multi-Head Location Attention Fusion
    num_attention_heads: int = 4
    attention_dim: int = 128
    fusion_dropout: float = 0.25
    
    # Multi-Task Prediction Heads
    murmur_head_hidden: int = 64
    outcome_head_hidden: int = 64
    head_dropout: float = 0.30
    
    # Training Parameters
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 60
    early_stopping_patience: int = 15
    min_delta: float = 1e-4
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    min_lr: float = 1e-6
    
    # Loss Weighting (Calculated strictly from Train split: 485 Absent / 125 Present; 322 Normal / 288 Abnormal)
    murmur_pos_weight: float = 3.88         # 485 / 125
    outcome_pos_weight: float = 1.118       # 322 / 288
    task_weight_murmur: float = 1.0
    task_weight_outcome: float = 1.0
    
    # Regularization & Data Augmentation (Training split only)
    enable_augmentation: bool = True
    time_mask_max_frames: int = 30
    freq_mask_max_channels: int = 12
    noise_std: float = 0.01
    
    # Decision Thresholds
    murmur_threshold: float = 0.50
    outcome_threshold: float = 0.50
    
    # Reproducibility
    random_seed: int = 42
    
    # Device
    device: str = "cpu"                     # "cuda" or "cpu"
