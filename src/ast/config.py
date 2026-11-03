"""
Configuration for Audio Spectrogram Transformer (AST) Representation Branch.
Defines parameters for AST model loading, audio preprocessing, and embedding extraction.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal
import torch

from src.utils.config import BASE_DIR, PROCESSED_DATA_DIR, MODULE1_TARGET_SAMPLE_RATE

# Default AST Pretrained Model Identifier
DEFAULT_AST_MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

@dataclass
class ASTConfig:
    """
    Configuration settings for Audio Spectrogram Transformer (AST) branch.
    """
    model_name: str = DEFAULT_AST_MODEL_NAME
    target_sample_rate: int = 16000           # AST standard sampling rate: 16 kHz
    input_sample_rate: int = MODULE1_TARGET_SAMPLE_RATE  # Module 1 rate: 2000 Hz
    num_mel_bins: int = 128                   # Frequency bins for AST
    max_length: int = 1024                    # Time frames for AST STFT
    embedding_dim: int = 768                  # Fixed hidden size of ViT/AST backbone
    pooling_method: Literal["mean", "cls", "pooled"] = "mean"
    normalize_embedding: bool = True          # L2 unit normalization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dir: Optional[Path] = None
