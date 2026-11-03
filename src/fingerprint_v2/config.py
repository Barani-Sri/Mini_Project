"""
Configuration for Heart Sound Fingerprint V2 (Temporal-Aware).
Defines temporal acoustic feature statistics, component dimensions, fusion weights, and versioning.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any

FINGERPRINT_V2_VERSION = "v2"

@dataclass
class FingerprintV2Config:
    """
    Configuration for Heart Sound Fingerprint V2 generation.
    
    Acoustic Dimensions Breakdown (Version 'v2'):
    - Temporal-Aware MFCC Component (240 dims):
      Derived from 40 static MFCCs + 40 velocity Deltas + 40 acceleration Delta-Deltas (120 channels),
      preserving both Mean (120) and Standard Deviation (120) across temporal frames.
    - Temporal-Aware Log-Mel Component (256 dims):
      Derived from 128 critical Mel-bands in the 25-400Hz PCG diagnostic bandwidth,
      preserving both Mean (128) and Standard Deviation (128) across temporal frames.
    - AST Component (768 dims):
      Pretrained Audio Spectrogram Transformer patch self-attention representations.
      
    Total Raw Composite Dimension = 240 + 256 + 768 = 1,264 dimensions.
    """
    version: str = FINGERPRINT_V2_VERSION
    use_mfcc: bool = True
    use_mel: bool = True
    use_ast: bool = True
    
    # Temporal statistical moments to compute ('mean', 'std')
    temporal_stats: Tuple[str, ...] = ("mean", "std")
    
    # Base dimensionalities per time-slice
    mfcc_base_dim: int = 120
    mel_base_dim: int = 128
    ast_dim: int = 768
    
    # Component weighting in fused vector (matching V1 baseline weighting)
    weight_mfcc: float = 1.0
    weight_mel: float = 1.0
    weight_ast: float = 2.0
    
    # Normalization strategies
    normalize_components: bool = True   # L2 normalize each sub-representation before fusion
    normalize_output: bool = True       # L2 normalize final fused fingerprint vector to unit sphere
    
    @property
    def mfcc_dim(self) -> int:
        return self.mfcc_base_dim * len(self.temporal_stats) if self.use_mfcc else 0

    @property
    def mel_dim(self) -> int:
        return self.mel_base_dim * len(self.temporal_stats) if self.use_mel else 0

    @property
    def total_dimension(self) -> int:
        dim = 0
        if self.use_mfcc:
            dim += self.mfcc_dim
        if self.use_mel:
            dim += self.mel_dim
        if self.use_ast:
            dim += self.ast_dim
        return dim

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "total_dimension": self.total_dimension,
            "temporal_stats": list(self.temporal_stats),
            "components": {
                "mfcc": self.mfcc_dim,
                "mel": self.mel_dim,
                "ast": self.ast_dim if self.use_ast else 0
            },
            "weights": {
                "mfcc": self.weight_mfcc,
                "mel": self.weight_mel,
                "ast": self.weight_ast
            },
            "normalize_components": self.normalize_components,
            "normalize_output": self.normalize_output
        }
