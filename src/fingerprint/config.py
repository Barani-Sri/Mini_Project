"""
Configuration for Heart Sound Fingerprinting.
Defines acoustic feature component dimensions, fusion weights, and versioning for reproducible acoustic fingerprints.
"""
from dataclasses import dataclass
from typing import Dict, Any

# Current Standard Fingerprint Version
FINGERPRINT_VERSION = "v1"

@dataclass
class FingerprintConfig:
    """
    Configuration for Heart Sound Fingerprint generation.
    
    Acoustic Dimensions Breakdown (Version 'v1'):
    - MFCC Component (120 dims): Captures temporal timbre dynamics and envelope cepstral shape
      derived from 40 static MFCCs + 40 velocity Deltas + 40 acceleration Delta-Deltas.
    - Log-Mel Component (128 dims): Captures full frequency-band acoustic power distribution
      across 128 critical Mel-bands in the 25-400Hz PCG diagnostic bandwidth.
    - AST Component (768 dims): Captures deep self-attention patch contextual representations
      from the pretrained Audio Spectrogram Transformer.
    
    Total Raw Composite Dimension = 120 + 128 + 768 = 1,016 dimensions.
    """
    version: str = FINGERPRINT_VERSION
    use_mfcc: bool = True
    use_mel: bool = True
    use_ast: bool = True
    
    # Sub-component dimensionality
    mfcc_dim: int = 120
    mel_dim: int = 128
    ast_dim: int = 768
    
    # Component weighting in fused vector
    weight_mfcc: float = 1.0
    weight_mel: float = 1.0
    weight_ast: float = 2.0
    
    # Normalization
    normalize_components: bool = True   # L2 normalize each sub-representation before fusion
    normalize_output: bool = True       # L2 normalize final fused fingerprint vector to unit sphere
    
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
            "components": {
                "mfcc": self.mfcc_dim if self.use_mfcc else 0,
                "mel": self.mel_dim if self.use_mel else 0,
                "ast": self.ast_dim if self.use_ast else 0
            },
            "weights": {
                "mfcc": self.weight_mfcc,
                "mel": self.weight_mel,
                "ast": self.weight_ast
            },
            "normalize_output": self.normalize_output
        }
