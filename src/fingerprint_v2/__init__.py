"""
Heart Sound Fingerprint V2 (Temporal-Aware) Package.
Provides temporal-aware acoustic fingerprinting and acoustic similarity comparison.
"""
from src.fingerprint_v2.config import FingerprintV2Config, FINGERPRINT_V2_VERSION
from src.fingerprint_v2.fingerprint_v2_generator import (
    HeartSoundFingerprintV2,
    HeartSoundFingerprintV2Generator,
    generate_fingerprint_v2
)
from src.fingerprint_v2.fingerprint_v2_similarity import (
    AcousticSimilarityCalculatorV2,
    calculate_acoustic_similarity_v2
)

__all__ = [
    "FingerprintV2Config",
    "FINGERPRINT_V2_VERSION",
    "HeartSoundFingerprintV2",
    "HeartSoundFingerprintV2Generator",
    "generate_fingerprint_v2",
    "AcousticSimilarityCalculatorV2",
    "calculate_acoustic_similarity_v2"
]
