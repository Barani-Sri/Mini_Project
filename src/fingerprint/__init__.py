"""
Heart Sound Fingerprinting and Acoustic Similarity Module.
Provides fixed-dimensional, normalized acoustic fingerprints and cosine similarity metrics.
"""
from src.fingerprint.config import FingerprintConfig, FINGERPRINT_VERSION
from src.fingerprint.generator import (
    HeartSoundFingerprint,
    HeartSoundFingerprintGenerator,
    generate_fingerprint
)
from src.fingerprint.similarity import (
    AcousticSimilarityCalculator,
    calculate_acoustic_similarity
)

__all__ = [
    "FingerprintConfig",
    "FINGERPRINT_VERSION",
    "HeartSoundFingerprint",
    "HeartSoundFingerprintGenerator",
    "generate_fingerprint",
    "AcousticSimilarityCalculator",
    "calculate_acoustic_similarity"
]
