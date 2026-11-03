"""
Module 2 Feature Extraction and Multi-Location Representations.
"""
from src.features.config import FeatureConfig, LOCATION_TO_ID, ID_TO_LOCATION, ORDERED_LOCATIONS, NUM_LOCATIONS
from src.features.melspectrogram import MelSpectrogramExtractor
from src.features.mfcc import MFCCExtractor
from src.features.quality import FeatureQualityAssessment, QualityMetrics
from src.features.location import encode_location, decode_location, MultiLocationPatientTensor
from src.features.pipeline import Module2Pipeline

__all__ = [
    "FeatureConfig",
    "LOCATION_TO_ID",
    "ID_TO_LOCATION",
    "ORDERED_LOCATIONS",
    "NUM_LOCATIONS",
    "MelSpectrogramExtractor",
    "MFCCExtractor",
    "FeatureQualityAssessment",
    "QualityMetrics",
    "encode_location",
    "decode_location",
    "MultiLocationPatientTensor",
    "Module2Pipeline"
]
