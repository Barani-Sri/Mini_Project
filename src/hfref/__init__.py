"""
LubDub HFrEF Multi-Modal ECG + Multi-Location PCG Module.
Independent analysis package for Heart Failure with Reduced Ejection Fraction (HFrEF)
and specialist clinical recommendations alongside the existing LubDub Murmur system.
"""

from src.hfref.config import (
    HFREF_DIR,
    HFREF_ZIP_PATH,
    HFREF_EXTRACTED_DIR,
    HFREF_ECG_SAMPLE_RATE,
    HFREF_PCG_SAMPLE_RATE,
    HFREF_PCG_TARGET_SR,
    AUSCULTATION_CODE_MAP,
    REVERSE_CODE_MAP
)

from src.hfref.dataset_loader import (
    HFrEFDatasetLoader,
    HFrEFSubjectRecord,
    get_dataset_loader
)

from src.hfref.ecg_processing import (
    preprocess_ecg,
    extract_ecg_features
)

from src.hfref.pcg_processing import (
    preprocess_hfref_pcg,
    extract_pcg_acoustic_features
)

from src.hfref.feature_extraction import (
    extract_multimodal_features,
    MultiModalSubjectFeatures
)

from src.hfref.fusion import (
    fuse_ecg_pcg_features
)

from src.hfref.analysis import (
    HFrEFAnalysisEngine,
    get_hfref_analyzer
)

from src.hfref.recommendation import (
    SpecialistRecommender,
    get_specialist_recommender
)

__all__ = [
    "HFREF_DIR",
    "HFREF_ZIP_PATH",
    "HFREF_EXTRACTED_DIR",
    "HFREF_ECG_SAMPLE_RATE",
    "HFREF_PCG_SAMPLE_RATE",
    "HFREF_PCG_TARGET_SR",
    "AUSCULTATION_CODE_MAP",
    "REVERSE_CODE_MAP",
    "HFrEFDatasetLoader",
    "HFrEFSubjectRecord",
    "get_dataset_loader",
    "preprocess_ecg",
    "extract_ecg_features",
    "preprocess_hfref_pcg",
    "extract_pcg_acoustic_features",
    "extract_multimodal_features",
    "MultiModalSubjectFeatures",
    "fuse_ecg_pcg_features",
    "HFrEFAnalysisEngine",
    "get_hfref_analyzer",
    "SpecialistRecommender",
    "get_specialist_recommender"
]
