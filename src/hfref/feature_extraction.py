"""
Multimodal Feature Extraction for HFrEF Subjects.
Combines ECG features with multi-location PCG features across the 4 anatomical sites:
- Aortic (RUSB / AV)
- Pulmonary (LUSB / PV)
- Tricuspid (LLSB / TV)
- Mitral (APEX / MV)
Generates structured subject representations with location availability masks.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np

from src.hfref.config import (
    ORDERED_HFREF_LOCATIONS,
    AUSCULTATION_CODE_MAP,
    ECG_FEATURE_DIM,
    PCG_FEATURE_PER_LOC_DIM,
    TOTAL_FUSED_DIM
)
from src.hfref.ecg_processing import extract_ecg_features
from src.hfref.pcg_processing import extract_pcg_acoustic_features

@dataclass
class MultiModalSubjectFeatures:
    """Container for multimodal features extracted from a subject."""
    subject_id: str
    lvef: Optional[float]
    is_hfref: Optional[bool]
    fold: Optional[int]
    ecg_features: Dict[str, Any]
    pcg_location_features: Dict[str, Dict[str, Any]]
    fused_vector: np.ndarray
    location_mask: np.ndarray      # (5,) binary mask: [ECG, RUSB(AV), LUSB(PV), LLSB(TV), APEX(MV)]
    available_pcg_locations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Converts extracted features into JSON-serializable dictionary."""
        return {
            "subject_id": self.subject_id,
            "lvef": self.lvef,
            "is_hfref": self.is_hfref,
            "fold": self.fold,
            "ecg_metrics": {
                k: v for k, v in self.ecg_features.items() if k != "feature_vector"
            },
            "location_metrics": {
                site: {
                    "valve_name": AUSCULTATION_CODE_MAP[site]["valve_name"],
                    "lubdub_code": AUSCULTATION_CODE_MAP[site]["lubdub_code"],
                    "metrics": {
                        k: v for k, v in feats.items() if k != "feature_vector"
                    }
                }
                for site, feats in self.pcg_location_features.items()
            },
            "location_mask": self.location_mask.tolist(),
            "fused_vector_dim": int(len(self.fused_vector)),
            "available_pcg_locations": self.available_pcg_locations
        }


def extract_multimodal_features(
    subject_signals: Dict[str, Any]
) -> MultiModalSubjectFeatures:
    """
    Extracts multimodal features from decoded subject signals.
    Args:
        subject_signals: dict returned by HFrEFDatasetLoader.load_subject_signals()
                         or constructed dynamically from uploaded recordings.
    """
    sid = str(subject_signals.get("subject_id", "unknown"))
    lvef = subject_signals.get("lvef")
    is_hfref = subject_signals.get("is_hfref")
    fold = subject_signals.get("fold")

    ecg_signals = subject_signals.get("ecg_signals", {})
    pcg_signals = subject_signals.get("pcg_signals", {})

    # 1. ECG feature extraction
    # Find primary ECG signal
    primary_ecg_data = None
    primary_ecg_sr = 500
    for site in ["APEX", "RUSB", "LLSB", "LUSB"]:
        if site in ecg_signals:
            primary_ecg_data = ecg_signals[site]["signal"]
            primary_ecg_sr = ecg_signals[site]["sr"]
            break

    if primary_ecg_data is not None and len(primary_ecg_data) > 0:
        ecg_feats = extract_ecg_features(primary_ecg_data, fs=primary_ecg_sr)
        ecg_vec = ecg_feats["feature_vector"]
        ecg_available = 1.0
    else:
        ecg_feats = {
            "feature_vector": np.zeros(ECG_FEATURE_DIM, dtype=np.float32),
            "heart_rate_bpm": 0.0,
            "rr_mean_ms": 0.0,
            "sdnn_ms": 0.0,
            "rmssd_ms": 0.0,
            "pnn50_percent": 0.0,
            "r_peaks_count": 0,
            "energy": 0.0
        }
        ecg_vec = ecg_feats["feature_vector"]
        ecg_available = 0.0

    # 2. Multi-location PCG feature extraction
    pcg_loc_feats = {}
    pcg_vectors = []
    pcg_masks = []
    avail_locs = []

    for site in ORDERED_HFREF_LOCATIONS: # ["RUSB", "LUSB", "LLSB", "APEX"]
        if site in pcg_signals and len(pcg_signals[site]["signal"]) > 0:
            sig = pcg_signals[site]["signal"]
            sr = pcg_signals[site]["sr"]
            p_feats = extract_pcg_acoustic_features(sig, sr=sr)
            pcg_loc_feats[site] = p_feats
            pcg_vectors.append(p_feats["feature_vector"])
            pcg_masks.append(1.0)
            avail_locs.append(site)
        else:
            p_feats = {
                "feature_vector": np.zeros(PCG_FEATURE_PER_LOC_DIM, dtype=np.float32),
                "rms_energy": 0.0,
                "spectral_centroid_hz": 0.0,
                "spectral_bandwidth_hz": 0.0,
                "zero_crossing_rate": 0.0,
                "shannon_energy_mean": 0.0
            }
            pcg_loc_feats[site] = p_feats
            pcg_vectors.append(p_feats["feature_vector"])
            pcg_masks.append(0.0)

    # 3. Concatenate into fused feature vector
    fused = np.concatenate([ecg_vec] + pcg_vectors).astype(np.float32)
    loc_mask = np.array([ecg_available] + pcg_masks, dtype=np.float32)

    return MultiModalSubjectFeatures(
        subject_id=sid,
        lvef=lvef,
        is_hfref=is_hfref,
        fold=fold,
        ecg_features=ecg_feats,
        pcg_location_features=pcg_loc_feats,
        fused_vector=fused,
        location_mask=loc_mask,
        available_pcg_locations=avail_locs
    )
