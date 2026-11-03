"""
Multimodal ECG + Multi-Location PCG Fusion.
Combines electrical activity (ECG) with mechanical cardiac acoustics (4 valve locations)
using masked feature weighting and location attribution.
"""
from typing import Dict, List, Tuple, Any
import numpy as np

from src.hfref.config import (
    ORDERED_HFREF_LOCATIONS,
    AUSCULTATION_CODE_MAP,
    ECG_FEATURE_DIM,
    PCG_FEATURE_PER_LOC_DIM
)
from src.hfref.feature_extraction import MultiModalSubjectFeatures

def compute_location_energy_weights(
    multimodal_features: MultiModalSubjectFeatures
) -> Dict[str, Dict[str, Any]]:
    """
    Computes normalized relative contribution/activity weights
    for ECG and each of the 4 auscultation locations.
    """
    mask = multimodal_features.location_mask # [ECG, RUSB, LUSB, LLSB, APEX]
    energies = []

    # ECG energy
    ecg_energy = float(multimodal_features.ecg_features.get("energy", 0.0))
    energies.append(ecg_energy if mask[0] > 0 else 0.0)

    # 4 PCG location energies
    for i, site in enumerate(ORDERED_HFREF_LOCATIONS):
        if mask[i + 1] > 0:
            rms = float(multimodal_features.pcg_location_features[site].get("rms_energy", 0.0))
            shannon = float(multimodal_features.pcg_location_features[site].get("shannon_energy_mean", 0.0))
            combined_e = rms * 0.7 + shannon * 0.3
            energies.append(combined_e)
        else:
            energies.append(0.0)

    energies_arr = np.array(energies, dtype=np.float32)
    total = np.sum(energies_arr)
    if total > 1e-9:
        weights = energies_arr / total
    else:
        # Equal distribution across available channels
        active_count = np.sum(mask)
        weights = mask / active_count if active_count > 0 else np.ones(5) / 5.0

    weights_dict = {
        "ECG": {
            "weight": float(weights[0]),
            "percent": round(float(weights[0]) * 100, 2),
            "is_active": bool(mask[0] > 0)
        }
    }

    for i, site in enumerate(ORDERED_HFREF_LOCATIONS):
        info = AUSCULTATION_CODE_MAP[site]
        weights_dict[site] = {
            "valve_name": info["valve_name"],
            "lubdub_code": info["lubdub_code"],
            "weight": float(weights[i + 1]),
            "percent": round(float(weights[i + 1]) * 100, 2),
            "is_active": bool(mask[i + 1] > 0)
        }

    return weights_dict


def fuse_ecg_pcg_features(
    multimodal_features: MultiModalSubjectFeatures
) -> Dict[str, Any]:
    """
    Fuses multimodal features into an analysis-ready representation
    with channel weights and clinical valve attributes.
    """
    weights = compute_location_energy_weights(multimodal_features)
    
    return {
        "subject_id": multimodal_features.subject_id,
        "lvef": multimodal_features.lvef,
        "is_hfref": multimodal_features.is_hfref,
        "fold": multimodal_features.fold,
        "fused_vector": multimodal_features.fused_vector,
        "location_mask": multimodal_features.location_mask,
        "channel_weights": weights,
        "available_locations_count": int(np.sum(multimodal_features.location_mask))
    }
