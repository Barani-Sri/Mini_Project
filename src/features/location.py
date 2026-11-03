"""
Multi-Location Representation, Patient-Level Grouping, and Fusion Encodings.
Encodes auscultation sites (AV, MV, PV, TV, Phc), creates missing-location masks,
and constructs (5, C, T) multi-location tensors for downstream fusion architectures.
"""
from typing import Dict, List, Tuple, Optional, Union
import numpy as np

from src.features.config import (
    LOCATION_TO_ID,
    ID_TO_LOCATION,
    ORDERED_LOCATIONS,
    NUM_LOCATIONS
)

def encode_location(location_str: str) -> int:
    """
    Deterministically maps auscultation string to integer ID (0 to 4).
    AV=0, MV=1, PV=2, TV=3, Phc=4.
    """
    clean_loc = location_str.strip()
    if clean_loc not in LOCATION_TO_ID:
        # Fallback for unexpected naming or suffixes
        for known_loc in ORDERED_LOCATIONS:
            if known_loc in clean_loc:
                return LOCATION_TO_ID[known_loc]
        return LOCATION_TO_ID["Phc"]
    return LOCATION_TO_ID[clean_loc]

def decode_location(location_id: int) -> str:
    """Decodes integer ID back to location string."""
    return ID_TO_LOCATION.get(location_id, "Unknown")

class MultiLocationPatientTensor:
    """
    Constructs multi-location representation tensors for a single patient.
    Fixed location order: [AV(0), MV(1), PV(2), TV(3), Phc(4)].
    """
    def __init__(
        self,
        num_channels: int = 120,
        num_time_frames: int = 501
    ):
        self.num_channels = num_channels
        self.num_time_frames = num_time_frames
        self.num_locations = NUM_LOCATIONS  # 5

    def build_patient_tensor(
        self,
        location_features_dict: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds a (5, C, T) tensor from available location features for a patient.
        Zero-pads missing locations and creates a (5,) location mask.
        
        Args:
            location_features_dict: Dict mapping location string ('AV', 'MV', etc.)
                                    to feature array of shape (C, T).
        Returns:
            patient_tensor: np.ndarray of shape (5, C, T)
            location_mask: np.ndarray of shape (5,) with 1 for present, 0 for missing.
        """
        tensor = np.zeros(
            (self.num_locations, self.num_channels, self.num_time_frames),
            dtype=np.float32
        )
        mask = np.zeros(self.num_locations, dtype=np.float32)

        for loc_name, feat in location_features_dict.items():
            loc_id = encode_location(loc_name)
            if loc_id < self.num_locations:
                # Ensure correct shape
                if feat.shape == (self.num_channels, self.num_time_frames):
                    tensor[loc_id] = feat
                    mask[loc_id] = 1.0
                elif len(feat.shape) == 3:
                    # If multiple windows for this location, take mean across windows
                    tensor[loc_id] = np.mean(feat, axis=0)
                    mask[loc_id] = 1.0

        return tensor, mask

    @staticmethod
    def format_early_fusion(patient_tensor: np.ndarray) -> np.ndarray:
        """
        Early Fusion: Flattens location dimension into channel dimension.
        Input: (5, C, T) -> Output: (5 * C, T)
        """
        n_locs, c, t = patient_tensor.shape
        return patient_tensor.reshape(n_locs * c, t).astype(np.float32)

    @staticmethod
    def format_intermediate_fusion(patient_tensor: np.ndarray) -> np.ndarray:
        """
        Intermediate Fusion: Retains explicit (5, C, T) representation
        for multi-branch valve encoders.
        """
        return patient_tensor.astype(np.float32)

    @staticmethod
    def format_attention_fusion(
        patient_tensor: np.ndarray,
        location_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Attention Fusion: Returns paired location features (5, C, T)
        and attention mask (5,).
        """
        return patient_tensor.astype(np.float32), location_mask.astype(np.float32)
