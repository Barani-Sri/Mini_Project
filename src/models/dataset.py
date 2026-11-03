"""
Module 3 Patient-Level Dataset and DataLoader for Multi-Location Heart Sound PCG features.
Loads pre-aggregated (5, 120, T) patient acoustic tensors and location masks from module2_features.h5.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

from src.models.config import (
    Module3Config,
    MODULE2_H5_PATH,
    MODULE3_METADATA_CSV_PATH
)

class CirCorPatientDataset(Dataset):
    """
    Patient-level Dataset for Multi-Location Heart Sound representations.
    Loads patient multi-location tensors (5, 120, 501), location masks (5,),
    Murmur ground truth (binary: 0=Absent, 1=Present), and Clinical Outcome (binary: 0=Normal, 1=Abnormal).
    """
    def __init__(
        self,
        h5_path: Union[str, Path] = MODULE2_H5_PATH,
        split: str = "train",
        metadata_csv_path: Optional[Union[str, Path]] = MODULE3_METADATA_CSV_PATH,
        config: Optional[Module3Config] = None,
        is_training: bool = False
    ):
        self.h5_path = Path(h5_path)
        self.split = split.lower()
        self.config = config if config is not None else Module3Config()
        self.is_training = is_training
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"Missing Module 2 features HDF5 at {self.h5_path}")

        # Load outcome mapping from metadata CSV if available
        self.patient_outcome_map: Dict[str, int] = {}
        if metadata_csv_path is not None and Path(metadata_csv_path).exists():
            df = pd.read_csv(metadata_csv_path)
            if "patient_id" in df.columns and "outcome_binary" in df.columns:
                pat_df = df.drop_duplicates(subset=["patient_id"])
                for _, row in pat_df.iterrows():
                    pid = str(row["patient_id"])
                    ob = int(row["outcome_binary"])
                    self.patient_outcome_map[pid] = ob

        # Pre-load in-memory patient arrays for the specific split (~150-730 MB, ultra-fast access)
        with h5py.File(self.h5_path, "r") as f:
            if "patients" not in f or self.split not in f["patients"]:
                raise KeyError(f"Patient split '{self.split}' not found under /patients in {self.h5_path}")
            
            p_grp = f["patients"][self.split]
            self.features = np.array(p_grp["patient_features"][:], dtype=np.float32)   # (N, 5, 120, 501)
            self.masks = np.array(p_grp["location_masks"][:], dtype=np.float32)         # (N, 5)
            self.murmur_labels = np.array(p_grp["patient_labels"][:], dtype=np.int64)   # (N,)
            pids_raw = p_grp["patient_ids"][:]
            self.patient_ids = [p.decode("utf-8") if isinstance(p, bytes) else str(p) for p in pids_raw]

        # Construct Outcome labels array aligned with patient_ids
        outcome_list = []
        for pid in self.patient_ids:
            outcome_val = self.patient_outcome_map.get(pid, 0)
            outcome_list.append(outcome_val)
        self.outcome_labels = np.array(outcome_list, dtype=np.int64)

        self.num_patients = len(self.patient_ids)

    def __len__(self) -> int:
        return self.num_patients

    def _apply_augmentation(self, tensor: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Applies SpecAugment-style time masking, channel masking, and gentle noise to available locations.
        tensor shape: (5, 120, 501)
        """
        aug_tensor = tensor.copy()
        
        # Apply only to available auscultation locations (mask == 1)
        for loc_idx in range(tensor.shape[0]):
            if mask[loc_idx] == 0:
                continue
                
            feat = aug_tensor[loc_idx] # (120, 501)
            
            # Time masking
            if np.random.rand() < 0.5:
                t_mask_len = np.random.randint(5, self.config.time_mask_max_frames + 1)
                t_start = np.random.randint(0, max(1, self.config.time_frames - t_mask_len))
                feat[:, t_start:t_start + t_mask_len] = 0.0
                
            # Channel (Frequency) masking
            if np.random.rand() < 0.5:
                c_mask_len = np.random.randint(2, self.config.freq_mask_max_channels + 1)
                c_start = np.random.randint(0, max(1, self.config.in_channels - c_mask_len))
                feat[c_start:c_start + c_mask_len, :] = 0.0
                
            # Additive Gaussian noise
            if np.random.rand() < 0.3:
                noise = np.random.normal(0.0, self.config.noise_std, size=feat.shape).astype(np.float32)
                feat += noise

            aug_tensor[loc_idx] = feat
            
        return aug_tensor

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        feature = self.features[idx] # (5, 120, 501)
        mask = self.masks[idx]       # (5,)
        murmur_label = self.murmur_labels[idx]
        outcome_label = self.outcome_labels[idx]
        patient_id = self.patient_ids[idx]

        if self.is_training and self.config.enable_augmentation:
            feature = self._apply_augmentation(feature, mask)

        return {
            "patient_id": patient_id,
            "features": torch.from_numpy(feature).float(),         # (5, 120, 501)
            "location_mask": torch.from_numpy(mask).float(),       # (5,)
            "murmur_label": torch.tensor(murmur_label, dtype=torch.float32),
            "outcome_label": torch.tensor(outcome_label, dtype=torch.float32)
        }

def create_module3_dataloaders(
    h5_path: Union[str, Path] = MODULE2_H5_PATH,
    metadata_csv_path: Optional[Union[str, Path]] = MODULE3_METADATA_CSV_PATH,
    config: Optional[Module3Config] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train, validation, and test patient splits.
    """
    cfg = config if config is not None else Module3Config()

    train_dataset = CirCorPatientDataset(
        h5_path=h5_path,
        split="train",
        metadata_csv_path=metadata_csv_path,
        config=cfg,
        is_training=True
    )
    val_dataset = CirCorPatientDataset(
        h5_path=h5_path,
        split="validation",
        metadata_csv_path=metadata_csv_path,
        config=cfg,
        is_training=False
    )
    test_dataset = CirCorPatientDataset(
        h5_path=h5_path,
        split="test",
        metadata_csv_path=metadata_csv_path,
        config=cfg,
        is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader, test_loader
