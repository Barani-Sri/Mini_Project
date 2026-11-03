"""
Unit tests for Module 3 Patient-Level Dataset and DataLoader.
"""
import pytest
import numpy as np
import torch
from pathlib import Path

from src.models.config import Module3Config, MODULE2_H5_PATH, MODULE3_METADATA_CSV_PATH
from src.models.dataset import CirCorPatientDataset, create_module3_dataloaders

def test_dataset_initialization():
    """Verify dataset loads successfully from module2_features.h5 for all splits."""
    if not MODULE2_H5_PATH.exists():
        pytest.skip(f"Skipping: {MODULE2_H5_PATH} does not exist.")

    for split, expected_count in [("train", 610), ("validation", 131), ("test", 132)]:
        ds = CirCorPatientDataset(
            h5_path=MODULE2_H5_PATH,
            split=split,
            metadata_csv_path=MODULE3_METADATA_CSV_PATH
        )
        assert len(ds) == expected_count, f"Expected {expected_count} patients in {split}, got {len(ds)}"

def test_dataset_item_structure_and_shapes():
    """Verify tensor shapes, types, and mask integrity for an individual patient item."""
    if not MODULE2_H5_PATH.exists():
        pytest.skip(f"Skipping: {MODULE2_H5_PATH} does not exist.")

    ds = CirCorPatientDataset(
        h5_path=MODULE2_H5_PATH,
        split="test",
        metadata_csv_path=MODULE3_METADATA_CSV_PATH
    )
    sample = ds[0]

    assert "patient_id" in sample
    assert "features" in sample
    assert "location_mask" in sample
    assert "murmur_label" in sample
    assert "outcome_label" in sample

    assert sample["features"].shape == (5, 120, 501)
    assert sample["location_mask"].shape == (5,)
    assert sample["features"].dtype == torch.float32
    assert sample["location_mask"].dtype == torch.float32
    assert sample["murmur_label"].item() in [0.0, 1.0]
    assert sample["outcome_label"].item() in [0.0, 1.0]
    assert torch.sum(sample["location_mask"]).item() >= 1.0

def test_no_patient_leakage_between_datasets():
    """Verify zero patient ID overlap across train, validation, and test datasets."""
    if not MODULE2_H5_PATH.exists():
        pytest.skip(f"Skipping: {MODULE2_H5_PATH} does not exist.")

    ds_train = CirCorPatientDataset(h5_path=MODULE2_H5_PATH, split="train")
    ds_val = CirCorPatientDataset(h5_path=MODULE2_H5_PATH, split="validation")
    ds_test = CirCorPatientDataset(h5_path=MODULE2_H5_PATH, split="test")

    train_pids = set(ds_train.patient_ids)
    val_pids = set(ds_val.patient_ids)
    test_pids = set(ds_test.patient_ids)

    assert len(train_pids.intersection(val_pids)) == 0, "Train-Val patient leakage detected!"
    assert len(train_pids.intersection(test_pids)) == 0, "Train-Test patient leakage detected!"
    assert len(val_pids.intersection(test_pids)) == 0, "Val-Test patient leakage detected!"

def test_dataloader_batching():
    """Verify DataLoader creates valid batched tensors."""
    if not MODULE2_H5_PATH.exists():
        pytest.skip(f"Skipping: {MODULE2_H5_PATH} does not exist.")

    cfg = Module3Config(batch_size=8)
    train_loader, val_loader, test_loader = create_module3_dataloaders(config=cfg)

    batch = next(iter(train_loader))
    assert batch["features"].shape == (8, 5, 120, 501)
    assert batch["location_mask"].shape == (8, 5)
    assert batch["murmur_label"].shape == (8,)
    assert batch["outcome_label"].shape == (8,)
    assert len(batch["patient_id"]) == 8
