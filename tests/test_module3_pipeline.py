"""
Unit tests for Module 3 training, evaluation, explainability, and metrics calculation.
"""
import pytest
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

from src.models.config import Module3Config
from src.models.multi_location_net import MultiLocationHeartSoundNet
from src.models.trainer import Module3Trainer
from src.models.evaluator import compute_metrics_for_target, Module3Evaluator
from src.models.explainability import Module3Explainability

def test_compute_metrics_for_target():
    """Verify diagnostic metric computation with known synthetic inputs."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    metrics = compute_metrics_for_target(y_true, y_prob, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2

def test_trainer_smoke_test(tmp_path):
    """Verify trainer runs 2 epochs on synthetic micro-dataset and saves checkpoint."""
    cfg = Module3Config(
        max_epochs=2,
        batch_size=4,
        early_stopping_patience=5,
        enable_augmentation=False
    )
    model = MultiLocationHeartSoundNet(cfg)

    # Create synthetic dataset
    N = 8
    features = torch.randn(N, 5, 120, 501)
    masks = torch.ones(N, 5)
    y_m = torch.tensor([0, 1, 0, 1, 0, 0, 1, 0], dtype=torch.float32)
    y_o = torch.tensor([0, 1, 1, 1, 0, 1, 0, 0], dtype=torch.float32)
    pids = [f"P_{i}" for i in range(N)]

    class SyntheticDataset(torch.utils.data.Dataset):
        def __len__(self):
            return N
        def __getitem__(self, i):
            return {
                "patient_id": pids[i],
                "features": features[i],
                "location_mask": masks[i],
                "murmur_label": y_m[i],
                "outcome_label": y_o[i]
            }

    loader = DataLoader(SyntheticDataset(), batch_size=4)
    chk_path = tmp_path / "smoke_model.pt"

    trainer = Module3Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=cfg,
        model_save_path=chk_path
    )

    results = trainer.train(verbose=False)

    assert results["best_epoch"] in [1, 2]
    assert chk_path.exists()

def test_explainability_smoke_test():
    """Verify location attention analysis and ablation functions execute properly."""
    cfg = Module3Config(batch_size=4)
    model = MultiLocationHeartSoundNet(cfg)
    explainer = Module3Explainability(model, cfg)

    import pandas as pd
    fake_df = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3", "P4"],
        "true_murmur": [0, 1, 0, 1],
        "true_clinical_outcome": [0, 1, 1, 0],
        "attn_AV": [0.2, 0.4, 0.1, 0.3],
        "attn_MV": [0.3, 0.2, 0.4, 0.3],
        "attn_PV": [0.2, 0.2, 0.2, 0.2],
        "attn_TV": [0.2, 0.1, 0.2, 0.1],
        "attn_Phc": [0.1, 0.1, 0.1, 0.1]
    })

    attn_analysis = explainer.analyze_attention_weights(fake_df)
    assert "overall_mean_attention" in attn_analysis
    assert "AV" in attn_analysis["overall_mean_attention"]
    assert "MV" in attn_analysis["overall_mean_attention"]
