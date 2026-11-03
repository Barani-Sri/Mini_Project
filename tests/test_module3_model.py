"""
Unit tests for Module 3 Multi-Location Neural Network and Attention Fusion.
"""
import pytest
import torch
import numpy as np

from src.models.config import Module3Config
from src.models.multi_location_net import (
    SharedLocationEncoder,
    MaskedLocationAttentionFusion,
    MultiLocationHeartSoundNet
)

def test_shared_location_encoder_forward():
    """Verify shared 2D CNN location encoder maps (B, 120, 501) to (B, latent_dim)."""
    cfg = Module3Config(latent_dim=128)
    encoder = SharedLocationEncoder(cfg)
    encoder.eval()

    # Batch of 4 single-location spectrograms
    x = torch.randn(4, 120, 501)
    with torch.no_grad():
        out = encoder(x)

    assert out.shape == (4, 128)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

def test_masked_location_attention_normalization_and_masking():
    """Verify masked attention correctly zeroes out missing locations and normalizes to 1."""
    cfg = Module3Config(latent_dim=128, attention_dim=128)
    fusion = MaskedLocationAttentionFusion(cfg)
    fusion.eval()

    embeddings = torch.randn(3, 5, 128)
    # Patient 0 has AV, MV, PV (TV, Phc missing)
    # Patient 1 has all 5 locations
    # Patient 2 has only MV
    masks = torch.tensor([
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 0.0, 0.0]
    ], dtype=torch.float32)

    with torch.no_grad():
        fused, attn_weights = fusion(embeddings, masks)

    assert fused.shape == (3, 128)
    assert attn_weights.shape == (3, 5)

    # Attention weights must sum to 1.0 per patient
    sums = torch.sum(attn_weights, dim=1)
    assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    # Missing locations must have near-zero attention weight
    assert attn_weights[0, 3].item() < 1e-4
    assert attn_weights[0, 4].item() < 1e-4
    assert attn_weights[2, 0].item() < 1e-4
    assert attn_weights[2, 1].item() > 0.999 # Only location available

def test_multi_location_heart_sound_net_forward_and_gradients():
    """Verify end-to-end forward pass, output shapes, and backward gradient flow."""
    cfg = Module3Config(batch_size=2, latent_dim=128)
    model = MultiLocationHeartSoundNet(cfg)
    model.train()

    features = torch.randn(2, 5, 120, 501, requires_grad=True)
    masks = torch.tensor([[1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0, 0.0]], dtype=torch.float32)

    outputs = model(features, masks)

    assert "murmur_logits" in outputs
    assert "outcome_logits" in outputs
    assert "murmur_probs" in outputs
    assert "outcome_probs" in outputs
    assert "attention_weights" in outputs
    assert "fused_patient_repr" in outputs

    assert outputs["murmur_logits"].shape == (2, 1)
    assert outputs["outcome_logits"].shape == (2, 1)
    assert outputs["murmur_probs"].shape == (2, 1)
    assert outputs["outcome_probs"].shape == (2, 1)
    assert outputs["attention_weights"].shape == (2, 5)
    assert outputs["fused_patient_repr"].shape == (2, 128)

    # Probabilities in range [0, 1]
    assert (outputs["murmur_probs"] >= 0.0).all() and (outputs["murmur_probs"] <= 1.0).all()
    assert (outputs["outcome_probs"] >= 0.0).all() and (outputs["outcome_probs"] <= 1.0).all()

    # Backward pass
    loss = outputs["murmur_logits"].sum() + outputs["outcome_logits"].sum()
    loss.backward()

    assert features.grad is not None
    assert not torch.isnan(features.grad).any()
