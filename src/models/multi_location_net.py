"""
Multi-Location Deep Learning Architecture for Heart Sound Murmur & Clinical Outcome Detection.
Implements:
1. Shared 2D CNN Location Feature Encoder
2. Masked Multi-Head / Scaled Dot-Product Location Attention Fusion
3. Multi-Task Diagnostic Prediction Heads (Murmur Detection & Clinical Outcome)
"""
from typing import Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.config import Module3Config

class ConvBlock2D(nn.Module):
    """
    Standard 2D Convolutional Block with BatchNorm, SiLU activation, and Dropout.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int] = (3, 3),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (1, 1),
        dropout: float = 0.2
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.bn(self.conv(x))))


class SharedLocationEncoder(nn.Module):
    """
    Shared 2D Convolutional Encoder applied identically across all auscultation sites (AV, MV, PV, TV, Phc).
    Input:  (B * L, 1, 120, 501) or (B * L, 120, 501)
    Output: (B * L, latent_dim)
    """
    def __init__(self, config: Optional[Module3Config] = None):
        super().__init__()
        self.config = config if config is not None else Module3Config()
        
        # Block 1: (1, 120, 501) -> (32, 60, 250) -> Pool -> (32, 30, 125)
        self.block1 = nn.Sequential(
            ConvBlock2D(1, 32, kernel_size=(3, 5), stride=(2, 2), padding=(1, 2), dropout=self.config.encoder_dropout),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        # Block 2: (32, 30, 125) -> (64, 15, 62) -> Pool -> (64, 7, 31)
        self.block2 = nn.Sequential(
            ConvBlock2D(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dropout=self.config.encoder_dropout),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        # Block 3: (64, 7, 31) -> (128, 7, 31)
        self.block3 = nn.Sequential(
            ConvBlock2D(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout=self.config.encoder_dropout),
            ConvBlock2D(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout=self.config.encoder_dropout)
        )
        
        # Adaptive pooling & linear projection
        self.adaptive_pool_avg = nn.AdaptiveAvgPool2d((1, 1))
        self.adaptive_pool_max = nn.AdaptiveMaxPool2d((1, 1))
        
        self.projection = nn.Sequential(
            nn.Linear(128 * 2, self.config.latent_dim),
            nn.LayerNorm(self.config.latent_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(p=self.config.encoder_dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is (N, 120, 501), add channel dim -> (N, 1, 120, 501)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        
        h_avg = self.adaptive_pool_avg(h).flatten(1)
        h_max = self.adaptive_pool_max(h).flatten(1)
        h_cat = torch.cat([h_avg, h_max], dim=1) # (N, 256)
        
        return self.projection(h_cat) # (N, latent_dim)


class MaskedLocationAttentionFusion(nn.Module):
    """
    Interpretable Masked Attention Fusion over Auscultation Locations.
    Learns diagnostic attention weights across available cardiac valve locations.
    Missing locations (mask=0) are masked with -inf prior to softmax.
    """
    def __init__(self, config: Optional[Module3Config] = None):
        super().__init__()
        self.config = config if config is not None else Module3Config()
        
        self.query_proj = nn.Linear(self.config.latent_dim, self.config.attention_dim)
        self.key_proj = nn.Linear(self.config.latent_dim, self.config.attention_dim)
        self.value_proj = nn.Linear(self.config.latent_dim, self.config.latent_dim)
        
        # Global learnable diagnostic query vector
        self.global_query = nn.Parameter(torch.randn(1, 1, self.config.attention_dim))
        
        self.dropout = nn.Dropout(p=self.config.fusion_dropout)
        self.layer_norm = nn.LayerNorm(self.config.latent_dim)

    def forward(
        self,
        location_embeddings: torch.Tensor,
        location_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        location_embeddings: (B, 5, latent_dim)
        location_mask:       (B, 5) where 1 = available, 0 = missing
        
        Returns:
            fused_patient_vector: (B, latent_dim)
            attention_weights:    (B, 5) normalized attention over locations
        """
        B, L, D = location_embeddings.shape
        
        # Keys and Values
        K = self.key_proj(location_embeddings)   # (B, 5, attention_dim)
        V = self.value_proj(location_embeddings) # (B, 5, latent_dim)
        
        # Expand global query for batch: (B, 1, attention_dim)
        Q = self.global_query.expand(B, -1, -1)
        
        # Scaled dot-product attention: (B, 1, 5)
        scores = torch.bmm(Q, K.transpose(1, 2)) / np.sqrt(self.config.attention_dim)
        scores = scores.squeeze(1) # (B, 5)
        
        # Apply location mask: replace missing locations (mask == 0) with large negative value
        mask_bool = location_mask > 0.5
        # If a patient has all masks 0 (edge case safeguard), keep at least first location
        all_zeros = (~mask_bool.any(dim=1, keepdim=True)).expand_as(mask_bool)
        effective_mask = mask_bool | all_zeros
        
        scores = scores.masked_fill(~effective_mask, -1e9)
        
        # Softmax over available locations -> (B, 5)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights (during training)
        attn_dropped = self.dropout(attention_weights)
        
        # Weighted sum of location representations: (B, 1, 5) x (B, 5, latent_dim) -> (B, latent_dim)
        fused = torch.bmm(attn_dropped.unsqueeze(1), V).squeeze(1)
        
        fused = self.layer_norm(fused)
        
        return fused, attention_weights


class MultiLocationHeartSoundNet(nn.Module):
    """
    End-to-End Multi-Location Multi-Task Neural Network for CirCor Heart Sound PCG analysis.
    Predicts:
    1. Heart Murmur Presence (Binary)
    2. Clinical Outcome (Binary)
    With interpretable location attention weights.
    """
    def __init__(self, config: Optional[Module3Config] = None):
        super().__init__()
        self.config = config if config is not None else Module3Config()
        
        # Shared 2D CNN Encoder
        self.location_encoder = SharedLocationEncoder(self.config)
        
        # Masked Location Attention Fusion
        self.fusion = MaskedLocationAttentionFusion(self.config)
        
        # Multi-Task Prediction Head 1: Murmur Detection (Logit)
        self.murmur_head = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.murmur_head_hidden),
            nn.BatchNorm1d(self.config.murmur_head_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p=self.config.head_dropout),
            nn.Linear(self.config.murmur_head_hidden, 1)
        )
        
        # Multi-Task Prediction Head 2: Clinical Outcome Prediction (Logit)
        self.outcome_head = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.outcome_head_hidden),
            nn.BatchNorm1d(self.config.outcome_head_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(p=self.config.head_dropout),
            nn.Linear(self.config.outcome_head_hidden, 1)
        )

    def forward(
        self,
        features: torch.Tensor,
        location_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        features:      (B, 5, 120, 501)
        location_mask: (B, 5)
        
        Returns dictionary:
            murmur_logits:       (B, 1)
            outcome_logits:      (B, 1)
            murmur_probs:        (B, 1)
            outcome_probs:       (B, 1)
            attention_weights:   (B, 5)
            fused_patient_repr:  (B, latent_dim)
        """
        B, L, C, T = features.shape
        
        # Flatten batch and locations to pass through shared encoder: (B * 5, 1, 120, 501)
        features_flat = features.view(B * L, 1, C, T)
        
        # Encode individual locations: (B * 5, latent_dim)
        loc_embeddings_flat = self.location_encoder(features_flat)
        
        # Reshape back to (B, 5, latent_dim)
        loc_embeddings = loc_embeddings_flat.view(B, L, self.config.latent_dim)
        
        # Apply location masking to embeddings
        mask_expanded = location_mask.unsqueeze(-1) # (B, 5, 1)
        loc_embeddings = loc_embeddings * mask_expanded
        
        # Fuse representations with masked attention
        fused_patient, attn_weights = self.fusion(loc_embeddings, location_mask) # (B, latent_dim), (B, 5)
        
        # Predict logits
        murmur_logits = self.murmur_head(fused_patient)   # (B, 1)
        outcome_logits = self.outcome_head(fused_patient) # (B, 1)
        
        murmur_probs = torch.sigmoid(murmur_logits)
        outcome_probs = torch.sigmoid(outcome_logits)
        
        return {
            "murmur_logits": murmur_logits,
            "outcome_logits": outcome_logits,
            "murmur_probs": murmur_probs,
            "outcome_probs": outcome_probs,
            "attention_weights": attn_weights,
            "fused_patient_repr": fused_patient
        }
