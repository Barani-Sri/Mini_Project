"""
Unit Tests for Module 2 Multi-Location Representation and Fusion.
"""
import pytest
import numpy as np

from src.features.location import (
    encode_location,
    decode_location,
    MultiLocationPatientTensor
)

def test_location_encoding():
    assert encode_location("AV") == 0
    assert encode_location("MV") == 1
    assert encode_location("PV") == 2
    assert encode_location("TV") == 3
    assert encode_location("Phc") == 4

    assert decode_location(0) == "AV"
    assert decode_location(1) == "MV"
    assert decode_location(2) == "PV"
    assert decode_location(3) == "TV"
    assert decode_location(4) == "Phc"

def test_multi_location_tensor_construction():
    builder = MultiLocationPatientTensor(num_channels=120, num_time_frames=501)
    
    # Patient with only AV and TV
    av_feat = np.ones((120, 501), dtype=np.float32) * 2.0
    tv_feat = np.ones((120, 501), dtype=np.float32) * 3.0
    
    loc_dict = {"AV": av_feat, "TV": tv_feat}
    tensor, mask = builder.build_patient_tensor(loc_dict)
    
    assert tensor.shape == (5, 120, 501)
    assert mask.shape == (5,)
    
    # AV(0) and TV(3) are present, MV(1), PV(2), Phc(4) are missing (zeros)
    assert mask[0] == 1.0
    assert mask[1] == 0.0
    assert mask[2] == 0.0
    assert mask[3] == 1.0
    assert mask[4] == 0.0
    
    assert np.all(tensor[0] == 2.0)
    assert np.all(tensor[1] == 0.0)  # Missing zero padded
    assert np.all(tensor[2] == 0.0)  # Missing zero padded
    assert np.all(tensor[3] == 3.0)
    assert np.all(tensor[4] == 0.0)  # Missing zero padded

def test_fusion_formatting():
    builder = MultiLocationPatientTensor(num_channels=120, num_time_frames=501)
    loc_dict = {
        "AV": np.ones((120, 501), dtype=np.float32),
        "MV": np.ones((120, 501), dtype=np.float32)
    }
    tensor, mask = builder.build_patient_tensor(loc_dict)
    
    # Early fusion: (5 * 120, 501) = (600, 501)
    early = builder.format_early_fusion(tensor)
    assert early.shape == (600, 501)
    
    # Intermediate fusion: (5, 120, 501)
    inter = builder.format_intermediate_fusion(tensor)
    assert inter.shape == (5, 120, 501)
    
    # Attention fusion: features (5, 120, 501), mask (5,)
    att_feat, att_mask = builder.format_attention_fusion(tensor, mask)
    assert att_feat.shape == (5, 120, 501)
    assert att_mask.shape == (5,)
