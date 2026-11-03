"""
Unit tests for HFrEF multimodal feature fusion, analysis engine, and specialist recommendation.
"""
import pytest
import numpy as np

from src.hfref.config import TOTAL_FUSED_DIM, ECG_FEATURE_DIM, PCG_FEATURE_PER_LOC_DIM
from src.hfref.feature_extraction import (
    extract_multimodal_features,
    MultiModalSubjectFeatures
)
from src.hfref.fusion import (
    compute_location_energy_weights,
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

def test_multimodal_feature_extraction_and_masking():
    """Verify multimodal extraction shapes, dimensions, and location masking."""
    fs_ecg = 500
    fs_pcg = 2000
    
    # Create synthetic subject signals with 1 ECG and 2 PCG locations (RUSB, APEX)
    mock_signals = {
        "subject_id": "TEST_SUBJ_42",
        "lvef": 32.0,
        "is_hfref": True,
        "fold": 2,
        "ecg_signals": {
            "APEX": {"signal": np.random.randn(int(3.0 * fs_ecg)), "sr": fs_ecg}
        },
        "pcg_signals": {
            "RUSB": {"signal": np.random.randn(int(3.0 * fs_pcg)), "sr": fs_pcg},
            "APEX": {"signal": np.random.randn(int(3.0 * fs_pcg)), "sr": fs_pcg}
        }
    }

    feats = extract_multimodal_features(mock_signals)
    assert feats.subject_id == "TEST_SUBJ_42"
    assert feats.lvef == 32.0
    assert feats.is_hfref is True
    assert feats.fold == 2
    assert len(feats.fused_vector) == TOTAL_FUSED_DIM
    
    # Mask order: [ECG, RUSB(AV), LUSB(PV), LLSB(TV), APEX(MV)]
    mask = feats.location_mask
    assert mask[0] == 1.0 # ECG active
    assert mask[1] == 1.0 # RUSB active
    assert mask[2] == 0.0 # LUSB missing
    assert mask[3] == 0.0 # LLSB missing
    assert mask[4] == 1.0 # APEX active

def test_fusion_and_location_weights():
    """Verify energy-based relative channel weighting in fusion module."""
    mock_feats = MultiModalSubjectFeatures(
        subject_id="S1",
        lvef=55.0,
        is_hfref=False,
        fold=1,
        ecg_features={"feature_vector": np.ones(ECG_FEATURE_DIM), "energy": 0.5},
        pcg_location_features={
            "RUSB": {"feature_vector": np.ones(PCG_FEATURE_PER_LOC_DIM), "rms_energy": 0.4, "shannon_energy_mean": 0.05},
            "LUSB": {"feature_vector": np.zeros(PCG_FEATURE_PER_LOC_DIM), "rms_energy": 0.0, "shannon_energy_mean": 0.0},
            "LLSB": {"feature_vector": np.zeros(PCG_FEATURE_PER_LOC_DIM), "rms_energy": 0.0, "shannon_energy_mean": 0.0},
            "APEX": {"feature_vector": np.ones(PCG_FEATURE_PER_LOC_DIM), "rms_energy": 0.6, "shannon_energy_mean": 0.08}
        },
        fused_vector=np.ones(TOTAL_FUSED_DIM, dtype=np.float32),
        location_mask=np.array([1.0, 1.0, 0.0, 0.0, 1.0]),
        available_pcg_locations=["RUSB", "APEX"]
    )

    fused_out = fuse_ecg_pcg_features(mock_feats)
    ch_weights = fused_out["channel_weights"]
    
    assert ch_weights["ECG"]["is_active"] is True
    assert ch_weights["RUSB"]["is_active"] is True
    assert ch_weights["LUSB"]["is_active"] is False
    assert ch_weights["LLSB"]["is_active"] is False
    assert ch_weights["APEX"]["is_active"] is True

    # Active weights should sum to ~100%
    total_pct = sum(w["percent"] for w in ch_weights.values())
    assert 99.0 <= total_pct <= 101.0

def test_analysis_engine_no_fabrication_when_dataset_missing():
    """Verify analysis engine returns False for train_and_evaluate_folds without fabricating metrics."""
    engine = HFrEFAnalysisEngine(model_path=None)
    # When dataset loader has no files:
    res = engine.train_and_evaluate_folds()
    assert res["success"] is False
    assert res["metrics"] is None

def test_analysis_engine_inference():
    """Verify inference output structure and clinical bounds."""
    engine = get_hfref_analyzer()
    mock_feats = MultiModalSubjectFeatures(
        subject_id="SUBJ_TEST",
        lvef=35.0,
        is_hfref=True,
        fold=3,
        ecg_features={
            "feature_vector": np.zeros(ECG_FEATURE_DIM),
            "heart_rate_bpm": 88.0,
            "sdnn_ms": 22.0,
            "rmssd_ms": 18.0
        },
        pcg_location_features={
            "RUSB": {"feature_vector": np.zeros(PCG_FEATURE_PER_LOC_DIM), "rms_energy": 0.12},
            "LUSB": {"feature_vector": np.zeros(PCG_FEATURE_PER_LOC_DIM), "rms_energy": 0.08},
            "LLSB": {"feature_vector": np.zeros(PCG_FEATURE_PER_LOC_DIM), "rms_energy": 0.09},
            "APEX": {"feature_vector": np.zeros(PCG_FEATURE_PER_LOC_DIM), "rms_energy": 0.18, "shannon_energy_mean": 0.09}
        },
        fused_vector=np.zeros(TOTAL_FUSED_DIM, dtype=np.float32),
        location_mask=np.ones(5),
        available_pcg_locations=["RUSB", "LUSB", "LLSB", "APEX"]
    )

    result = engine.analyze_features(mock_feats)
    assert result["subject_id"] == "SUBJ_TEST"
    assert result["prediction"] in ["HFrEF", "Non-HFrEF"]
    assert 0.0 <= result["hfref_probability"] <= 1.0
    assert "locations_analysis" in result
    assert "ecg_analysis" in result
    assert "disclaimer" in result

def test_specialist_recommender():
    """Verify recommendation logic across combinations of LubDub and HFrEF findings."""
    recommender = get_specialist_recommender()

    # Case 1: Both HFrEF and Murmur detected
    rec1 = recommender.generate_recommendation(
        hfref_result={"prediction": "HFrEF", "prediction_class": 1, "hfref_probability": 0.85},
        lubdub_result={"murmur_detection": {"prediction": "Present", "prediction_class": 1}}
    )
    assert "Cardiologist" in rec1["suggested_specialist"]
    assert rec1["priority_class"] == "urgent"
    assert len(rec1["recommended_investigations"]) > 0

    # Case 2: Only HFrEF detected
    rec2 = recommender.generate_recommendation(
        hfref_result={"prediction": "HFrEF", "prediction_class": 1, "hfref_probability": 0.72},
        lubdub_result={"murmur_detection": {"prediction": "Absent", "prediction_class": 0}}
    )
    assert "Cardiologist" in rec2["suggested_specialist"]
    assert rec2["priority_class"] == "warning"

    # Case 3: Only Murmur detected
    rec3 = recommender.generate_recommendation(
        hfref_result={"prediction": "Non-HFrEF", "prediction_class": 0, "hfref_probability": 0.20},
        lubdub_result={"murmur_detection": {"prediction": "Present", "prediction_class": 1}}
    )
    assert "Cardiologist" in rec3["suggested_specialist"]
    assert rec3["priority_class"] == "warning"

    # Case 4: Both normal
    rec4 = recommender.generate_recommendation(
        hfref_result={"prediction": "Non-HFrEF", "prediction_class": 0, "hfref_probability": 0.15},
        lubdub_result={
            "murmur_detection": {"prediction": "Absent", "prediction_class": 0},
            "clinical_outcome": {"prediction": "Normal", "prediction_class": 0}
        }
    )
    assert "Primary Care" in rec4["suggested_specialist"] or "General" in rec4["suggested_specialist"]
    assert rec4["priority_class"] == "normal"
