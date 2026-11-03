"""
Integration unit tests for end-to-end pipeline, patient stratification, and baseline models.
"""
import pytest
import numpy as np
import pandas as pd

from src.data.dataset import CirCorDataset
from src.data.splitting import (
    get_patient_kfold_splits,
    assign_patient_folds,
    patient_train_val_test_split
)
from src.models.baseline import PCGBaselineClassifier
from src.models.patient_aggregator import PatientPredictionAggregator
from src.utils.metrics import compute_binary_metrics, compute_physionet_murmur_weighted_cost
from src.utils.config import RAW_DATA_DIR

@pytest.fixture
def dataset():
    return CirCorDataset(data_dir=RAW_DATA_DIR)

def test_patient_stratified_kfold_no_leakage(dataset):
    patients_df = dataset.get_patients_df()
    recs_df = dataset.get_recordings_df()
    
    # K-Fold splits
    splits = get_patient_kfold_splits(patients_df, n_splits=5, random_state=42)
    assert len(splits) == 5
    
    for train_pids, val_pids in splits:
        train_set = set(train_pids)
        val_set = set(val_pids)
        # Verify 0 overlap between train and val
        assert len(train_set.intersection(val_set)) == 0
        assert len(train_set) + len(val_set) == len(patients_df)

def test_patient_folds_assignment_on_recordings(dataset):
    recs_df = dataset.get_recordings_df()
    folded_recs = assign_patient_folds(recs_df, n_splits=5, random_state=42, patient_id_col="patient_id")
    
    assert "fold" in folded_recs.columns
    assert set(folded_recs["fold"].unique()) == {0, 1, 2, 3, 4}
    
    # Check that each patient has exactly 1 unique fold
    patient_folds = folded_recs.groupby("patient_id")["fold"].nunique()
    assert (patient_folds == 1).all(), "Patient data leaked across multiple folds!"

def test_patient_train_val_test_split_no_leakage(dataset):
    patients_df = dataset.get_patients_df()
    train_df, val_df, test_df = patient_train_val_test_split(patients_df, test_size=0.2, val_size=0.1, random_state=42)
    
    train_pids = set(train_df["Patient ID"])
    val_pids = set(val_df["Patient ID"])
    test_pids = set(test_df["Patient ID"])
    
    assert len(train_pids.intersection(val_pids)) == 0
    assert len(train_pids.intersection(test_pids)) == 0
    assert len(val_pids.intersection(test_pids)) == 0
    assert len(train_pids) + len(val_pids) + len(test_pids) == len(patients_df)

def test_baseline_classifier_fit_predict():
    # Synthetic feature matrix
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 20), columns=[f"feat_{i}" for i in range(20)])
    y = np.random.randint(0, 2, size=100)
    
    clf = PCGBaselineClassifier(model_type="random_forest", n_estimators=10)
    clf.fit(X, y)
    
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    
    assert len(preds) == 100
    assert probs.shape == (100, 2)
    assert set(np.unique(preds)).issubset({0, 1})
    
    metrics = compute_binary_metrics(y, preds, probs[:, 1])
    assert "accuracy" in metrics
    assert "roc_auc" in metrics

def test_patient_aggregator():
    # Mock recording-level predictions for 2 patients
    # Patient 1 has 3 recordings: [0.1, 0.2, 0.8] -> max = 0.8 (Positive)
    # Patient 2 has 2 recordings: [0.1, 0.3] -> max = 0.3 (Negative)
    mock_recs = pd.DataFrame([
        {"patient_id": "P1", "location": "AV", "pred_prob": 0.1, "patient_murmur_binary": 1},
        {"patient_id": "P1", "location": "MV", "pred_prob": 0.2, "patient_murmur_binary": 1},
        {"patient_id": "P1", "location": "TV", "pred_prob": 0.8, "patient_murmur_binary": 1},
        {"patient_id": "P2", "location": "AV", "pred_prob": 0.1, "patient_murmur_binary": 0},
        {"patient_id": "P2", "location": "PV", "pred_prob": 0.3, "patient_murmur_binary": 0},
    ])
    
    aggregator = PatientPredictionAggregator(strategy="max", threshold=0.5)
    patient_df = aggregator.aggregate_patient_probabilities(mock_recs)
    
    assert len(patient_df) == 2
    p1 = patient_df[patient_df["patient_id"] == "P1"].iloc[0]
    p2 = patient_df[patient_df["patient_id"] == "P2"].iloc[0]
    
    assert np.isclose(p1["patient_prob"], 0.8)
    assert p1["patient_pred"] == 1
    assert np.isclose(p2["patient_prob"], 0.3)
    assert p2["patient_pred"] == 0
