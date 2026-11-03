import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.models.baseline import PCGBaselineClassifier
from src.models.patient_aggregator import PatientPredictionAggregator
from src.utils.metrics import (
    compute_binary_metrics,
    compute_physionet_murmur_weighted_cost,
    compute_physionet_outcome_cost
)
from src.utils.config import PROCESSED_DATA_DIR, MODELS_DIR

def prepare_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Separates feature columns and encodes tabular demographic features.
    """
    df = df.copy()
    
    # Encode categorical demographics
    if "age" in df.columns:
        age_dummies = pd.get_dummies(df["age"], prefix="demog_age", dtype=float)
        df = pd.concat([df, age_dummies], axis=1)
        
    if "sex" in df.columns:
        sex_dummies = pd.get_dummies(df["sex"], prefix="demog_sex", dtype=float)
        df = pd.concat([df, sex_dummies], axis=1)
        
    if "pregnancy_status" in df.columns:
        df["demog_pregnancy"] = df["pregnancy_status"].astype(float)
        
    if "height" in df.columns:
        df["demog_height"] = pd.to_numeric(df["height"], errors="coerce")
        
    if "weight" in df.columns:
        df["demog_weight"] = pd.to_numeric(df["weight"], errors="coerce")

    # Exclude non-feature identifiers and target metadata
    meta_cols = {
        "patient_id", "recording_id", "wav_path", "hea_path", "tsv_path",
        "patient_murmur", "patient_murmur_binary", "recording_murmur_label",
        "outcome", "outcome_binary", "fold", "age", "sex", "pregnancy_status",
        "height", "weight", "most_audible_location", "systolic_timing",
        "systolic_grading", "location"
    }
    
    feature_cols = [c for c in df.columns if c not in meta_cols]
    return df, feature_cols

def evaluate_murmur_detection(df: pd.DataFrame, feature_cols: List[str], model_type: str = "xgboost"):
    print(f"\n--- [TASK 1] 5-Fold Stratified CV: MURMUR DETECTION ({model_type.upper()}) ---")
    
    # Filter known classes: 0 = Absent, 1 = Present (exclude Unknown = -1)
    task_df = df[df["patient_murmur_binary"].isin([0, 1])].copy()
    folds = sorted(task_df["fold"].unique())
    
    oof_recs = []
    oof_patients = []
    
    for fold in folds:
        train_data = task_df[task_df["fold"] != fold]
        val_data = task_df[task_df["fold"] == fold].copy()
        
        X_train = train_data[feature_cols]
        # Train on patient_murmur_binary (or recording_murmur_label)
        y_train = train_data["patient_murmur_binary"].values.astype(int)
        
        X_val = val_data[feature_cols]
        y_val = val_data["patient_murmur_binary"].values.astype(int)
        
        model = PCGBaselineClassifier(model_type=model_type, random_state=42 + fold)
        model.fit(X_train, y_train)
        
        val_probs = model.predict_proba(X_val)[:, 1]
        val_data["pred_prob"] = val_probs
        oof_recs.append(val_data)
        
        # Aggregate to patient level
        aggregator = PatientPredictionAggregator(strategy="max", threshold=0.5)
        patient_val = aggregator.aggregate_patient_probabilities(val_data)
        oof_patients.append(patient_val)
        
    full_oof_recs = pd.concat(oof_recs, ignore_index=True)
    full_oof_patients = pd.concat(oof_patients, ignore_index=True)
    
    # 1. Recording-level metrics
    rec_metrics = compute_binary_metrics(
        full_oof_recs["patient_murmur_binary"].values,
        (full_oof_recs["pred_prob"] >= 0.5).astype(int),
        full_oof_recs["pred_prob"].values
    )
    
    # 2. Patient-level metrics
    pat_metrics = compute_binary_metrics(
        full_oof_patients["patient_murmur_binary"].values,
        full_oof_patients["patient_pred"].values,
        full_oof_patients["patient_prob"].values
    )
    
    pat_metrics["physionet_weighted_murmur_score"] = compute_physionet_murmur_weighted_cost(
        full_oof_patients["patient_murmur_binary"].values,
        full_oof_patients["patient_pred"].values
    )
    
    print(f"  [Recording Level] Accuracy: {rec_metrics['accuracy']:.4f} | Balanced Acc: {rec_metrics['balanced_accuracy']:.4f} | ROC-AUC: {rec_metrics['roc_auc']:.4f} | F1: {rec_metrics['f1']:.4f}")
    print(f"  [Patient Level]   Accuracy: {pat_metrics['accuracy']:.4f} | Balanced Acc: {pat_metrics['balanced_accuracy']:.4f} | ROC-AUC: {pat_metrics['roc_auc']:.4f} | F1: {pat_metrics['f1']:.4f}")
    print(f"  [PhysioNet Murmur Score]: {pat_metrics['physionet_weighted_murmur_score']:.4f}")
    
    return {"recording_metrics": rec_metrics, "patient_metrics": pat_metrics}

def evaluate_outcome_detection(df: pd.DataFrame, feature_cols: List[str], model_type: str = "xgboost"):
    print(f"\n--- [TASK 2] 5-Fold Stratified CV: CLINICAL OUTCOME DETECTION ({model_type.upper()}) ---")
    
    task_df = df[df["outcome_binary"].isin([0, 1])].copy()
    folds = sorted(task_df["fold"].unique())
    
    oof_recs = []
    oof_patients = []
    
    for fold in folds:
        train_data = task_df[task_df["fold"] != fold]
        val_data = task_df[task_df["fold"] == fold].copy()
        
        X_train = train_data[feature_cols]
        y_train = train_data["outcome_binary"].values.astype(int)
        
        X_val = val_data[feature_cols]
        
        model = PCGBaselineClassifier(model_type=model_type, random_state=42 + fold)
        model.fit(X_train, y_train)
        
        val_probs = model.predict_proba(X_val)[:, 1]
        val_data["pred_prob"] = val_probs
        oof_recs.append(val_data)
        
        aggregator = PatientPredictionAggregator(strategy="mean", threshold=0.5)
        patient_val = aggregator.aggregate_patient_probabilities(val_data)
        oof_patients.append(patient_val)
        
    full_oof_patients = pd.concat(oof_patients, ignore_index=True)
    
    pat_metrics = compute_binary_metrics(
        full_oof_patients["outcome_binary"].values,
        full_oof_patients["patient_pred"].values,
        full_oof_patients["patient_prob"].values
    )
    
    pat_metrics["physionet_outcome_mean_cost"] = compute_physionet_outcome_cost(
        full_oof_patients["outcome_binary"].values,
        full_oof_patients["patient_pred"].values
    )
    
    print(f"  [Patient Level]   Accuracy: {pat_metrics['accuracy']:.4f} | Balanced Acc: {pat_metrics['balanced_accuracy']:.4f} | ROC-AUC: {pat_metrics['roc_auc']:.4f} | F1: {pat_metrics['f1']:.4f}")
    print(f"  [PhysioNet Outcome Cost]: ${pat_metrics['physionet_outcome_mean_cost']:.2f}")
    
    return pat_metrics

def main():
    feat_csv = PROCESSED_DATA_DIR / "circor_features.csv"
    if not feat_csv.exists():
        print(f"Error: {feat_csv} not found. Running extract_features first...")
        from scripts.extract_features import run_feature_extraction
        run_feature_extraction(max_workers=4)
        
    print(f"Loading feature table from {feat_csv}...")
    raw_df = pd.read_csv(feat_csv)
    df, feature_cols = prepare_feature_matrix(raw_df)
    print(f"Dataset shape: {df.shape}, Number of acoustic/clinical features: {len(feature_cols)}")
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    
    # Evaluate baselines
    for model_type in ["xgboost", "random_forest", "logistic_regression"]:
        m_res = evaluate_murmur_detection(df, feature_cols, model_type=model_type)
        o_res = evaluate_outcome_detection(df, feature_cols, model_type=model_type)
        results[model_type] = {
            "murmur_task": m_res,
            "outcome_task": o_res
        }
        
    # Fit and save final full-dataset models
    print("\nFitting and saving final production baseline models on full dataset...")
    # Murmur model
    known_murmur_df = df[df["patient_murmur_binary"].isin([0, 1])]
    final_murmur_clf = PCGBaselineClassifier(model_type="xgboost")
    final_murmur_clf.fit(known_murmur_df[feature_cols], known_murmur_df["patient_murmur_binary"].values.astype(int))
    final_murmur_clf.save(MODELS_DIR / "murmur_baseline_xgboost.joblib")
    
    # Outcome model
    final_outcome_clf = PCGBaselineClassifier(model_type="xgboost")
    final_outcome_clf.fit(df[feature_cols], df["outcome_binary"].values.astype(int))
    final_outcome_clf.save(MODELS_DIR / "outcome_baseline_xgboost.joblib")
    
    with open(MODELS_DIR / "baseline_cv_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nAll models and benchmark results saved to {MODELS_DIR}/")

if __name__ == "__main__":
    main()
