"""
Module 3 Comprehensive Evaluator for Unseen Test Patients.
Calculates diagnostic classification metrics, PhysioNet challenge scores, and generates per-patient predictions CSV.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from src.models.config import (
    Module3Config,
    MODULE3_PREDICTIONS_PATH,
    AUSCULTATION_LOCATIONS
)
from src.models.multi_location_net import MultiLocationHeartSoundNet
from src.utils.metrics import compute_physionet_murmur_weighted_cost, compute_physionet_outcome_cost

def compute_metrics_for_target(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes all standard diagnostic metrics for binary prediction.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = 0.0

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "sensitivity": float(sensitivity),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": cm.tolist(),
        "roc_curve_data": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve_data": {"precision": prec_curve.tolist(), "recall": rec_curve.tolist()}
    }


class Module3Evaluator:
    """
    Evaluator for patient-level multi-task model on unseen test data.
    """
    def __init__(
        self,
        model: MultiLocationHeartSoundNet,
        config: Optional[Module3Config] = None
    ):
        self.config = config if config is not None else Module3Config()
        self.device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def evaluate_test_set(
        self,
        test_loader: DataLoader,
        save_predictions_path: Optional[Union[str, Path]] = MODULE3_PREDICTIONS_PATH
    ) -> Dict[str, Any]:
        """
        Executes full evaluation on unseen test patients.
        """
        patient_ids = []
        murmur_trues, murmur_probs = [], []
        outcome_trues, outcome_probs = [], []
        attention_weights_list = []
        masks_list = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                location_mask = batch["location_mask"].to(self.device)
                pids = batch["patient_id"]
                y_m = batch["murmur_label"]
                y_o = batch["outcome_label"]

                outputs = self.model(features, location_mask)

                patient_ids.extend(pids)
                murmur_trues.extend(y_m.numpy())
                murmur_probs.extend(outputs["murmur_probs"].cpu().numpy().ravel())
                outcome_trues.extend(y_o.numpy())
                outcome_probs.extend(outputs["outcome_probs"].cpu().numpy().ravel())
                attention_weights_list.extend(outputs["attention_weights"].cpu().numpy())
                masks_list.extend(location_mask.cpu().numpy())

        murmur_trues = np.array(murmur_trues, dtype=int)
        murmur_probs = np.array(murmur_probs, dtype=float)
        outcome_trues = np.array(outcome_trues, dtype=int)
        outcome_probs = np.array(outcome_probs, dtype=float)
        attention_weights = np.array(attention_weights_list, dtype=float)
        masks = np.array(masks_list, dtype=float)

        # 1. Compute Standard Diagnostic Metrics
        murmur_metrics = compute_metrics_for_target(
            murmur_trues, murmur_probs, threshold=self.config.murmur_threshold
        )
        outcome_metrics = compute_metrics_for_target(
            outcome_trues, outcome_probs, threshold=self.config.outcome_threshold
        )

        # 2. Compute PhysioNet 2022 Challenge Domain Cost Metrics
        m_preds = (murmur_probs >= self.config.murmur_threshold).astype(int)
        o_preds = (outcome_probs >= self.config.outcome_threshold).astype(int)

        murmur_physionet_score = compute_physionet_murmur_weighted_cost(murmur_trues, m_preds)
        outcome_physionet_cost = compute_physionet_outcome_cost(outcome_trues, o_preds)

        murmur_metrics["physionet_weighted_score"] = murmur_physionet_score
        outcome_metrics["physionet_mean_cost"] = outcome_physionet_cost

        # 3. Create Per-Patient Predictions DataFrame
        pred_rows = []
        loc_names = ["AV", "MV", "PV", "TV", "Phc"]

        for i in range(len(patient_ids)):
            pid = patient_ids[i]
            avail_locs = [loc_names[j] for j in range(5) if masks[i, j] > 0.5]
            
            row = {
                "patient_id": pid,
                "true_murmur": int(murmur_trues[i]),
                "predicted_murmur": int(m_preds[i]),
                "murmur_probability": float(murmur_probs[i]),
                "true_clinical_outcome": int(outcome_trues[i]),
                "predicted_clinical_outcome": int(o_preds[i]),
                "clinical_outcome_probability": float(outcome_probs[i]),
                "available_locations": "+".join(avail_locs),
                "num_locations": len(avail_locs),
                "attn_AV": float(attention_weights[i, 0]),
                "attn_MV": float(attention_weights[i, 1]),
                "attn_PV": float(attention_weights[i, 2]),
                "attn_TV": float(attention_weights[i, 3]),
                "attn_Phc": float(attention_weights[i, 4])
            }
            pred_rows.append(row)

        pred_df = pd.DataFrame(pred_rows)

        # Save to CSV
        if save_predictions_path is not None:
            save_p = Path(save_predictions_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            pred_df.to_csv(save_p, index=False)

        return {
            "num_test_patients": len(patient_ids),
            "murmur_detection": murmur_metrics,
            "clinical_outcome": outcome_metrics,
            "predictions_df": pred_df,
            "raw_arrays": {
                "patient_ids": patient_ids,
                "murmur_trues": murmur_trues,
                "murmur_probs": murmur_probs,
                "outcome_trues": outcome_trues,
                "outcome_probs": outcome_probs,
                "attention_weights": attention_weights,
                "masks": masks
            }
        }
