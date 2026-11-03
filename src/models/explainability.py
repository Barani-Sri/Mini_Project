"""
Module 3 Auscultation Location Explainability & Ablation Analysis.
Provides:
1. Multi-Valve Attention Distribution Analysis across cardiac auscultation sites.
2. Systematic Location Ablation Analysis (evaluating diagnostic degradation when masking out individual sites).
"""
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score

from src.models.config import Module3Config, AUSCULTATION_LOCATIONS
from src.models.multi_location_net import MultiLocationHeartSoundNet

class Module3Explainability:
    """
    Analyzes and explains multi-location PCG feature utilization.
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
        self.location_names = ["AV", "MV", "PV", "TV", "Phc"]

    def analyze_attention_weights(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyzes attention weights across all patients, stratified by Murmur and Outcome.
        """
        attn_cols = [f"attn_{loc}" for loc in self.location_names]
        
        overall_mean = {loc: float(predictions_df[f"attn_{loc}"].mean()) for loc in self.location_names}
        overall_std = {loc: float(predictions_df[f"attn_{loc}"].std()) for loc in self.location_names}

        # Stratified by Murmur Present (1) vs Absent (0)
        present_df = predictions_df[predictions_df["true_murmur"] == 1]
        absent_df = predictions_df[predictions_df["true_murmur"] == 0]

        murmur_present_mean = {loc: float(present_df[f"attn_{loc}"].mean()) for loc in self.location_names}
        murmur_absent_mean = {loc: float(absent_df[f"attn_{loc}"].mean()) for loc in self.location_names}

        # Stratified by Outcome Abnormal (1) vs Normal (0)
        abnormal_df = predictions_df[predictions_df["true_clinical_outcome"] == 1]
        normal_df = predictions_df[predictions_df["true_clinical_outcome"] == 0]

        outcome_abnormal_mean = {loc: float(abnormal_df[f"attn_{loc}"].mean()) for loc in self.location_names}
        outcome_normal_mean = {loc: float(normal_df[f"attn_{loc}"].mean()) for loc in self.location_names}

        # Find most attended location per patient
        pred_attn_matrix = predictions_df[attn_cols].values
        top_loc_indices = np.argmax(pred_attn_matrix, axis=1)
        top_loc_counts = {self.location_names[i]: int(np.sum(top_loc_indices == i)) for i in range(len(self.location_names))}

        return {
            "overall_mean_attention": overall_mean,
            "overall_std_attention": overall_std,
            "murmur_present_mean_attention": murmur_present_mean,
            "murmur_absent_mean_attention": murmur_absent_mean,
            "outcome_abnormal_mean_attention": outcome_abnormal_mean,
            "outcome_normal_mean_attention": outcome_normal_mean,
            "top_attended_location_patient_counts": top_loc_counts
        }

    def perform_location_ablation(
        self,
        test_loader: DataLoader
    ) -> Dict[str, Dict[str, float]]:
        """
        Systematically ablates (zeros out) each auscultation location one by one,
        and measures the resulting degradation in Murmur and Outcome ROC-AUC.
        """
        # 1. Baseline Full Model Performance
        base_m_trues, base_m_probs = [], []
        base_o_trues, base_o_probs = [], []
        
        all_features = []
        all_masks = []
        all_y_m = []
        all_y_o = []

        with torch.no_grad():
            for batch in test_loader:
                feat = batch["features"].to(self.device)
                mask = batch["location_mask"].to(self.device)
                y_m = batch["murmur_label"]
                y_o = batch["outcome_label"]

                out = self.model(feat, mask)
                
                base_m_trues.extend(y_m.numpy())
                base_m_probs.extend(out["murmur_probs"].cpu().numpy().ravel())
                base_o_trues.extend(y_o.numpy())
                base_o_probs.extend(out["outcome_probs"].cpu().numpy().ravel())

                all_features.append(feat)
                all_masks.append(mask)
                all_y_m.append(y_m)
                all_y_o.append(y_o)

        base_m_trues = np.array(base_m_trues)
        base_m_probs = np.array(base_m_probs)
        base_o_trues = np.array(base_o_trues)
        base_o_probs = np.array(base_o_probs)

        base_m_auc = float(roc_auc_score(base_m_trues, base_m_probs))
        base_o_auc = float(roc_auc_score(base_o_trues, base_o_probs))
        base_m_f1 = float(f1_score(base_m_trues, (base_m_probs >= 0.5).astype(int), zero_division=0))
        base_o_f1 = float(f1_score(base_o_trues, (base_o_probs >= 0.5).astype(int), zero_division=0))

        ablation_results = {
            "baseline": {
                "murmur_auc": base_m_auc,
                "murmur_f1": base_m_f1,
                "outcome_auc": base_o_auc,
                "outcome_f1": base_o_f1,
                "delta_murmur_auc": 0.0,
                "delta_outcome_auc": 0.0
            }
        }

        # 2. Ablate Each Location (AV, MV, PV, TV, Phc)
        for loc_idx, loc_name in enumerate(self.location_names):
            abl_m_probs, abl_o_probs = [], []

            with torch.no_grad():
                for feat, mask in zip(all_features, all_masks):
                    # Zero out the specific location mask
                    abl_mask = mask.clone()
                    abl_mask[:, loc_idx] = 0.0
                    
                    out = self.model(feat, abl_mask)
                    abl_m_probs.extend(out["murmur_probs"].cpu().numpy().ravel())
                    abl_o_probs.extend(out["outcome_probs"].cpu().numpy().ravel())

            abl_m_probs = np.array(abl_m_probs)
            abl_o_probs = np.array(abl_o_probs)

            abl_m_auc = float(roc_auc_score(base_m_trues, abl_m_probs))
            abl_o_auc = float(roc_auc_score(base_o_trues, abl_o_probs))
            abl_m_f1 = float(f1_score(base_m_trues, (abl_m_probs >= 0.5).astype(int), zero_division=0))
            abl_o_f1 = float(f1_score(base_o_trues, (abl_o_probs >= 0.5).astype(int), zero_division=0))

            ablation_results[f"without_{loc_name}"] = {
                "murmur_auc": abl_m_auc,
                "murmur_f1": abl_m_f1,
                "outcome_auc": abl_o_auc,
                "outcome_f1": abl_o_f1,
                "delta_murmur_auc": float(base_m_auc - abl_m_auc),
                "delta_outcome_auc": float(base_o_auc - abl_o_auc)
            }

        return ablation_results
