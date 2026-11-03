"""
Multi-instance patient aggregator for combining multi-valve PCG recordings.
Aggregates predictions from AV, PV, TV, and MV auscultation locations to generate a patient-level diagnosis.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

class PatientPredictionAggregator:
    """
    Aggregates recording-level predictions to patient-level diagnostic decisions.
    Supported strategies:
    - 'max': Max probability across all auscultation sites (standard for murmur detection)
    - 'mean': Average probability across all auscultation sites
    - 'noisy_or': Noisy-OR probabilistic pooling: 1 - prod(1 - p_i)
    - 'ensemble': Weighted combination of max and mean
    """
    def __init__(self, strategy: str = "max", threshold: float = 0.5):
        self.strategy = strategy
        self.threshold = threshold

    def aggregate_patient_probabilities(
        self,
        recordings_df: pd.DataFrame,
        prob_col: str = "pred_prob",
        patient_id_col: str = "patient_id"
    ) -> pd.DataFrame:
        """
        Aggregates recording probabilities to produce one row per patient.
        """
        df = recordings_df.copy()
        patient_rows = []
        
        for pid, group in df.groupby(patient_id_col):
            probs = group[prob_col].values.astype(float)
            
            if self.strategy == "max":
                p_patient = float(np.max(probs))
            elif self.strategy == "mean":
                p_patient = float(np.mean(probs))
            elif self.strategy == "noisy_or":
                p_patient = float(1.0 - np.prod(1.0 - probs))
            elif self.strategy == "ensemble":
                p_patient = float(0.7 * np.max(probs) + 0.3 * np.mean(probs))
            else:
                p_patient = float(np.max(probs))
                
            pred_class = 1 if p_patient >= self.threshold else 0
            
            row = {
                patient_id_col: pid,
                "patient_prob": p_patient,
                "patient_pred": pred_class,
                "num_recordings": len(group)
            }
            
            # Carry over ground truth patient labels if present
            for target_col in ["patient_murmur_binary", "patient_murmur", "outcome_binary", "outcome"]:
                if target_col in group.columns:
                    row[target_col] = group[target_col].iloc[0]
                    
            patient_rows.append(row)
            
        return pd.DataFrame(patient_rows)
