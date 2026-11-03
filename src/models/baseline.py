"""
Baseline classification models for tabular PCG features.
Implements Random Forest, XGBoost, and Logistic Regression with patient-level evaluation.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from src.models.patient_aggregator import PatientPredictionAggregator
from src.utils.metrics import compute_binary_metrics, compute_physionet_murmur_weighted_cost, compute_physionet_outcome_cost

class PCGBaselineClassifier:
    """
    Supervised classifier wrapper trained on tabular acoustic/segmentation PCG features.
    """
    def __init__(
        self,
        model_type: str = "xgboost",
        n_estimators: int = 150,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_state: int = 42
    ):
        self.model_type = model_type.lower()
        self.random_state = random_state
        
        if self.model_type == "rf" or self.model_type == "random_forest":
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                class_weight="balanced"
            )
        elif self.model_type == "lr" or self.model_type == "logistic_regression":
            clf = LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state
            )
        else: # Default XGBoost
            clf = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                eval_metric="logloss",
                random_state=random_state
            )
            
        self.pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", clf)
        ])
        self.feature_names_: List[str] = []

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_mat = X.values
        else:
            X_mat = X
            self.feature_names_ = [f"feature_{i}" for i in range(X_mat.shape[1])]
            
        self.pipeline.fit(X_mat, y)
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        return self.pipeline.predict_proba(X_mat)

    def predict(self, X: Union[np.ndarray, pd.DataFrame], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def save(self, filepath: Union[str, Path]):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "feature_names": self.feature_names_, "model_type": self.model_type}, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "PCGBaselineClassifier":
        data = joblib.load(filepath)
        instance = cls(model_type=data["model_type"])
        instance.pipeline = data["pipeline"]
        instance.feature_names_ = data.get("feature_names", [])
        return instance
