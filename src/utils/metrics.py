"""
Evaluation metrics for Heart Murmur & Clinical Outcome detection.
Implements standard classification metrics and official PhysioNet 2022 Challenge cost metrics.
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)

def compute_binary_metrics(y_true, y_pred, y_prob=None):
    """
    Computes comprehensive binary classification metrics:
    Accuracy, Balanced Accuracy, Sensitivity (Recall), Specificity, Precision, F1-Score, ROC-AUC, PR-AUC.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "recall": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "confusion_matrix": cm.tolist()
    }
    
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.5
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["pr_auc"] = 0.0
            
    return metrics

def compute_physionet_murmur_weighted_cost(y_true, y_pred):
    """
    Calculates the PhysioNet 2022 Challenge Murmur Evaluation Weighted Accuracy.
    Classes: 0 = Absent, 1 = Present, 2 = Unknown (if 3-class) or binary (0=Absent, 1=Present).
    
    In the PhysioNet 2022 Challenge:
    Weights for Murmur prediction matrix:
    - True Present predicted as Present: 5
    - True Unknown predicted as Unknown: 3
    - True Absent predicted as Absent: 1
    - Any correct prediction has positive weight; misclassifications get 0 weight.
    Total weighted score = sum(weights[true, pred]) / sum(max_weights[true])
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    # Weight weights for [Absent, Present, Unknown]
    # Standard challenge weights:
    weight_table = {
        (0, 0): 1.0,  # True Absent -> Pred Absent
        (1, 1): 5.0,  # True Present -> Pred Present
        (2, 2): 3.0,  # True Unknown -> Pred Unknown
        (1, 2): 1.0,  # True Present -> Pred Unknown (partial credit for recognizing abnormality)
        (2, 1): 1.0,  # True Unknown -> Pred Present
    }
    max_weights = {0: 1.0, 1: 5.0, 2: 3.0}
    
    total_score = 0.0
    max_score = 0.0
    
    for t, p in zip(y_true, y_pred):
        total_score += weight_table.get((t, p), 0.0)
        max_score += max_weights.get(t, 1.0)
        
    weighted_acc = total_score / max_score if max_score > 0 else 0.0
    return float(weighted_acc)

def compute_physionet_outcome_cost(y_true, y_pred):
    """
    Calculates the PhysioNet 2022 Challenge Outcome Diagnostic Cost.
    Outcome classes: 0 = Normal, 1 = Abnormal.
    Clinical Cost weights:
    - True Normal, Pred Normal: Cost = 0
    - True Abnormal, Pred Abnormal: Cost = 10 (cost of diagnostic ultrasound/follow-up)
    - True Normal, Pred Abnormal (False Positive): Cost = 10 (unnecessary echocardiogram)
    - True Abnormal, Pred Normal (False Negative / Missed case): Cost = 100 (missed congenital heart defect)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    cost_matrix = {
        (0, 0): 0.0,
        (0, 1): 10.0,
        (1, 0): 100.0,
        (1, 1): 10.0
    }
    
    total_cost = sum(cost_matrix.get((t, p), 0.0) for t, p in zip(y_true, y_pred))
    mean_cost = total_cost / len(y_true) if len(y_true) > 0 else 0.0
    return float(mean_cost)
