"""
Module 3 Visualization Suite for Multi-Location Heart Murmur & Clinical Outcome Detection.
Generates publication-quality figures saved under data/processed/module3_results/ using pure Matplotlib.
"""
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve

from src.models.config import MODULE3_RESULTS_DIR

def set_plot_style():
    """Sets a clean, publication-ready styling for matplotlib figures."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#CCCCCC"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.color"] = "#F0F0F0"
    plt.rcParams["grid.linestyle"] = "--"


class Module3Visualizer:
    """
    Generates standardized diagnostic and explainability visualizations for Module 3.
    """
    def __init__(self, output_dir: Path = MODULE3_RESULTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_plot_style()

    def plot_training_history(self, history: Dict[str, List[float]]) -> Tuple[Path, Path]:
        """Plots training vs validation loss and metrics curves."""
        epochs = history["epoch"]
        
        # 1. Loss Curve
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.plot(epochs, history["train_loss"], label="Train Loss (Combined)", color="#2563eb", lw=2)
        ax.plot(epochs, history["val_loss"], label="Validation Loss", color="#dc2626", lw=2, linestyle="--")
        ax.plot(epochs, history["val_murmur_loss"], label="Val Murmur Loss", color="#059669", lw=1.2, alpha=0.7)
        ax.plot(epochs, history["val_outcome_loss"], label="Val Outcome Loss", color="#d97706", lw=1.2, alpha=0.7)
        ax.set_title("Module 3: Multi-Task Training and Validation Loss", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Weighted BCE Loss", fontsize=11)
        ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
        plt.tight_layout()
        loss_path = self.output_dir / "loss_curve.png"
        fig.savefig(loss_path, dpi=300)
        plt.close(fig)

        # 2. Metrics Curve
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.plot(epochs, history["val_murmur_auc"], label="Val Murmur ROC-AUC", color="#2563eb", lw=2)
        ax.plot(epochs, history["val_murmur_f1"], label="Val Murmur F1", color="#3b82f6", lw=1.5, linestyle=":")
        ax.plot(epochs, history["val_outcome_auc"], label="Val Outcome ROC-AUC", color="#dc2626", lw=2)
        ax.plot(epochs, history["val_outcome_f1"], label="Val Outcome F1", color="#ef4444", lw=1.5, linestyle=":")
        ax.plot(epochs, history["val_combined_score"], label="Composite Score", color="#7c3aed", lw=2.5, linestyle="--")
        ax.set_title("Validation Metrics Across Training Epochs", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Metric Value", fontsize=11)
        ax.set_ylim(0.0, 1.05)
        ax.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")
        plt.tight_layout()
        metrics_path = self.output_dir / "metrics_curve.png"
        fig.savefig(metrics_path, dpi=300)
        plt.close(fig)

        return loss_path, metrics_path

    def _draw_confusion_matrix(self, cm: np.ndarray, labels: List[str], title: str, cmap_name: str = "Blues") -> plt.Figure:
        """Helper to draw heatmap confusion matrix using pure matplotlib."""
        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.get_cmap(cmap_name))
        
        # Color bar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9)

        # Axis ticks
        tick_marks = np.arange(len(labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)

        # Text annotations inside boxes
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > thresh else "black"
                ax.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                        color=color, fontsize=14, fontweight="bold")

        ax.grid(False)
        plt.tight_layout()
        return fig

    def plot_confusion_matrices(
        self,
        murmur_cm: List[List[int]],
        outcome_cm: List[List[int]]
    ) -> Tuple[Path, Path]:
        """Plots confusion matrices for Murmur and Clinical Outcome."""
        # Murmur CM
        cm_m = np.array(murmur_cm)
        fig_m = self._draw_confusion_matrix(
            cm_m,
            labels=["Murmur Absent", "Murmur Present"],
            title="Murmur Detection Confusion Matrix (Test)",
            cmap_name="Blues"
        )
        m_path = self.output_dir / "confusion_matrix_murmur.png"
        fig_m.savefig(m_path, dpi=300)
        plt.close(fig_m)

        # Outcome CM
        cm_o = np.array(outcome_cm)
        fig_o = self._draw_confusion_matrix(
            cm_o,
            labels=["Normal", "Abnormal"],
            title="Clinical Outcome Confusion Matrix (Test)",
            cmap_name="Greens"
        )
        o_path = self.output_dir / "confusion_matrix_outcome.png"
        fig_o.savefig(o_path, dpi=300)
        plt.close(fig_o)

        return m_path, o_path

    def plot_roc_pr_curves(
        self,
        raw_arrays: Dict[str, Any],
        m_metrics: Dict[str, Any],
        o_metrics: Dict[str, Any]
    ) -> Tuple[Path, Path]:
        """Plots ROC and Precision-Recall curves."""
        m_true = raw_arrays["murmur_trues"]
        m_prob = raw_arrays["murmur_probs"]
        o_true = raw_arrays["outcome_trues"]
        o_prob = raw_arrays["outcome_probs"]

        # 1. ROC Curves
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        fpr_m, tpr_m, _ = roc_curve(m_true, m_prob)
        fpr_o, tpr_o, _ = roc_curve(o_true, o_prob)

        ax.plot(fpr_m, tpr_m, color="#2563eb", lw=2.5, label=f"Murmur Detection (AUC = {m_metrics['roc_auc']:.3f})")
        ax.plot(fpr_o, tpr_o, color="#059669", lw=2.5, label=f"Clinical Outcome (AUC = {o_metrics['roc_auc']:.3f})")
        ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", lw=1.5, label="Random Chance (AUC = 0.500)")

        ax.set_title("Receiver Operating Characteristic (ROC) Curves (Test)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e2e8f0")
        plt.tight_layout()
        roc_path = self.output_dir / "roc_curves.png"
        fig.savefig(roc_path, dpi=300)
        plt.close(fig)

        # 2. PR Curves
        fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
        prec_m, rec_m, _ = precision_recall_curve(m_true, m_prob)
        prec_o, rec_o, _ = precision_recall_curve(o_true, o_prob)

        ax.plot(rec_m, prec_m, color="#2563eb", lw=2.5, label=f"Murmur Detection (PR-AUC = {m_metrics['pr_auc']:.3f})")
        ax.plot(rec_o, prec_o, color="#059669", lw=2.5, label=f"Clinical Outcome (PR-AUC = {o_metrics['pr_auc']:.3f})")

        ax.set_title("Precision-Recall Curves (Test)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
        ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="#e2e8f0")
        plt.tight_layout()
        pr_path = self.output_dir / "pr_curves.png"
        fig.savefig(pr_path, dpi=300)
        plt.close(fig)

        return roc_path, pr_path

    def plot_location_importance(
        self,
        attn_analysis: Dict[str, Any],
        ablation_results: Dict[str, Dict[str, float]]
    ) -> Path:
        """Plots attention weights and ablation impact by cardiac valve location."""
        loc_names = ["AV", "MV", "PV", "TV", "Phc"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

        # Subplot 1: Attention Distribution
        means_m_pres = [attn_analysis["murmur_present_mean_attention"][l] for l in loc_names]
        means_m_abs = [attn_analysis["murmur_absent_mean_attention"][l] for l in loc_names]

        x = np.arange(len(loc_names))
        width = 0.35

        ax1.bar(x - width/2, means_m_pres, width, label="Murmur Present", color="#dc2626", alpha=0.85)
        ax1.bar(x + width/2, means_m_abs, width, label="Murmur Absent", color="#3b82f6", alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(loc_names, fontweight="bold")
        ax1.set_title("Learned Attention Weight by Auscultation Site", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Auscultation Location", fontsize=10)
        ax1.set_ylabel("Mean Attention Weight", fontsize=10)
        ax1.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")

        # Subplot 2: Ablation Delta AUC
        delta_m_auc = [ablation_results.get(f"without_{l}", {}).get("delta_murmur_auc", 0.0) for l in loc_names]
        delta_o_auc = [ablation_results.get(f"without_{l}", {}).get("delta_outcome_auc", 0.0) for l in loc_names]

        ax2.bar(x - width/2, delta_m_auc, width, label="Murmur AUC Drop", color="#7c3aed", alpha=0.85)
        ax2.bar(x + width/2, delta_o_auc, width, label="Outcome AUC Drop", color="#059669", alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(loc_names, fontweight="bold")
        ax2.set_title("Performance Degradation upon Location Masking (Ablation)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Ablated Location", fontsize=10)
        ax2.set_ylabel(r"$\Delta$ ROC-AUC Drop", fontsize=10)
        ax2.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")

        plt.tight_layout()
        loc_path = self.output_dir / "location_importance.png"
        fig.savefig(loc_path, dpi=300)
        plt.close(fig)
        return loc_path

    def plot_probability_distributions(self, pred_df: pd.DataFrame) -> Path:
        """Plots probability histograms stratified by ground truth class."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

        # Murmur Probabilities
        abs_probs = pred_df[pred_df["true_murmur"] == 0]["murmur_probability"]
        pres_probs = pred_df[pred_df["true_murmur"] == 1]["murmur_probability"]
        
        bins = np.linspace(0.0, 1.0, 21)
        ax1.hist(abs_probs, bins=bins, alpha=0.6, color="#3b82f6", label="Murmur Absent (0)", density=True)
        ax1.hist(pres_probs, bins=bins, alpha=0.6, color="#dc2626", label="Murmur Present (1)", density=True)
        ax1.axvline(0.5, color="#64748b", linestyle="--", lw=1.5, label="Decision Threshold (0.50)")
        ax1.set_title("Murmur Predicted Probability Distribution", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Predicted Murmur Probability", fontsize=10)
        ax1.set_ylabel("Density", fontsize=10)
        ax1.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")

        # Outcome Probabilities
        norm_probs = pred_df[pred_df["true_clinical_outcome"] == 0]["clinical_outcome_probability"]
        abn_probs = pred_df[pred_df["true_clinical_outcome"] == 1]["clinical_outcome_probability"]
        
        ax2.hist(norm_probs, bins=bins, alpha=0.6, color="#059669", label="Normal Outcome (0)", density=True)
        ax2.hist(abn_probs, bins=bins, alpha=0.6, color="#d97706", label="Abnormal Outcome (1)", density=True)
        ax2.axvline(0.5, color="#64748b", linestyle="--", lw=1.5, label="Decision Threshold (0.50)")
        ax2.set_title("Clinical Outcome Predicted Probability Distribution", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Predicted Outcome Probability", fontsize=10)
        ax2.set_ylabel("Density", fontsize=10)
        ax2.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")

        plt.tight_layout()
        prob_path = self.output_dir / "prediction_probability_distribution.png"
        fig.savefig(prob_path, dpi=300)
        plt.close(fig)
        return prob_path

    def plot_class_distributions(self) -> Path:
        """Plots patient class distributions across splits."""
        splits = ["Train", "Validation", "Test"]
        absent_counts = [485, 104, 105]
        present_counts = [125, 27, 27]
        normal_counts = [322, 70, 69]
        abnormal_counts = [288, 61, 63]
        
        x = np.arange(len(splits))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

        # Murmur
        ax1.bar(x - width/2, absent_counts, width, label="Absent (0)", color="#3b82f6", alpha=0.85)
        ax1.bar(x + width/2, present_counts, width, label="Present (1)", color="#dc2626", alpha=0.85)
        ax1.set_xticks(x)
        ax1.set_xticklabels(splits, fontweight="bold")
        ax1.set_title("Murmur Target Split Distribution", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Number of Patients", fontsize=10)
        ax1.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")

        # Outcome
        ax2.bar(x - width/2, normal_counts, width, label="Normal (0)", color="#059669", alpha=0.85)
        ax2.bar(x + width/2, abnormal_counts, width, label="Abnormal (1)", color="#d97706", alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(splits, fontweight="bold")
        ax2.set_title("Clinical Outcome Split Distribution", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Number of Patients", fontsize=10)
        ax2.legend(frameon=True, facecolor="white", edgecolor="#e2e8f0")

        plt.tight_layout()
        class_path = self.output_dir / "class_distributions.png"
        fig.savefig(class_path, dpi=300)
        plt.close(fig)
        return class_path
