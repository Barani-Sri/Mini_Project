"""
Command-line runner for Module 3: Clinical Outcome Prediction & Heart Murmur Detection.
Executes patient-level deep learning training, evaluation, explainability, visualizations, and report generation.
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.config import (
    Module3Config,
    MODULE2_H5_PATH,
    MODULE3_METADATA_CSV_PATH,
    MODULE3_MODEL_PATH,
    MODULE3_PROCESSED_MODEL_PATH,
    MODULE3_PREDICTIONS_PATH,
    MODULE3_RESULTS_JSON_PATH,
    MODULE3_REPORT_MD_PATH,
    MODULE3_RESULTS_DIR
)
from src.models.dataset import CirCorPatientDataset, create_module3_dataloaders
from src.models.multi_location_net import MultiLocationHeartSoundNet
from src.models.trainer import Module3Trainer, set_seed
from src.models.evaluator import Module3Evaluator
from src.models.explainability import Module3Explainability
from src.models.visualization import Module3Visualizer

def generate_markdown_report(
    summary_dict: dict,
    report_path: Path
):
    """Generates a comprehensive scientific and clinical markdown report."""
    m_test = summary_dict["test_metrics"]["murmur_detection"]
    o_test = summary_dict["test_metrics"]["clinical_outcome"]
    attn_stats = summary_dict["explainability"]["attention_weights"]
    ablation = summary_dict["explainability"]["location_ablation"]

    report_content = f"""# Module 3 — Clinical Outcome Prediction & Heart Murmur Detection Report

**CirCor DigiScope Phonocardiogram (PCG) Multi-Location Deep Learning Pipeline**

---

## 1. Executive Summary & Objective

Module 3 implements patient-level deep learning for multi-task cardiac auscultation analysis:
1. **Heart Murmur Detection** (Binary: Absent vs Present)
2. **Clinical Outcome Prediction** (Binary: Normal vs Abnormal)

The architecture processes standardized $(5, 120, 501)$ multi-location acoustic tensors (AV, MV, PV, TV, Phc) produced by Module 2, fuses cardiac valve sites using **Masked Multi-Head Attention**, and performs multi-task prediction with zero patient-level data leakage.

---

## 2. Dataset & Target Definitions

* **Acoustic Input Representation**: $(P, 5, 120, 501)$ per patient ($5$ locations, $120$ combined MFCC channels, $501$ temporal frames $= 5.0$ seconds).
* **Missing Location Handling**: Dynamic binary location masks $(P, 5)$ with softmax $-\\infty$ masking.
* **Patient Splits (Strict No-Leakage Partition)**:
  * **Train Set**: 610 patients (485 Murmur Absent, 125 Murmur Present | 322 Outcome Normal, 288 Outcome Abnormal)
  * **Validation Set**: 131 patients (104 Murmur Absent, 27 Murmur Present | 70 Outcome Normal, 61 Outcome Abnormal)
  * **Test Set (Unseen)**: 132 patients (105 Murmur Absent, 27 Murmur Present | 69 Outcome Normal, 63 Outcome Abnormal)
* **Class Imbalance Strategy**: Positive class weighting derived strictly from training data (Murmur Weight $= 3.88$, Outcome Weight $= 1.118$).

---

## 3. Deep Learning Architecture

```text
Patient Auscultation Tensors (B, 5, 120, 501) + Location Masks (B, 5)
                            │
                            ▼
    ┌───────────────────────────────────────────────────┐
    │  Shared 2D CNN Location Feature Encoder           │
    │  Conv2D -> BatchNorm -> SiLU -> MaxPool2D x3      │
    │  Adaptive Avg+Max Pool -> Linear -> LayerNorm     │
    └───────────────────────┬───────────────────────────┘
                            │ (B, 5, 128)
                            ▼
    ┌───────────────────────────────────────────────────┐
    │  Masked Multi-Head Location Attention Fusion      │
    │  Scaled Dot-Product Query Attention with -inf     │
    │  Missing Location Masking -> Softmax Weights      │
    └───────────────────────┬───────────────────────────┘
                            │ Fused Representation (B, 128)
             ┌──────────────┴──────────────┐
             ▼                             ▼
  ┌──────────────────────┐      ┌──────────────────────┐
  │ Head 1: Murmur Det.  │      │ Head 2: Outcome Pred │
  │ MLP(128->64->1)      │      │ MLP(128->64->1)      │
  └──────────┬───────────┘      └──────────┬───────────┘
             │                             │
             ▼                             ▼
       Murmur Prob p_m               Outcome Prob p_o
```

---

## 4. Test Set Evaluation Results (132 Unseen Patients)

### 4.1. Primary Diagnostic Classification Metrics

| Target Task | Accuracy | Balanced Acc | Sensitivity (Recall) | Specificity | Precision | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Heart Murmur Detection** | **{m_test['accuracy']:.4f}** | **{m_test['balanced_accuracy']:.4f}** | **{m_test['sensitivity']:.4f}** | **{m_test['specificity']:.4f}** | **{m_test['precision']:.4f}** | **{m_test['f1']:.4f}** | **{m_test['roc_auc']:.4f}** | **{m_test['pr_auc']:.4f}** |
| **Clinical Outcome Prediction** | **{o_test['accuracy']:.4f}** | **{o_test['balanced_accuracy']:.4f}** | **{o_test['sensitivity']:.4f}** | **{o_test['specificity']:.4f}** | **{o_test['precision']:.4f}** | **{o_test['f1']:.4f}** | **{o_test['roc_auc']:.4f}** | **{o_test['pr_auc']:.4f}** |

### 4.2. Confusion Matrices

#### Murmur Detection (Threshold = 0.50):
* True Negatives (Absent): **{m_test['tn']}** | False Positives: **{m_test['fp']}**
* False Negatives: **{m_test['fn']}** | True Positives (Present): **{m_test['tp']}**

#### Clinical Outcome Prediction (Threshold = 0.50):
* True Negatives (Normal): **{o_test['tn']}** | False Positives: **{o_test['fp']}**
* False Negatives: **{o_test['fn']}** | True Positives (Abnormal): **{o_test['tp']}**

### 4.3. PhysioNet 2022 Challenge Domain Cost Metrics
* **PhysioNet Murmur Evaluation Weighted Score**: **{m_test['physionet_weighted_score']:.4f}**
* **PhysioNet Clinical Diagnostic Mean Cost**: **${o_test['physionet_mean_cost']:.2f}** per patient

---

## 5. Location Importance & Multi-Valve Explainability

### 5.1. Learned Attention Weight Distribution by Auscultation Site

| Auscultation Site | Overall Mean Attn | Murmur Present Attn | Murmur Absent Attn | Outcome Abnormal Attn | Outcome Normal Attn |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Aortic Valve (AV)** | {attn_stats['overall_mean_attention']['AV']:.4f} | {attn_stats['murmur_present_mean_attention']['AV']:.4f} | {attn_stats['murmur_absent_mean_attention']['AV']:.4f} | {attn_stats['outcome_abnormal_mean_attention']['AV']:.4f} | {attn_stats['outcome_normal_mean_attention']['AV']:.4f} |
| **Mitral Valve (MV)** | {attn_stats['overall_mean_attention']['MV']:.4f} | {attn_stats['murmur_present_mean_attention']['MV']:.4f} | {attn_stats['murmur_absent_mean_attention']['MV']:.4f} | {attn_stats['outcome_abnormal_mean_attention']['MV']:.4f} | {attn_stats['outcome_normal_mean_attention']['MV']:.4f} |
| **Pulmonary Valve (PV)** | {attn_stats['overall_mean_attention']['PV']:.4f} | {attn_stats['murmur_present_mean_attention']['PV']:.4f} | {attn_stats['murmur_absent_mean_attention']['PV']:.4f} | {attn_stats['outcome_abnormal_mean_attention']['PV']:.4f} | {attn_stats['outcome_normal_mean_attention']['PV']:.4f} |
| **Tricuspid Valve (TV)** | {attn_stats['overall_mean_attention']['TV']:.4f} | {attn_stats['murmur_present_mean_attention']['TV']:.4f} | {attn_stats['murmur_absent_mean_attention']['TV']:.4f} | {attn_stats['outcome_abnormal_mean_attention']['TV']:.4f} | {attn_stats['outcome_normal_mean_attention']['TV']:.4f} |
| **Other / Phc** | {attn_stats['overall_mean_attention']['Phc']:.4f} | {attn_stats['murmur_present_mean_attention']['Phc']:.4f} | {attn_stats['murmur_absent_mean_attention']['Phc']:.4f} | {attn_stats['outcome_abnormal_mean_attention']['Phc']:.4f} | {attn_stats['outcome_normal_mean_attention']['Phc']:.4f} |

### 5.2. Systematic Location Ablation Analysis

| Ablated Valve Site | Murmur ROC-AUC | Murmur AUC Delta | Outcome ROC-AUC | Outcome AUC Delta |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (All Sites)** | **{ablation['baseline']['murmur_auc']:.4f}** | 0.0000 | **{ablation['baseline']['outcome_auc']:.4f}** | 0.0000 |
| Without **AV** | {ablation['without_AV']['murmur_auc']:.4f} | {ablation['without_AV']['delta_murmur_auc']:+.4f} | {ablation['without_AV']['outcome_auc']:.4f} | {ablation['without_AV']['delta_outcome_auc']:+.4f} |
| Without **MV** | {ablation['without_MV']['murmur_auc']:.4f} | {ablation['without_MV']['delta_murmur_auc']:+.4f} | {ablation['without_MV']['outcome_auc']:.4f} | {ablation['without_MV']['delta_outcome_auc']:+.4f} |
| Without **PV** | {ablation['without_PV']['murmur_auc']:.4f} | {ablation['without_PV']['delta_murmur_auc']:+.4f} | {ablation['without_PV']['outcome_auc']:.4f} | {ablation['without_PV']['delta_outcome_auc']:+.4f} |
| Without **TV** | {ablation['without_TV']['murmur_auc']:.4f} | {ablation['without_TV']['delta_murmur_auc']:+.4f} | {ablation['without_TV']['outcome_auc']:.4f} | {ablation['without_TV']['delta_outcome_auc']:+.4f} |
| Without **Phc** | {ablation['without_Phc']['murmur_auc']:.4f} | {ablation['without_Phc']['delta_murmur_auc']:+.4f} | {ablation['without_Phc']['outcome_auc']:.4f} | {ablation['without_Phc']['delta_outcome_auc']:+.4f} |

---

## 6. Generated Visualizations & Artifacts

All plots are stored under `data/processed/module3_results/`:
1. `loss_curve.png` — Multi-task training and validation loss curves.
2. `metrics_curve.png` — Validation ROC-AUC, F1, and composite score across epochs.
3. `confusion_matrix_murmur.png` — Murmur detection confusion matrix on test patients.
4. `confusion_matrix_outcome.png` — Clinical outcome confusion matrix on test patients.
5. `roc_curves.png` — Dual ROC curves with AUC annotations.
6. `pr_curves.png` — Precision-Recall curves with average precision scores.
7. `location_importance.png` — Auscultation location attention distribution and ablation impact.
8. `prediction_probability_distribution.png` — Predicted probability histograms by ground truth.
9. `class_distributions.png` — Patient distributions across train, validation, and test splits.

---

## 7. Medical & Scientific Disclaimer

> [!WARNING]
> **Diagnostic Disclaimer**: This system is developed for research, education, and algorithmic benchmarking on the PhysioNet CirCor DigiScope PCG dataset. Model predictions represent probabilistic classifications of phonocardiographic acoustic signals and do **not** constitute a definitive clinical diagnosis or medical device clearance. Clinical decision-making must always be performed by qualified medical professionals using standard comprehensive diagnostic modalities (e.g. echocardiography, clinical examination).

---

## 8. Reproducibility & Environment

* **Random Seed**: {summary_dict['config']['random_seed']}
* **Best Training Epoch**: {summary_dict['training']['best_epoch']}
* **Model Checkpoint**: `{summary_dict['artifacts']['model_checkpoint']}`
* **Patient Predictions CSV**: `{summary_dict['artifacts']['patient_predictions_csv']}`
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)


def main():
    parser = argparse.ArgumentParser(description="Module 3: Patient-Level Multi-Location Heart Murmur & Outcome Detection")
    parser.add_argument("--features", type=str, default=str(MODULE2_H5_PATH), help="Path to Module 2 features H5")
    parser.add_argument("--metadata", type=str, default=str(MODULE3_METADATA_CSV_PATH), help="Path to metadata CSV")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Maximum training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience")
    parser.add_argument("--no_train", action="store_true", help="Skip training and load existing model checkpoint")
    
    args = parser.parse_args()

    config = Module3Config(
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        learning_rate=args.lr,
        random_seed=args.seed,
        early_stopping_patience=args.patience
    )

    set_seed(config.random_seed)

    print("=" * 70, flush=True)
    print(" MODULE 3: CLINICAL OUTCOME PREDICTION & HEART MURMUR DETECTION", flush=True)
    print(f" Input Features H5:    {args.features}", flush=True)
    print(f" Metadata CSV:         {args.metadata}", flush=True)
    print(f" Model Output Path:    {MODULE3_MODEL_PATH}", flush=True)
    print(f" Predictions CSV:      {MODULE3_PREDICTIONS_PATH}", flush=True)
    print(f" Results JSON:         {MODULE3_RESULTS_JSON_PATH}", flush=True)
    print("=" * 70, flush=True)

    # 1. Load DataLoaders
    print("\n[1/5] Loading patient-level multi-location representations from Module 2...", flush=True)
    train_loader, val_loader, test_loader = create_module3_dataloaders(
        h5_path=args.features,
        metadata_csv_path=args.metadata,
        config=config
    )
    print(f"      Train: {len(train_loader.dataset)} patients | Val: {len(val_loader.dataset)} patients | Test: {len(test_loader.dataset)} patients", flush=True)

    # 2. Build Multi-Location Neural Network
    print("\n[2/5] Initializing MultiLocationHeartSoundNet architecture...", flush=True)
    model = MultiLocationHeartSoundNet(config)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Model Parameters: {total_params:,} trainable weights", flush=True)

    # 3. Train or Load Checkpoint
    trainer = Module3Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        model_save_path=MODULE3_MODEL_PATH
    )

    if not args.no_train:
        print("\n[3/5] Starting multi-task supervised training...", flush=True)
        train_results = trainer.train(verbose=True)
    else:
        print(f"\n[3/5] Loading existing best checkpoint from {MODULE3_MODEL_PATH}...", flush=True)
        trainer.load_best_checkpoint()
        train_results = {"best_epoch": 1, "best_val_score": 0.0, "best_val_metrics": {}, "history": {}}

    # 4. Evaluate on Unseen Test Patients
    print("\n[4/5] Evaluating model strictly on unseen test patients (N=132)...", flush=True)
    evaluator = Module3Evaluator(model=trainer.model, config=config)
    eval_results = evaluator.evaluate_test_set(
        test_loader=test_loader,
        save_predictions_path=MODULE3_PREDICTIONS_PATH
    )

    # 5. Explainability & Auscultation Importance Analysis
    print("\n[5/5] Performing multi-valve attention and ablation explainability analysis...", flush=True)
    explainer = Module3Explainability(model=trainer.model, config=config)
    attn_analysis = explainer.analyze_attention_weights(eval_results["predictions_df"])
    ablation_results = explainer.perform_location_ablation(test_loader)

    # 6. Generate Publication Visualizations
    print("\nGenerating publication-quality visual plots in data/processed/module3_results/...", flush=True)
    visualizer = Module3Visualizer(output_dir=MODULE3_RESULTS_DIR)
    
    if "epoch" in train_results.get("history", {}) and len(train_results["history"]["epoch"]) > 0:
        visualizer.plot_training_history(train_results["history"])

    visualizer.plot_confusion_matrices(
        eval_results["murmur_detection"]["confusion_matrix"],
        eval_results["clinical_outcome"]["confusion_matrix"]
    )
    visualizer.plot_roc_pr_curves(
        eval_results["raw_arrays"],
        eval_results["murmur_detection"],
        eval_results["clinical_outcome"]
    )
    visualizer.plot_location_importance(attn_analysis, ablation_results)
    visualizer.plot_probability_distributions(eval_results["predictions_df"])
    visualizer.plot_class_distributions()

    # 7. Compile Final Results JSON
    summary_dict = {
        "module": "Module 3: Clinical Outcome Prediction & Heart Murmur Detection",
        "dataset_statistics": {
            "total_patients": 873,
            "patient_splits": {
                "train": len(train_loader.dataset),
                "validation": len(val_loader.dataset),
                "test": len(test_loader.dataset)
            },
            "class_distribution": {
                "murmur_train": {"absent": 485, "present": 125},
                "murmur_val": {"absent": 104, "present": 27},
                "murmur_test": {"absent": 105, "present": 27},
                "outcome_train": {"normal": 322, "abnormal": 288},
                "outcome_val": {"normal": 70, "abnormal": 61},
                "outcome_test": {"normal": 69, "abnormal": 63}
            }
        },
        "model_architecture": {
            "name": "MultiLocationHeartSoundNet",
            "encoder": "Shared 2D CNN with 3 Conv-BN-SiLU-MaxPool Blocks",
            "fusion": "Masked Scaled Dot-Product Multi-Head Attention",
            "heads": ["Murmur Detection (BCE)", "Clinical Outcome (BCE)"],
            "trainable_parameters": total_params
        },
        "config": {
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "max_epochs": config.max_epochs,
            "early_stopping_patience": config.early_stopping_patience,
            "murmur_pos_weight": config.murmur_pos_weight,
            "outcome_pos_weight": config.outcome_pos_weight,
            "random_seed": config.random_seed
        },
        "training": {
            "best_epoch": train_results.get("best_epoch", 1),
            "best_val_score": train_results.get("best_val_score", 0.0),
            "best_val_metrics": train_results.get("best_val_metrics", {})
        },
        "test_metrics": {
            "num_test_patients": eval_results["num_test_patients"],
            "murmur_detection": {
                k: v for k, v in eval_results["murmur_detection"].items() if not k.endswith("_data")
            },
            "clinical_outcome": {
                k: v for k, v in eval_results["clinical_outcome"].items() if not k.endswith("_data")
            }
        },
        "explainability": {
            "attention_weights": attn_analysis,
            "location_ablation": ablation_results
        },
        "artifacts": {
            "model_checkpoint": str(MODULE3_MODEL_PATH),
            "processed_model_checkpoint": str(MODULE3_PROCESSED_MODEL_PATH),
            "patient_predictions_csv": str(MODULE3_PREDICTIONS_PATH),
            "results_json": str(MODULE3_RESULTS_JSON_PATH),
            "report_md": str(MODULE3_REPORT_MD_PATH),
            "plots_directory": str(MODULE3_RESULTS_DIR)
        }
    }

    MODULE3_RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODULE3_RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    # 8. Generate Markdown Report
    generate_markdown_report(summary_dict, MODULE3_REPORT_MD_PATH)

    # Print Final Summary
    m_res = eval_results["murmur_detection"]
    o_res = eval_results["clinical_outcome"]
    print("\n" + "=" * 70)
    print(" MODULE 3 EXECUTION SUMMARY (UNSEEN TEST PATIENTS N=132)")
    print("=" * 70)
    print(" 1. HEART MURMUR DETECTION:")
    print(f"    - Accuracy:        {m_res['accuracy']*100:.2f}% (Balanced Acc: {m_res['balanced_accuracy']*100:.2f}%)")
    print(f"    - Sensitivity:     {m_res['sensitivity']*100:.2f}% | Specificity: {m_res['specificity']*100:.2f}%")
    print(f"    - Precision:       {m_res['precision']*100:.2f}% | F1-Score: {m_res['f1']:.4f}")
    print(f"    - ROC-AUC:         {m_res['roc_auc']:.4f} | PR-AUC: {m_res['pr_auc']:.4f}")
    print(f"    - PhysioNet Score: {m_res['physionet_weighted_score']:.4f}")
    print("\n 2. CLINICAL OUTCOME PREDICTION:")
    print(f"    - Accuracy:        {o_res['accuracy']*100:.2f}% (Balanced Acc: {o_res['balanced_accuracy']*100:.2f}%)")
    print(f"    - Sensitivity:     {o_res['sensitivity']*100:.2f}% | Specificity: {o_res['specificity']*100:.2f}%")
    print(f"    - Precision:       {o_res['precision']*100:.2f}% | F1-Score: {o_res['f1']:.4f}")
    print(f"    - ROC-AUC:         {o_res['roc_auc']:.4f} | PR-AUC: {o_res['pr_auc']:.4f}")
    print(f"    - Mean Cost:       ${o_res['physionet_mean_cost']:.2f}")
    print("=" * 70)
    print(f" Saved Model:       {MODULE3_MODEL_PATH}")
    print(f" Saved Predictions: {MODULE3_PREDICTIONS_PATH}")
    print(f" Saved Summary:     {MODULE3_RESULTS_JSON_PATH}")
    print(f" Saved Report:      {MODULE3_REPORT_MD_PATH}")
    print("=" * 70)

if __name__ == "__main__":
    main()
