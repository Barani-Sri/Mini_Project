"""
Module 3 Multi-Task Training Engine with Early Stopping, Checkpointing, and Class Imbalance Weighting.
"""
import os
import random
import time
import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, roc_auc_score, balanced_accuracy_score

from src.models.config import (
    Module3Config,
    MODULE3_MODEL_PATH,
    MODULE3_PROCESSED_MODEL_PATH
)
from src.models.multi_location_net import MultiLocationHeartSoundNet

def set_seed(seed: int = 42):
    """Sets deterministic random seeds across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Module3Trainer:
    """
    Supervised Multi-Task Trainer for Multi-Location Heart Sound Model.
    """
    def __init__(
        self,
        model: MultiLocationHeartSoundNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[Module3Config] = None,
        model_save_path: Optional[Path] = None
    ):
        self.config = config if config is not None else Module3Config()
        self.device = torch.device(self.config.device if torch.cuda.is_available() and self.config.device == "cuda" else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_save_path = Path(model_save_path) if model_save_path is not None else MODULE3_MODEL_PATH

        set_seed(self.config.random_seed)

        # Weighted Binary Cross Entropy Losses (Weights strictly calculated from training data)
        self.murmur_pos_weight = torch.tensor([self.config.murmur_pos_weight], device=self.device)
        self.outcome_pos_weight = torch.tensor([self.config.outcome_pos_weight], device=self.device)
        
        self.murmur_criterion = nn.BCEWithLogitsLoss(pos_weight=self.murmur_pos_weight)
        self.outcome_criterion = nn.BCEWithLogitsLoss(pos_weight=self.outcome_pos_weight)

        # Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience,
            min_lr=self.config.min_lr
        )

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_murmur_loss": [],
            "train_outcome_loss": [],
            "val_loss": [],
            "val_murmur_loss": [],
            "val_outcome_loss": [],
            "val_murmur_f1": [],
            "val_murmur_auc": [],
            "val_outcome_f1": [],
            "val_outcome_auc": [],
            "val_combined_score": [],
            "lr": []
        }
        
        self.best_val_score = -float("inf")
        self.best_epoch = 0
        self.best_val_metrics: Dict[str, float] = {}

    def _train_epoch(self) -> Tuple[float, float, float]:
        """Trains for one epoch over the training split."""
        self.model.train()
        total_loss, total_m_loss, total_o_loss = 0.0, 0.0, 0.0
        num_batches = 0

        for batch in self.train_loader:
            features = batch["features"].to(self.device)         # (B, 5, 120, 501)
            location_mask = batch["location_mask"].to(self.device) # (B, 5)
            y_murmur = batch["murmur_label"].unsqueeze(1).to(self.device)   # (B, 1)
            y_outcome = batch["outcome_label"].unsqueeze(1).to(self.device) # (B, 1)

            self.optimizer.zero_grad()
            outputs = self.model(features, location_mask)

            loss_m = self.murmur_criterion(outputs["murmur_logits"], y_murmur)
            loss_o = self.outcome_criterion(outputs["outcome_logits"], y_outcome)
            loss = self.config.task_weight_murmur * loss_m + self.config.task_weight_outcome * loss_o

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_m_loss += loss_m.item()
            total_o_loss += loss_o.item()
            num_batches += 1

        return (
            total_loss / num_batches,
            total_m_loss / num_batches,
            total_o_loss / num_batches
        )

    def _evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluates on a dataloader without gradients."""
        self.model.eval()
        total_loss, total_m_loss, total_o_loss = 0.0, 0.0, 0.0
        num_batches = 0

        all_m_true, all_m_prob = [], []
        all_o_true, all_o_prob = [], []

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                location_mask = batch["location_mask"].to(self.device)
                y_murmur = batch["murmur_label"].unsqueeze(1).to(self.device)
                y_outcome = batch["outcome_label"].unsqueeze(1).to(self.device)

                outputs = self.model(features, location_mask)

                loss_m = self.murmur_criterion(outputs["murmur_logits"], y_murmur)
                loss_o = self.outcome_criterion(outputs["outcome_logits"], y_outcome)
                loss = self.config.task_weight_murmur * loss_m + self.config.task_weight_outcome * loss_o

                total_loss += loss.item()
                total_m_loss += loss_m.item()
                total_o_loss += loss_o.item()
                num_batches += 1

                all_m_true.extend(y_murmur.cpu().numpy().ravel())
                all_m_prob.extend(outputs["murmur_probs"].cpu().numpy().ravel())
                all_o_true.extend(y_outcome.cpu().numpy().ravel())
                all_o_prob.extend(outputs["outcome_probs"].cpu().numpy().ravel())

        all_m_true = np.array(all_m_true)
        all_m_prob = np.array(all_m_prob)
        all_o_true = np.array(all_o_true)
        all_o_prob = np.array(all_o_prob)

        m_pred = (all_m_prob >= self.config.murmur_threshold).astype(int)
        o_pred = (all_o_prob >= self.config.outcome_threshold).astype(int)

        m_f1 = float(f1_score(all_m_true, m_pred, zero_division=0))
        o_f1 = float(f1_score(all_o_true, o_pred, zero_division=0))
        
        try:
            m_auc = float(roc_auc_score(all_m_true, all_m_prob))
        except Exception:
            m_auc = 0.5
            
        try:
            o_auc = float(roc_auc_score(all_o_true, all_o_prob))
        except Exception:
            o_auc = 0.5

        # Combined composite score for model checkpointing
        combined_score = 0.5 * (m_auc + m_f1) + 0.5 * (o_auc + o_f1)

        return {
            "val_loss": total_loss / num_batches,
            "val_murmur_loss": total_m_loss / num_batches,
            "val_outcome_loss": total_o_loss / num_batches,
            "val_murmur_f1": m_f1,
            "val_murmur_auc": m_auc,
            "val_outcome_f1": o_f1,
            "val_outcome_auc": o_auc,
            "val_combined_score": combined_score
        }

    def train(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Executes full multi-task training with early stopping.
        """
        if verbose:
            print("=" * 70, flush=True)
            print(" MODULE 3: MULTI-LOCATION MULTI-TASK NEURAL NETWORK TRAINING", flush=True)
            print(f" Device:              {self.device}", flush=True)
            print(f" Max Epochs:          {self.config.max_epochs} | Batch Size: {self.config.batch_size}", flush=True)
            print(f" Learning Rate:       {self.config.learning_rate} | Weight Decay: {self.config.weight_decay}", flush=True)
            print(f" Murmur Pos Weight:   {self.config.murmur_pos_weight:.3f} | Outcome Pos Weight: {self.config.outcome_pos_weight:.3f}", flush=True)
            print(f" Early Stop Patience: {self.config.early_stopping_patience} epochs", flush=True)
            print("=" * 70, flush=True)

        epochs_without_improvement = 0
        start_time = time.time()

        for epoch in range(1, self.config.max_epochs + 1):
            t_loss, t_m_loss, t_o_loss = self._train_epoch()
            val_metrics = self._evaluate(self.val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_metrics["val_combined_score"])

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(t_loss)
            self.history["train_murmur_loss"].append(t_m_loss)
            self.history["train_outcome_loss"].append(t_o_loss)
            self.history["val_loss"].append(val_metrics["val_loss"])
            self.history["val_murmur_loss"].append(val_metrics["val_murmur_loss"])
            self.history["val_outcome_loss"].append(val_metrics["val_outcome_loss"])
            self.history["val_murmur_f1"].append(val_metrics["val_murmur_f1"])
            self.history["val_murmur_auc"].append(val_metrics["val_murmur_auc"])
            self.history["val_outcome_f1"].append(val_metrics["val_outcome_f1"])
            self.history["val_outcome_auc"].append(val_metrics["val_outcome_auc"])
            self.history["val_combined_score"].append(val_metrics["val_combined_score"])
            self.history["lr"].append(current_lr)

            score = val_metrics["val_combined_score"]
            is_best = (score - self.best_val_score) > self.config.min_delta

            if is_best:
                self.best_val_score = score
                self.best_epoch = epoch
                self.best_val_metrics = val_metrics.copy()
                epochs_without_improvement = 0

                # Save best model checkpoint
                self._save_checkpoint(epoch, val_metrics)
                star = " *"
            else:
                epochs_without_improvement += 1
                star = ""

            if verbose and (epoch % 1 == 0 or epoch == self.config.max_epochs or is_best):
                print(
                    f"Epoch {epoch:02d}/{self.config.max_epochs:02d} | "
                    f"Train Loss: {t_loss:.4f} | "
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Murmur AUC/F1: {val_metrics['val_murmur_auc']:.3f}/{val_metrics['val_murmur_f1']:.3f} | "
                    f"Outcome AUC/F1: {val_metrics['val_outcome_auc']:.3f}/{val_metrics['val_outcome_f1']:.3f}{star}",
                    flush=True
                )

            if epochs_without_improvement >= self.config.early_stopping_patience:
                if verbose:
                    print(f"\n[Early Stopping] No improvement for {self.config.early_stopping_patience} consecutive epochs. Stopping at epoch {epoch}.", flush=True)
                break

        elapsed_time = time.time() - start_time
        if verbose:
            print(f"\nTraining completed in {elapsed_time:.1f}s. Best Epoch: {self.best_epoch} (Combined Score: {self.best_val_score:.4f})", flush=True)

        # Reload best model weights into current model
        self.load_best_checkpoint()

        return {
            "best_epoch": self.best_epoch,
            "best_val_score": self.best_val_score,
            "best_val_metrics": self.best_val_metrics,
            "elapsed_seconds": elapsed_time,
            "history": self.history
        }

    def _save_checkpoint(self, epoch: int, val_metrics: Dict[str, float]):
        """Persists best model checkpoint to primary and processed paths."""
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config_dict": dataclasses.asdict(self.config),
            "val_metrics": val_metrics
        }
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_dict, str(self.model_save_path))
        
        MODULE3_PROCESSED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint_dict, str(MODULE3_PROCESSED_MODEL_PATH))

    def load_best_checkpoint(self):
        """Loads weights from saved best checkpoint file."""
        if self.model_save_path.exists():
            try:
                checkpoint = torch.load(str(self.model_save_path), map_location=self.device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(str(self.model_save_path), map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
