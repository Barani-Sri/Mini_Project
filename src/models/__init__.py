"""
Module 3: Patient-Level Deep Learning Models and Auscultation Fusion for Heart Murmur & Outcome Detection.
"""
from src.models.config import Module3Config
from src.models.dataset import CirCorPatientDataset, create_module3_dataloaders
from src.models.multi_location_net import (
    MultiLocationHeartSoundNet,
    SharedLocationEncoder,
    MaskedLocationAttentionFusion
)
from src.models.trainer import Module3Trainer, set_seed
from src.models.evaluator import Module3Evaluator
from src.models.explainability import Module3Explainability
from src.models.visualization import Module3Visualizer
from src.models.patient_aggregator import PatientPredictionAggregator
from src.models.baseline import PCGBaselineClassifier

__all__ = [
    "Module3Config",
    "CirCorPatientDataset",
    "create_module3_dataloaders",
    "MultiLocationHeartSoundNet",
    "SharedLocationEncoder",
    "MaskedLocationAttentionFusion",
    "Module3Trainer",
    "set_seed",
    "Module3Evaluator",
    "Module3Explainability",
    "Module3Visualizer",
    "PatientPredictionAggregator",
    "PCGBaselineClassifier"
]
