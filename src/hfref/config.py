"""
Configuration for HFrEF Multi-Channel ECG and PCG Module.
Defines paths, sampling rates, DSP filter cutoffs, valve mappings,
and diagnostic thresholds.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

# Base and data directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# HFrEF specific paths
HFREF_DIR = DATA_DIR / "hfref"
HFREF_ZIP_PATH = HFREF_DIR / "HFrEF_5_folds.zip"
HFREF_EXTRACTED_DIR = HFREF_DIR / "extracted"
HFREF_MODEL_DIR = MODELS_DIR / "hfref"
HFREF_MODEL_PATH = HFREF_MODEL_DIR / "hfref_classifier.joblib"
HFREF_METRICS_PATH = HFREF_MODEL_DIR / "hfref_cv_metrics.json"

# Native dataset specifications (EKO DUO ECG + Digital Stethoscope)
HFREF_ECG_SAMPLE_RATE = 500     # 500 Hz native for ECG
HFREF_PCG_SAMPLE_RATE = 4000    # 4000 Hz native for PCG
HFREF_PCG_TARGET_SR = 2000      # 2000 Hz target for acoustic extraction
HFREF_RECORDING_DURATION_SEC = 30.0

# Auscultation site code mappings
# Official dataset uses APEX, LLSB, LUSB, RUSB
AUSCULTATION_CODE_MAP: Dict[str, Dict[str, str]] = {
    "APEX": {
        "lubdub_code": "MV",
        "valve_name": "Mitral Valve",
        "anatomical_site": "Cardiac Apex (5th Left Intercostal Space)",
        "full_code": "APEX"
    },
    "LLSB": {
        "lubdub_code": "TV",
        "valve_name": "Tricuspid Valve",
        "anatomical_site": "Lower Left Sternal Border (4th Intercostal Space)",
        "full_code": "LLSB"
    },
    "LUSB": {
        "lubdub_code": "PV",
        "valve_name": "Pulmonary Valve",
        "anatomical_site": "Left Upper Sternal Border (2nd Intercostal Space)",
        "full_code": "LUSB"
    },
    "RUSB": {
        "lubdub_code": "AV",
        "valve_name": "Aortic Valve",
        "anatomical_site": "Right Upper Sternal Border (2nd Intercostal Space)",
        "full_code": "RUSB"
    }
}

# Reverse mapping: LubDub valve code -> HFrEF site code
REVERSE_CODE_MAP: Dict[str, str] = {
    "MV": "APEX",
    "TV": "LLSB",
    "PV": "LUSB",
    "AV": "RUSB"
}

ORDERED_HFREF_LOCATIONS = ["RUSB", "LUSB", "LLSB", "APEX"]
ORDERED_VALVE_CODES = ["AV", "PV", "TV", "MV"]

# Clinical diagnostic thresholds
# Official criteria: LVEF <= 40% is HFrEF (reduced ejection fraction), LVEF > 40% is non-HFrEF
LVEF_HFREF_THRESHOLD = 40.0

# ECG DSP parameters
ECG_BANDPASS_LOW_HZ = 0.5
ECG_BANDPASS_HIGH_HZ = 45.0
ECG_FILTER_ORDER = 3
ECG_NOTCH_FREQ = 50.0  # Optional powerline filter (50Hz)

# PCG DSP parameters
PCG_BANDPASS_LOW_HZ = 25.0
PCG_BANDPASS_HIGH_HZ = 400.0
PCG_FILTER_ORDER = 4
PCG_NORM_METHOD = "peak"
PCG_TARGET_PEAK = 0.95

# Feature dimensions
ECG_FEATURE_DIM = 24
PCG_FEATURE_PER_LOC_DIM = 48  # 48 features per location
TOTAL_FUSED_DIM = ECG_FEATURE_DIM + (PCG_FEATURE_PER_LOC_DIM * len(ORDERED_HFREF_LOCATIONS)) # 24 + 192 = 216

@dataclass
class HFrEFConfig:
    """Config dataclass for HFrEF analysis."""
    ecg_sr: int = HFREF_ECG_SAMPLE_RATE
    pcg_sr: int = HFREF_PCG_SAMPLE_RATE
    pcg_target_sr: int = HFREF_PCG_TARGET_SR
    lvef_threshold: float = LVEF_HFREF_THRESHOLD
    n_folds: int = 5
    random_state: int = 42
