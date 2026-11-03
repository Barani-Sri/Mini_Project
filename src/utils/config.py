"""
Configuration constants and paths for Heart Murmur detection pipeline.
Contains global settings and specific configurations for Module 1.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "circor"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Audio DSP Parameters
SAMPLE_RATE = 4000  # Native CirCor dataset sampling rate
TARGET_SAMPLE_RATE = 4000
BANDPASS_LOW_HZ = 25.0
BANDPASS_HIGH_HZ = 400.0
BUTTERWORTH_ORDER = 4

# =====================================================================
# MODULE 1 SPECIFIC PARAMETERS
# =====================================================================
MODULE1_TARGET_SAMPLE_RATE = 2000  # Default 2000 Hz for Module 1
MODULE1_BANDPASS_LOW_HZ = 25.0
MODULE1_BANDPASS_HIGH_HZ = 400.0
MODULE1_BANDPASS_ORDER = 5         # Order 5 Butterworth for Module 1
MODULE1_NOTCH_ENABLED = False      # Optional 50Hz/60Hz notch filter
MODULE1_NOTCH_FREQ = 50.0
MODULE1_NOTCH_Q = 30.0
MODULE1_NORM_METHOD = "peak"       # Peak normalization default
MODULE1_TARGET_PEAK = 0.95         # Scale to 0.95 peak
MODULE1_WINDOW_SEC = 5.0           # 5-second fixed window
MODULE1_HOP_SEC = 2.5              # 2.5-second hop (50% overlap)
MODULE1_WINDOW_SAMPLES = int(MODULE1_WINDOW_SEC * MODULE1_TARGET_SAMPLE_RATE)  # 10,000
MODULE1_HOP_SAMPLES = int(MODULE1_HOP_SEC * MODULE1_TARGET_SAMPLE_RATE)        # 5,000

# Splitting Ratios
MODULE1_TRAIN_RATIO = 0.70
MODULE1_VAL_RATIO = 0.15
MODULE1_TEST_RATIO = 0.15
MODULE1_RANDOM_STATE = 42

# Module 1 Output Artifact Paths
MODULE1_H5_PATH = PROCESSED_DATA_DIR / "module1_processed.h5"
MODULE1_SUMMARY_PATH = PROCESSED_DATA_DIR / "module1_summary.json"
MODULE1_ERRORS_PATH = PROCESSED_DATA_DIR / "module1_errors.csv"

# Augmentation Parameters (Training Only)
AUG_MAX_TIME_SHIFT_SEC = 0.2
AUG_NOISE_STD = 0.005
AUG_TIME_STRETCH_RANGE = (0.95, 1.05)

# PCG Segmentation States
STATE_UNANNOTATED = 0
STATE_S1 = 1
STATE_SYSTOLE = 2
STATE_S2 = 3
STATE_DIASTOLE = 4
STATE_NAMES = {
    STATE_UNANNOTATED: "Unannotated",
    STATE_S1: "S1",
    STATE_SYSTOLE: "Systole",
    STATE_S2: "S2",
    STATE_DIASTOLE: "Diastole"
}

# Label Definitions
# Module 1 Binary Target:
# 0 = Normal
# 1 = Murmur / Abnormal
# -1 = Excluded / Unknown
LABEL_NORMAL = 0
LABEL_MURMUR_ABNORMAL = 1
LABEL_UNKNOWN_EXCLUDED = -1

MURMUR_ABSENT = 0
MURMUR_PRESENT = 1
MURMUR_UNKNOWN = 2

MURMUR_LABEL_MAP = {
    "Absent": MURMUR_ABSENT,
    "Present": MURMUR_PRESENT,
    "Unknown": MURMUR_UNKNOWN
}

OUTCOME_NORMAL = 0
OUTCOME_ABNORMAL = 1

OUTCOME_LABEL_MAP = {
    "Normal": OUTCOME_NORMAL,
    "Abnormal": OUTCOME_ABNORMAL
}

# Auscultation Sites
AUSCULTATION_LOCATIONS = ["AV", "PV", "TV", "MV", "Phc"]

# Feature Extraction Parameters (Downstream modules)
N_MFCC = 20
N_MELS = 40
FRAME_LENGTH_MS = 25.0
FRAME_STEP_MS = 10.0
