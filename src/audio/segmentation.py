"""
PCG Fundamental Heart Sound Cycle Segmentation & Interval Analysis.
Parses 4-state annotations (S1, Systole, S2, Diastole) and extracts physiological timing parameters.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.config import (
    SAMPLE_RATE,
    STATE_S1,
    STATE_SYSTOLE,
    STATE_S2,
    STATE_DIASTOLE,
    STATE_NAMES
)

def parse_segmentation_file(tsv_path: str) -> pd.DataFrame:
    """
    Parses CirCor .tsv segmentation file containing [start_sec, end_sec, state_id].
    """
    df = pd.read_csv(tsv_path, sep=r"\s+", header=None, names=["start", "end", "state"])
    df["start"] = df["start"].astype(float)
    df["end"] = df["end"].astype(float)
    df["state"] = df["state"].astype(int)
    df["duration"] = df["end"] - df["start"]
    df["state_name"] = df["state"].map(STATE_NAMES)
    return df

def extract_cardiac_cycles(
    audio: np.ndarray,
    seg_df: pd.DataFrame,
    sr: int = SAMPLE_RATE
) -> List[Dict[str, Union[np.ndarray, Dict[str, float]]]]:
    """
    Extracts individual full cardiac cycles (S1 -> Systole -> S2 -> Diastole) from audio.
    Returns a list of cycle dictionaries, each containing:
    - 'cycle_audio': np.ndarray of full cycle
    - 's1_audio': np.ndarray of S1 segment
    - 'systole_audio': np.ndarray of Systolic interval
    - 's2_audio': np.ndarray of S2 segment
    - 'diastole_audio': np.ndarray of Diastolic interval
    - 'timing': dictionary of durations in seconds
    """
    cycles = []
    
    # Identify sequence of states: S1 (1) -> Systole (2) -> S2 (3) -> Diastole (4)
    rows = seg_df.to_dict("records")
    i = 0
    while i < len(rows) - 3:
        if (rows[i]["state"] == STATE_S1 and
            rows[i+1]["state"] == STATE_SYSTOLE and
            rows[i+2]["state"] == STATE_S2 and
            rows[i+3]["state"] == STATE_DIASTOLE):
            
            s1_info = rows[i]
            sys_info = rows[i+1]
            s2_info = rows[i+2]
            dia_info = rows[i+3]
            
            start_sec = s1_info["start"]
            end_sec = dia_info["end"]
            
            start_sample = max(0, int(start_sec * sr))
            end_sample = min(len(audio), int(end_sec * sr))
            
            s1_start = max(0, int(s1_info["start"] * sr))
            s1_end = min(len(audio), int(s1_info["end"] * sr))
            
            sys_start = max(0, int(sys_info["start"] * sr))
            sys_end = min(len(audio), int(sys_info["end"] * sr))
            
            s2_start = max(0, int(s2_info["start"] * sr))
            s2_end = min(len(audio), int(s2_info["end"] * sr))
            
            dia_start = max(0, int(dia_info["start"] * sr))
            dia_end = min(len(audio), int(dia_info["end"] * sr))
            
            cycle_audio = audio[start_sample:end_sample]
            
            if len(cycle_audio) > int(0.2 * sr): # Valid cycle duration > 200ms
                cycle_dict = {
                    "cycle_audio": cycle_audio,
                    "s1_audio": audio[s1_start:s1_end],
                    "systole_audio": audio[sys_start:sys_end],
                    "s2_audio": audio[s2_start:s2_end],
                    "diastole_audio": audio[dia_start:dia_end],
                    "timing": {
                        "cycle_duration": end_sec - start_sec,
                        "s1_duration": s1_info["duration"],
                        "systole_duration": sys_info["duration"],
                        "s2_duration": s2_info["duration"],
                        "diastole_duration": dia_info["duration"],
                        "sys_dia_ratio": sys_info["duration"] / (dia_info["duration"] + 1e-6),
                        "s1_s2_ratio": s1_info["duration"] / (s2_info["duration"] + 1e-6)
                    }
                }
                cycles.append(cycle_dict)
            i += 4
        else:
            i += 1
            
    return cycles

def compute_cycle_timing_features(seg_df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes statistical timing and physiological metrics from segmentation table.
    Returns:
    - Mean and std of S1, Systole, S2, Diastole durations
    - Mean and std of Systole/Diastole ratio
    - Estimated Heart Rate (BPM) and Heart Rate Variability (HRV std)
    """
    features: Dict[str, float] = {}
    
    if seg_df is None or len(seg_df) == 0:
        return {
            "hr_bpm": 75.0,
            "hrv_std": 0.0,
            "mean_s1_dur": 0.1,
            "std_s1_dur": 0.0,
            "mean_sys_dur": 0.15,
            "std_sys_dur": 0.0,
            "mean_s2_dur": 0.1,
            "std_s2_dur": 0.0,
            "mean_dia_dur": 0.25,
            "std_dia_dur": 0.0,
            "mean_sys_dia_ratio": 0.6,
            "std_sys_dia_ratio": 0.0
        }
        
    seg_df = seg_df.copy()
    if "duration" not in seg_df.columns and "start" in seg_df.columns and "end" in seg_df.columns:
        seg_df["duration"] = seg_df["end"] - seg_df["start"]
        
    for st_id, st_name in STATE_NAMES.items():
        sub = seg_df[seg_df["state"] == st_id]["duration"]
        if len(sub) > 0:
            features[f"mean_{st_name.lower()}_dur"] = float(sub.mean())
            features[f"std_{st_name.lower()}_dur"] = float(sub.std()) if len(sub) > 1 else 0.0
            features[f"median_{st_name.lower()}_dur"] = float(sub.median())
        else:
            features[f"mean_{st_name.lower()}_dur"] = 0.0
            features[f"std_{st_name.lower()}_dur"] = 0.0
            features[f"median_{st_name.lower()}_dur"] = 0.0
            
    # Calculate cycle intervals (from S1 start to next S1 start)
    s1_starts = seg_df[seg_df["state"] == STATE_S1]["start"].values
    if len(s1_starts) > 1:
        rr_intervals = np.diff(s1_starts)
        # Filter valid intervals (0.3s to 2.0s -> 30 to 200 BPM)
        valid_rr = rr_intervals[(rr_intervals >= 0.3) & (rr_intervals <= 2.0)]
        if len(valid_rr) > 0:
            mean_rr = np.mean(valid_rr)
            features["hr_bpm"] = float(60.0 / mean_rr)
            features["hrv_std"] = float(np.std(valid_rr))
        else:
            features["hr_bpm"] = 75.0
            features["hrv_std"] = 0.0
    else:
        features["hr_bpm"] = 75.0
        features["hrv_std"] = 0.0
        
    # Systole to Diastole ratios
    sys_durs = seg_df[seg_df["state"] == STATE_SYSTOLE]["duration"].values
    dia_durs = seg_df[seg_df["state"] == STATE_DIASTOLE]["duration"].values
    min_len = min(len(sys_durs), len(dia_durs))
    if min_len > 0:
        ratios = sys_durs[:min_len] / (dia_durs[:min_len] + 1e-6)
        features["mean_sys_dia_ratio"] = float(np.mean(ratios))
        features["std_sys_dia_ratio"] = float(np.std(ratios)) if min_len > 1 else 0.0
    else:
        features["mean_sys_dia_ratio"] = 0.6
        features["std_sys_dia_ratio"] = 0.0
        
    return features
