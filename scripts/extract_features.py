import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import concurrent.futures

from src.data.dataset import CirCorDataset
from src.data.splitting import assign_patient_folds
from src.audio.preprocessing import preprocess_pcg_recording
from src.audio.features import extract_full_pcg_features
from src.utils.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_RATE

def process_single_recording(row_dict: Dict[str, Any], dataset: CirCorDataset) -> Dict[str, Any]:
    """
    Loads audio and segmentation for a single recording, preprocesses,
    and extracts the complete feature vector.
    """
    wav_path = row_dict["wav_path"]
    tsv_path = row_dict["tsv_path"]
    loc = row_dict["location"]
    
    # 1. Load Audio
    raw_audio, sr = dataset.load_audio(wav_path)
    
    # 2. Preprocess (Bandpass 25-400Hz, spike removal, zscore normalization)
    proc_audio, _ = preprocess_pcg_recording(
        raw_audio,
        sr=sr,
        target_sr=sr,
        apply_filter=True,
        apply_spike_removal=True,
        norm_method="zscore"
    )
    
    # 3. Load Segmentation if available
    seg_df = None
    if tsv_path and os.path.exists(tsv_path):
        try:
            seg_df = dataset.load_segmentation(tsv_path)
        except Exception:
            seg_df = None
            
    # 4. Extract Features
    features = extract_full_pcg_features(proc_audio, seg_df=seg_df, location=loc, sr=sr)
    
    # 5. Merge with metadata attributes
    res = dict(row_dict)
    res.update(features)
    return res

def run_feature_extraction(max_workers: int = 8):
    print("=" * 65)
    print(" CIRCOR DIGISCOPE BATCH FEATURE EXTRACTION PIPELINE")
    print("=" * 65)
    
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = CirCorDataset(data_dir=RAW_DATA_DIR)
    
    recs_df = dataset.get_recordings_df()
    print(f"Loaded {len(recs_df)} recordings from {dataset.get_patients_df().shape[0]} patients.")
    
    # Assign stratified 5-fold cross-validation IDs at patient level
    print("Assigning patient-stratified 5-fold cross-validation folds...")
    recs_df = assign_patient_folds(recs_df, n_splits=5, random_state=42, patient_id_col="patient_id")
    
    records = recs_df.to_dict("records")
    total_recs = len(records)
    print(f"Extracting acoustic & physiological features using {max_workers} worker threads...")
    
    start_time = time.time()
    extracted_rows = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_recording, r, dataset): i for i, r in enumerate(records)}
        
        done_cnt = 0
        last_print = start_time
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            extracted_rows.append(res)
            done_cnt += 1
            
            now = time.time()
            if now - last_print >= 2.0 or done_cnt == total_recs:
                pct = (done_cnt / total_recs) * 100
                speed = done_cnt / max(0.1, (now - start_time))
                print(f"  Processed: {done_cnt}/{total_recs} ({pct:.1f}%) - Speed: {speed:.1f} recs/sec", flush=True)
                last_print = now

    elapsed = time.time() - start_time
    print(f"Feature extraction completed in {elapsed:.1f}s.", flush=True)
    
    features_df = pd.DataFrame(extracted_rows)
    out_csv = PROCESSED_DATA_DIR / "circor_features.csv"
    features_df.to_csv(out_csv, index=False)
    print(f"Saved feature table: {out_csv} ({features_df.shape[0]} rows, {features_df.shape[1]} columns)")
    return features_df

if __name__ == "__main__":
    # Use 4-8 workers on Windows
    run_feature_extraction(max_workers=4)
