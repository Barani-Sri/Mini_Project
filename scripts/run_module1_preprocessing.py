"""
Command-line runner for Module 1: CirCor PCG Standardization & Windowing Pipeline.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.module1_preprocessing_pipeline import Module1Pipeline
from src.utils.config import (
    RAW_DATA_DIR,
    MODULE1_H5_PATH,
    MODULE1_SUMMARY_PATH,
    MODULE1_ERRORS_PATH,
    MODULE1_TARGET_SAMPLE_RATE,
    MODULE1_BANDPASS_LOW_HZ,
    MODULE1_BANDPASS_HIGH_HZ,
    MODULE1_BANDPASS_ORDER,
    MODULE1_WINDOW_SEC,
    MODULE1_HOP_SEC,
    MODULE1_NORM_METHOD,
    MODULE1_TARGET_PEAK
)

def main():
    parser = argparse.ArgumentParser(description="Module 1: CirCor PCG Data Preprocessing & Windowing")
    parser.add_argument("--data_dir", type=str, default=str(RAW_DATA_DIR), help="Path to raw CirCor directory")
    parser.add_argument("--output_h5", type=str, default=str(MODULE1_H5_PATH), help="Path to output .h5 file")
    parser.add_argument("--target_sr", type=int, default=MODULE1_TARGET_SAMPLE_RATE, help="Target sampling rate (Hz)")
    parser.add_argument("--bandpass_low", type=float, default=MODULE1_BANDPASS_LOW_HZ, help="Bandpass low cut (Hz)")
    parser.add_argument("--bandpass_high", type=float, default=MODULE1_BANDPASS_HIGH_HZ, help="Bandpass high cut (Hz)")
    parser.add_argument("--bandpass_order", type=int, default=MODULE1_BANDPASS_ORDER, help="Butterworth order")
    parser.add_argument("--notch_enabled", action="store_true", help="Enable 50Hz/60Hz notch filter")
    parser.add_argument("--notch_freq", type=float, default=50.0, help="Notch filter frequency (Hz)")
    parser.add_argument("--norm_method", type=str, default=MODULE1_NORM_METHOD, help="Normalization method (peak, zscore, minmax)")
    parser.add_argument("--target_peak", type=float, default=MODULE1_TARGET_PEAK, help="Target peak amplitude")
    parser.add_argument("--window_sec", type=float, default=MODULE1_WINDOW_SEC, help="Fixed window length in seconds")
    parser.add_argument("--hop_sec", type=float, default=MODULE1_HOP_SEC, help="Hop step in seconds")
    parser.add_argument("--no_augment", action="store_true", help="Disable training data augmentation")
    
    args = parser.parse_args()
    
    pipeline = Module1Pipeline(
        raw_data_dir=args.data_dir,
        output_h5_path=args.output_h5,
        summary_path=MODULE1_SUMMARY_PATH,
        errors_path=MODULE1_ERRORS_PATH,
        target_sr=args.target_sr,
        bandpass_low=args.bandpass_low,
        bandpass_high=args.bandpass_high,
        bandpass_order=args.bandpass_order,
        notch_enabled=args.notch_enabled,
        notch_freq=args.notch_freq,
        norm_method=args.norm_method,
        target_peak=args.target_peak,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        apply_train_augmentation=not args.no_augment
    )
    
    summary = pipeline.run()
    print("\nModule 1 Pipeline Summary:")
    print(f"- Total Patients Processed: {summary['total_patients']}")
    print(f"- Total Recordings Processed: {summary['successful_recordings']}")
    print(f"- Total Windows Generated: {summary['window_counts']['total_generated_windows']:,}")
    print(f"  * Train: {summary['window_counts']['train_windows']:,} windows ({summary['patient_splits']['train_patients']} patients)")
    print(f"  * Val:   {summary['window_counts']['validation_windows']:,} windows ({summary['patient_splits']['validation_patients']} patients)")
    print(f"  * Test:  {summary['window_counts']['test_windows']:,} windows ({summary['patient_splits']['test_patients']} patients)")
    print(f"- Final Waveform Shape: ({summary['dsp_configuration']['window_sample_length']},)")
    print(f"- Processed HDF5: {summary['artifacts']['hdf5_dataset']}")

if __name__ == "__main__":
    main()
