"""
Command-line runner for Module 2: Feature Extraction and Multi-Location Representation.
"""
import os
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.config import (
    FeatureConfig,
    MODULE2_H5_PATH,
    MODULE2_CHECKPOINT_H5_PATH,
    MODULE2_SUMMARY_PATH,
    MODULE2_ERRORS_PATH,
    MODULE2_REPORTS_DIR
)
from src.features.pipeline import Module2Pipeline
from src.utils.config import MODULE1_H5_PATH

def main():
    parser = argparse.ArgumentParser(description="Module 2: Heart Sound Feature Extraction & Multi-Location Representation")
    parser.add_argument("--input", type=str, default=str(MODULE1_H5_PATH), help="Path to input Module 1 processed HDF5")
    parser.add_argument("--output", type=str, default=str(MODULE2_H5_PATH), help="Path to output Module 2 features HDF5")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to temporary checkpoint HDF5")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for feature extraction")
    parser.add_argument("--n_mfcc", type=int, default=40, help="Number of MFCC coefficients")
    parser.add_argument("--n_mels", type=int, default=128, help="Number of Mel filterbanks")
    parser.add_argument("--no_deltas", action="store_true", help="Disable Delta and Delta-Delta calculation")
    parser.add_argument("--no_resume", action="store_true", help="Disable resumption and force fresh checkpoint")
    parser.add_argument("--flush_interval", type=int, default=1, help="Flush HDF5 every N batches (default 1)")
    parser.add_argument("--visualize", action="store_true", help="Generate example spectrogram and MFCC plots in reports/module2")
    
    args = parser.parse_args()

    config = FeatureConfig(
        n_mfcc=args.n_mfcc,
        n_mels=args.n_mels,
        include_delta=not args.no_deltas,
        include_delta_delta=not args.no_deltas,
        batch_size=args.batch_size,
        flush_interval_batches=args.flush_interval,
        resume=not args.no_resume
    )

    pipeline = Module2Pipeline(
        config=config,
        input_h5_path=args.input,
        output_h5_path=args.output,
        checkpoint_h5_path=args.checkpoint,
        summary_path=MODULE2_SUMMARY_PATH,
        errors_path=MODULE2_ERRORS_PATH
    )

    summary = pipeline.run()

    # Generate Visualizations if requested
    if args.visualize:
        try:
            from src.features.visualization import generate_module2_visualizations
            print("\nGenerating Module 2 visual reports in reports/module2/...", flush=True)
            generate_module2_visualizations(
                h5_path=args.output,
                output_dir=MODULE2_REPORTS_DIR
            )
        except Exception as e:
            print(f"Warning: Visualization generation skipped due to: {e}", flush=True)

    print("\n" + "=" * 70)
    print(" MODULE 2 EXECUTION SUMMARY")
    print("=" * 70)
    print(f"- Total Feature Samples: {summary['total_feature_samples']:,}")
    print(f"  * Train: {summary['split_feature_counts']['train']:,} samples ({summary['patient_split_counts']['train']} patients)")
    print(f"  * Val:   {summary['split_feature_counts']['validation']:,} samples ({summary['patient_split_counts']['validation']} patients)")
    print(f"  * Test:  {summary['split_feature_counts']['test']:,} samples ({summary['patient_split_counts']['test']} patients)")
    print(f"- MFCC Combined Tensor Shape: {summary['tensor_shapes']['mfcc_combined']}")
    print(f"- Mel-Spectrogram Tensor Shape: {summary['tensor_shapes']['mel_spectrogram']}")
    print(f"- Patient Multi-Location Tensor Shape: {summary['tensor_shapes']['patient_multi_location_tensor']}")
    print(f"- Location Mask Shape: {summary['tensor_shapes']['location_mask']}")
    print(f"- Output Features HDF5: {summary['artifacts']['hdf5_features']}")
    print(f"- Output HDF5 Size: {summary['artifacts']['hdf5_size_bytes'] / (1024*1024):.2f} MB")
    print("=" * 70)

if __name__ == "__main__":
    main()
