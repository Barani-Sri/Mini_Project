"""
Integration Tests for Module 2 Pipeline, Checkpoint Management, Interruption, and Resumption.
"""
import tempfile
import os
from pathlib import Path
import pytest
import h5py
import numpy as np

from src.features.config import FeatureConfig
from src.features.pipeline import Module2Pipeline

@pytest.fixture
def mock_m1_dataset(tmp_path):
    """Creates a deterministic synthetic Module 1 dataset."""
    mock_m1_h5 = tmp_path / "module1_mock.h5"
    str_dtype = h5py.string_dtype(encoding="utf-8")
    np.random.seed(42)

    with h5py.File(mock_m1_h5, "w") as f:
        f.attrs["target_sample_rate"] = 2000
        
        # Train: 3 patients -> 6 windows
        grp_tr = f.create_group("train")
        grp_tr.create_dataset("audio", data=np.random.randn(6, 10000).astype(np.float32))
        grp_tr.create_dataset("labels", data=np.array([1, 1, 0, 0, 1, 1], dtype=np.int64))
        grp_tr.create_dataset("patient_ids", data=np.array(["P1", "P1", "P2", "P2", "P3", "P3"], dtype=object), dtype=str_dtype)
        grp_tr.create_dataset("recording_ids", data=np.array(["P1_AV", "P1_MV", "P2_PV", "P2_TV", "P3_AV", "P3_Phc"], dtype=object), dtype=str_dtype)
        grp_tr.create_dataset("locations", data=np.array(["AV", "MV", "PV", "TV", "AV", "Phc"], dtype=object), dtype=str_dtype)

        # Val: 1 patient -> 2 windows
        grp_va = f.create_group("validation")
        grp_va.create_dataset("audio", data=np.random.randn(2, 10000).astype(np.float32))
        grp_va.create_dataset("labels", data=np.array([0, 0], dtype=np.int64))
        grp_va.create_dataset("patient_ids", data=np.array(["P4", "P4"], dtype=object), dtype=str_dtype)
        grp_va.create_dataset("recording_ids", data=np.array(["P4_AV", "P4_TV"], dtype=object), dtype=str_dtype)
        grp_va.create_dataset("locations", data=np.array(["AV", "TV"], dtype=object), dtype=str_dtype)

        # Test: 1 patient -> 2 windows
        grp_te = f.create_group("test")
        grp_te.create_dataset("audio", data=np.random.randn(2, 10000).astype(np.float32))
        grp_te.create_dataset("labels", data=np.array([1, 1], dtype=np.int64))
        grp_te.create_dataset("patient_ids", data=np.array(["P5", "P5"], dtype=object), dtype=str_dtype)
        grp_te.create_dataset("recording_ids", data=np.array(["P5_MV", "P5_PV"], dtype=object), dtype=str_dtype)
        grp_te.create_dataset("locations", data=np.array(["MV", "PV"], dtype=object), dtype=str_dtype)

    return mock_m1_h5

def test_end_to_end_module2_pipeline(tmp_path, mock_m1_dataset):
    out_m2_h5 = tmp_path / "module2_features.h5"
    out_sum_json = tmp_path / "module2_summary.json"
    out_err_csv = tmp_path / "module2_errors.csv"

    config = FeatureConfig(batch_size=2)
    pipeline = Module2Pipeline(
        config=config,
        input_h5_path=mock_m1_dataset,
        output_h5_path=out_m2_h5,
        summary_path=out_sum_json,
        errors_path=out_err_csv
    )

    summary = pipeline.run()

    # Assert Final Output Files Exist and Checkpoint is cleaned up
    assert out_m2_h5.exists()
    assert out_sum_json.exists()
    assert out_err_csv.exists()
    assert not pipeline.checkpoint_h5_path.exists()
    assert not pipeline.checkpoint_meta_path.exists()
    assert summary["total_feature_samples"] == 10
    assert summary["errors"]["failed_extractions"] == 0

    # Inspect Output HDF5
    with h5py.File(out_m2_h5, "r") as f:
        for s in ["train", "validation", "test"]:
            assert s in f
            grp = f[s]
            assert grp["mfcc"].shape[1:] == (40, 501)
            assert grp["mfcc_combined"].shape[1:] == (120, 501)
            assert grp["mel_spectrogram"].shape[1:] == (128, 501)

        # Patient Group
        assert "patients" in f
        assert "train" in f["patients"]
        pat_train = f["patients/train"]
        assert pat_train["patient_features"].shape == (3, 5, 120, 501)
        assert pat_train["location_masks"].shape == (3, 5)
        assert pat_train["patient_labels"].shape == (3,)

def test_interruption_and_resumption(tmp_path, mock_m1_dataset):
    out_m2_h5 = tmp_path / "module2_features_resumed.h5"
    chk_m2_h5 = tmp_path / "module2_features_resumed.checkpoint.h5"
    out_sum_json = tmp_path / "module2_summary_resumed.json"
    out_err_csv = tmp_path / "module2_errors_resumed.csv"

    config = FeatureConfig(batch_size=2, flush_interval_batches=1, resume=True)
    pipeline = Module2Pipeline(
        config=config,
        input_h5_path=mock_m1_dataset,
        output_h5_path=out_m2_h5,
        checkpoint_h5_path=chk_m2_h5,
        summary_path=out_sum_json,
        errors_path=out_err_csv
    )

    # 1. Simulate an interruption: Initialize checkpoint and process only 1 batch (2 samples of Train)
    with h5py.File(mock_m1_dataset, "r") as in_h5:
        out_h5 = pipeline._init_checkpoint_hdf5(in_h5, ["train", "validation", "test"])
        
        # Manually extract and write 1 batch to train group
        wave_b = in_h5["train"]["audio"][0:2]
        comb_b = [pipeline.mfcc_extractor(w) for w in wave_b]
        mel_b = [pipeline.mel_extractor(w) for w in wave_b]
        out_h5["train"]["mfcc_combined"][0:2] = np.array(comb_b, dtype=np.float32)
        out_h5["train"]["mel_spectrogram"][0:2] = np.array(mel_b, dtype=np.float32)
        out_h5["train"]["patient_ids"][0:2] = in_h5["train"]["patient_ids"][0:2]
        out_h5["train"]["locations"][0:2] = in_h5["train"]["locations"][0:2]
        out_h5["train"]["labels"][0:2] = in_h5["train"]["labels"][0:2]
        out_h5["train"]["quality_flags"][0:2] = ["GOOD", "GOOD"]
        out_h5["train"].attrs["processed_samples"] = 2
        out_h5["train"].attrs["completed"] = False
        out_h5.flush()
        out_h5.close()

    # Verify checkpoint exists and is readable while final output does not exist yet
    assert chk_m2_h5.exists()
    assert not out_m2_h5.exists()
    with h5py.File(chk_m2_h5, "r") as f:
        assert f["train"].attrs["processed_samples"] == 2
        assert not bool(f["train"].attrs["completed"])

    # 2. Run pipeline in resume mode
    summary = pipeline.run()

    # Verify pipeline resumed, completed, and promoted to final output
    assert out_m2_h5.exists()
    assert not chk_m2_h5.exists()
    assert summary["total_feature_samples"] == 10

    with h5py.File(out_m2_h5, "r") as f:
        assert f["train"]["mfcc_combined"].shape == (6, 120, 501)
        assert f["validation"]["mfcc_combined"].shape == (2, 120, 501)
        assert f["test"]["mfcc_combined"].shape == (2, 120, 501)
        assert f["patients/train"]["patient_features"].shape == (3, 5, 120, 501)

def test_resumed_output_matches_uninterrupted_output(tmp_path, mock_m1_dataset):
    """Verifies that an interrupted and resumed run produces identical numerical results to an uninterrupted run."""
    # 1. Uninterrupted run
    out_uninterrupted = tmp_path / "out_uninterrupted.h5"
    p1 = Module2Pipeline(
        config=FeatureConfig(batch_size=2, resume=False),
        input_h5_path=mock_m1_dataset,
        output_h5_path=out_uninterrupted,
        summary_path=tmp_path / "sum1.json",
        errors_path=tmp_path / "err1.csv"
    )
    p1.run()

    # 2. Interrupted then resumed run
    out_resumed = tmp_path / "out_resumed.h5"
    chk_resumed = tmp_path / "out_resumed.checkpoint.h5"
    p2 = Module2Pipeline(
        config=FeatureConfig(batch_size=2, resume=True),
        input_h5_path=mock_m1_dataset,
        output_h5_path=out_resumed,
        checkpoint_h5_path=chk_resumed,
        summary_path=tmp_path / "sum2.json",
        errors_path=tmp_path / "err2.csv"
    )

    # Interruption at 4 samples in Train
    with h5py.File(mock_m1_dataset, "r") as in_h5:
        out_h5 = p2._init_checkpoint_hdf5(in_h5, ["train", "validation", "test"])
        for start, end in [(0, 2), (2, 4)]:
            w_b = in_h5["train"]["audio"][start:end]
            mfc_b, del_b, d2_b, comb_b = [], [], [], []
            mel_b = []
            for w in w_b:
                m, d, d2, c = p2.mfcc_extractor.extract_all(w, standardize=True)
                mel = p2.mel_extractor.extract_log_mel(w, standardize=True)
                mfc_b.append(m); del_b.append(d); d2_b.append(d2); comb_b.append(c); mel_b.append(mel)
            out_h5["train"]["mfcc"][start:end] = np.array(mfc_b, dtype=np.float32)
            out_h5["train"]["delta_mfcc"][start:end] = np.array(del_b, dtype=np.float32)
            out_h5["train"]["delta_delta_mfcc"][start:end] = np.array(d2_b, dtype=np.float32)
            out_h5["train"]["mfcc_combined"][start:end] = np.array(comb_b, dtype=np.float32)
            out_h5["train"]["mel_spectrogram"][start:end] = np.array(mel_b, dtype=np.float32)
            out_h5["train"]["labels"][start:end] = in_h5["train"]["labels"][start:end]
            out_h5["train"]["patient_ids"][start:end] = in_h5["train"]["patient_ids"][start:end]
            out_h5["train"]["locations"][start:end] = in_h5["train"]["locations"][start:end]
            out_h5["train"]["recording_ids"][start:end] = in_h5["train"]["recording_ids"][start:end]
            out_h5["train"]["quality_flags"][start:end] = ["GOOD", "GOOD"]
            out_h5["train"]["quality_scores"][start:end] = [1.0, 1.0]
            out_h5["train"].attrs["processed_samples"] = end
        out_h5.flush()
        out_h5.close()

    # Resume to completion
    p2.run()

    # 3. Compare outputs
    with h5py.File(out_uninterrupted, "r") as f1, h5py.File(out_resumed, "r") as f2:
        for split in ["train", "validation", "test"]:
            for key in ["mfcc_combined", "mel_spectrogram", "labels"]:
                arr1 = f1[split][key][:]
                arr2 = f2[split][key][:]
                if np.issubdtype(arr1.dtype, np.floating):
                    assert np.allclose(arr1, arr2, atol=1e-5), f"Mismatch in {split}/{key}"
                else:
                    assert np.array_equal(arr1, arr2), f"Mismatch in {split}/{key}"

        # Patient tensors
        for split in ["train", "validation", "test"]:
            p_feat1 = f1[f"patients/{split}/patient_features"][:]
            p_feat2 = f2[f"patients/{split}/patient_features"][:]
            assert np.allclose(p_feat1, p_feat2, atol=1e-5)
            assert np.array_equal(f1[f"patients/{split}/location_masks"][:], f2[f"patients/{split}/location_masks"][:])

def test_corrupt_checkpoint_auto_reset(tmp_path, mock_m1_dataset):
    """Verifies that a damaged/corrupt checkpoint file is safely discarded and reset."""
    out_m2_h5 = tmp_path / "module2_features_corrupt_test.h5"
    chk_m2_h5 = tmp_path / "module2_features_corrupt_test.checkpoint.h5"
    
    # Write garbage corrupted data to checkpoint file
    chk_m2_h5.write_bytes(b"CORRUPT_HEADER_TRUNCATED_HDF5_DATA" * 50)
    assert chk_m2_h5.exists()

    pipeline = Module2Pipeline(
        config=FeatureConfig(batch_size=2, resume=True),
        input_h5_path=mock_m1_dataset,
        output_h5_path=out_m2_h5,
        checkpoint_h5_path=chk_m2_h5,
        summary_path=tmp_path / "sum_corr.json",
        errors_path=tmp_path / "err_corr.csv"
    )

    # Should detect corruption, discard corrupt checkpoint, and run fresh to success
    summary = pipeline.run()
    assert out_m2_h5.exists()
    assert summary["total_feature_samples"] == 10
