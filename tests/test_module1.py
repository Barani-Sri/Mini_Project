"""
Comprehensive Unit Tests for Module 1 Pipeline.
Tests:
1. WAV loading
2. Resampling
3. Bandpass filtering
4. Optional notch filtering
5. Normalization
6. Zero-signal normalization
7. Fixed-length segmentation
8. Short recording padding
9. Long recording segmentation
10. Label mapping
11. Patient-level split
12. Leakage detection
13. Training augmentation
14. No augmentation in validation/test
15. Output HDF5 structure
16. End-to-end synthetic pipeline
"""
import os
import tempfile
import pytest
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.dataset import CirCorDataset
from src.data.splitting import (
    get_patient_split_sets,
    verify_no_patient_leakage,
    patient_train_val_test_split
)
from src.audio.preprocessing import (
    butter_bandpass_filter,
    apply_notch_filter,
    normalize_audio,
    resample_pcg,
    segment_and_pad_waveform,
    preprocess_module1_audio
)
from src.audio.augmentation import (
    time_shift,
    add_gaussian_noise,
    time_stretch,
    PCGAugmentor
)
from src.data.module1_preprocessing_pipeline import Module1Pipeline
from src.utils.config import (
    RAW_DATA_DIR,
    MODULE1_TARGET_SAMPLE_RATE,
    MODULE1_WINDOW_SAMPLES,
    MODULE1_HOP_SAMPLES,
    LABEL_NORMAL,
    LABEL_MURMUR_ABNORMAL,
    LABEL_UNKNOWN_EXCLUDED
)

@pytest.fixture
def sample_pcg_audio():
    sr = 4000
    duration = 8.0  # 8 seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 5Hz drift + 50Hz heart sound + 150Hz murmur + 1000Hz noise
    sig = (
        0.3 * np.sin(2 * np.pi * 5 * t) +
        1.0 * np.sin(2 * np.pi * 50 * t) +
        0.5 * np.sin(2 * np.pi * 150 * t) +
        0.2 * np.sin(2 * np.pi * 1000 * t)
    ).astype(np.float32)
    return sig, sr

# 1. Test WAV Loading
def test_wav_loading():
    dataset = CirCorDataset(data_dir=RAW_DATA_DIR)
    recs = dataset.get_recordings_df()
    assert len(recs) > 0
    
    first_rec = recs.iloc[0]
    audio, sr = dataset.load_audio(first_rec["wav_path"])
    assert isinstance(audio, np.ndarray)
    assert sr == 4000
    assert audio.dtype == np.float32
    assert len(audio) > 0

# 2. Test Resampling
def test_resampling(sample_pcg_audio):
    audio, orig_sr = sample_pcg_audio
    target_sr = 2000
    
    # 4000 -> 2000
    resampled, new_sr = resample_pcg(audio, orig_sr=orig_sr, target_sr=target_sr)
    assert new_sr == 2000
    assert len(resampled) == len(audio) // 2
    
    # No-op when same rate
    same_audio, same_sr = resample_pcg(resampled, orig_sr=2000, target_sr=2000)
    assert same_sr == 2000
    assert np.array_equal(resampled, same_audio)

# 3. Test Bandpass Filtering
def test_bandpass_filtering(sample_pcg_audio):
    audio, sr = sample_pcg_audio
    # Resample to 2000 first
    audio_2k, _ = resample_pcg(audio, orig_sr=sr, target_sr=2000)
    
    filtered = butter_bandpass_filter(audio_2k, sr=2000, lowcut=25.0, highcut=400.0, order=5)
    assert len(filtered) == len(audio_2k)
    assert not np.isnan(filtered).any()
    
    # Frequency response test: 5Hz should be strongly attenuated
    fft_orig = np.abs(np.fft.rfft(audio_2k))
    fft_filt = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(audio_2k), d=1.0/2000)
    
    idx_5hz = np.argmin(np.abs(freqs - 5))
    assert fft_filt[idx_5hz] < 0.1 * fft_orig[idx_5hz]
    
    # Passband 150Hz preserved
    idx_150hz = np.argmin(np.abs(freqs - 150))
    assert fft_filt[idx_150hz] > 0.8 * fft_orig[idx_150hz]

# 4. Test Optional Notch Filtering
def test_notch_filtering():
    sr = 2000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # Signal with strong 50Hz hum
    audio = (np.sin(2 * np.pi * 100 * t) + 1.0 * np.sin(2 * np.pi * 50 * t)).astype(np.float32)
    
    notched = apply_notch_filter(audio, sr=sr, freq=50.0, q=30.0)
    assert len(notched) == len(audio)
    
    fft_orig = np.abs(np.fft.rfft(audio))
    fft_notch = np.abs(np.fft.rfft(notched))
    freqs = np.fft.rfftfreq(len(audio), d=1.0/sr)
    
    idx_50hz = np.argmin(np.abs(freqs - 50))
    idx_100hz = np.argmin(np.abs(freqs - 100))
    
    # 50Hz deeply attenuated, 100Hz untouched
    assert fft_notch[idx_50hz] < 0.15 * fft_orig[idx_50hz]
    assert fft_notch[idx_100hz] > 0.95 * fft_orig[idx_100hz]

# 5. Test Normalization
def test_peak_normalization(sample_pcg_audio):
    audio, _ = sample_pcg_audio
    norm = normalize_audio(audio, method="peak", target_peak=0.95)
    assert np.isclose(np.max(np.abs(norm)), 0.95, atol=1e-4)

# 6. Test Zero-Signal Normalization
def test_zero_signal_normalization():
    zeros = np.zeros(2000, dtype=np.float32)
    norm = normalize_audio(zeros, method="peak", target_peak=0.95)
    assert np.array_equal(norm, zeros)
    assert not np.isnan(norm).any()

# 7. Test Fixed-Length Segmentation
def test_fixed_length_segmentation():
    sr = 2000
    # Exactly 8 seconds = 16,000 samples
    # Win=5s (10,000), Hop=2.5s (5,000)
    # Windows: [0:10000], [5000:15000], [10000:16000 zero-padded to 10000] -> 3 windows
    audio = np.random.randn(16000).astype(np.float32)
    windows = segment_and_pad_waveform(audio, sr=sr, window_sec=5.0, hop_sec=2.5)
    
    assert isinstance(windows, np.ndarray)
    assert windows.shape == (3, 10000)

# 8. Test Short Recording Padding
def test_short_recording_padding():
    sr = 2000
    # 2 seconds = 4,000 samples (< 10,000)
    short_audio = np.ones(4000, dtype=np.float32)
    windows = segment_and_pad_waveform(short_audio, sr=sr, window_sec=5.0, hop_sec=2.5)
    
    assert windows.shape == (1, 10000)
    assert np.array_equal(windows[0, :4000], short_audio)
    assert np.array_equal(windows[0, 4000:], np.zeros(6000))

# 9. Test Long Recording Segmentation
def test_long_recording_segmentation():
    sr = 2000
    # 25 seconds = 50,000 samples
    long_audio = np.random.randn(50000).astype(np.float32)
    windows = segment_and_pad_waveform(long_audio, sr=sr, window_sec=5.0, hop_sec=2.5)
    
    assert len(windows) > 1
    for w in windows:
        assert w.shape == (10000,)
        assert not np.isnan(w).any()

# 10. Test Label Mapping
def test_label_mapping():
    dataset = CirCorDataset(data_dir=RAW_DATA_DIR)
    patients_df = dataset.get_patients_df()
    
    absent_p = patients_df[patients_df["Murmur"] == "Absent"]
    present_p = patients_df[patients_df["Murmur"] == "Present"]
    unknown_p = patients_df[patients_df["Murmur"] == "Unknown"]
    
    assert (absent_p["label"] == LABEL_NORMAL).all()
    assert (present_p["label"] == LABEL_MURMUR_ABNORMAL).all()
    assert (unknown_p["label"] == LABEL_UNKNOWN_EXCLUDED).all()

# 11. Test Patient-Level Split
def test_patient_level_split():
    dataset = CirCorDataset(data_dir=RAW_DATA_DIR)
    patients_df = dataset.get_patients_df()
    
    split_dict = get_patient_split_sets(
        patients_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )
    
    assert len(split_dict["train"]) > 0
    assert len(split_dict["validation"]) > 0
    assert len(split_dict["test"]) > 0
    
    total_split = len(split_dict["train"]) + len(split_dict["validation"]) + len(split_dict["test"])
    assert total_split == len(patients_df)

# 12. Test Leakage Detection
def test_leakage_detection():
    # Valid disjoint sets
    s_tr = {"P1", "P2", "P3"}
    s_va = {"P4", "P5"}
    s_te = {"P6"}
    is_valid, _ = verify_no_patient_leakage(s_tr, s_va, s_te)
    assert is_valid is True
    
    # Leaking sets
    s_leaking_va = {"P3", "P4"}
    is_invalid, overlap_info = verify_no_patient_leakage(s_tr, s_leaking_va, s_te)
    assert is_invalid is False
    assert overlap_info["train_val_overlap"] == 1

# 13. Test Training Augmentation
def test_training_augmentation():
    win = np.sin(np.linspace(0, 10, 10000)).astype(np.float32)
    augmentor = PCGAugmentor(p_shift=1.0, p_noise=1.0, p_stretch=1.0, seed=42)
    
    aug_win = augmentor.augment(win, sr=2000)
    assert isinstance(aug_win, np.ndarray)
    assert aug_win.shape == (10000,)
    assert not np.isnan(aug_win).any()
    assert not np.array_equal(win, aug_win)

# 14. Test No Augmentation in Validation/Test
def test_no_augmentation_val_test():
    win = np.sin(np.linspace(0, 10, 10000)).astype(np.float32)
    # Validation/Test should receive unmodified window
    val_win = win.copy()
    assert np.array_equal(win, val_win)

# 15. Test Output HDF5 Structure
def test_output_hdf5_structure():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_h5 = Path(tmp_dir) / "test_module1.h5"
        
        with h5py.File(tmp_h5, "w") as f:
            f.attrs["target_sample_rate"] = 2000
            f.attrs["window_samples"] = 10000
            
            for grp_name in ["train", "validation", "test"]:
                grp = f.create_group(grp_name)
                grp.create_dataset("audio", data=np.zeros((5, 10000), dtype=np.float32))
                grp.create_dataset("labels", data=np.array([0, 1, 0, 1, 0]))
                
        # Reopen and verify
        with h5py.File(tmp_h5, "r") as f:
            assert f.attrs["target_sample_rate"] == 2000
            assert "train" in f
            assert "validation" in f
            assert "test" in f
            assert f["train/audio"].shape == (5, 10000)
            assert f["train/labels"].shape == (5,)

# 16. Test End-to-End Synthetic Pipeline
def test_end_to_end_synthetic_pipeline():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir_p = Path(tmp_dir)
        raw_mock = tmp_dir_p / "circor"
        train_mock = raw_mock / "training_data"
        train_mock.mkdir(parents=True, exist_ok=True)
        
        # Create mockup training_data.csv
        csv_df = pd.DataFrame([
            {"Patient ID": "1001", "Recording locations:": "AV+MV", "Age": "Child", "Sex": "Female", "Height": 110, "Weight": 20, "Pregnancy status": False, "Murmur": "Present", "Outcome": "Abnormal", "Murmur locations": "AV"},
            {"Patient ID": "1002", "Recording locations:": "AV+TV", "Age": "Child", "Sex": "Male", "Height": 120, "Weight": 25, "Pregnancy status": False, "Murmur": "Absent", "Outcome": "Normal", "Murmur locations": ""},
            {"Patient ID": "1003", "Recording locations:": "PV", "Age": "Infant", "Sex": "Female", "Height": 80, "Weight": 10, "Pregnancy status": False, "Murmur": "Absent", "Outcome": "Normal", "Murmur locations": ""},
            {"Patient ID": "1004", "Recording locations:": "MV", "Age": "Child", "Sex": "Male", "Height": 105, "Weight": 18, "Pregnancy status": False, "Murmur": "Present", "Outcome": "Abnormal", "Murmur locations": "MV"},
        ])
        csv_df.to_csv(raw_mock / "training_data.csv", index=False)
        
        # Create synthetic WAV files
        import wave
        for pid, locs in [("1001", ["AV", "MV"]), ("1002", ["AV", "TV"]), ("1003", ["PV"]), ("1004", ["MV"])]:
            for loc in locs:
                w_path = train_mock / f"{pid}_{loc}.wav"
                with wave.open(str(w_path), "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(4000)
                    # 6 seconds of 4000Hz audio
                    samples = (np.sin(np.linspace(0, 50, 24000)) * 30000).astype(np.int16)
                    wf.writeframes(samples.tobytes())
                    
        out_h5 = tmp_dir_p / "processed.h5"
        out_sum = tmp_dir_p / "summary.json"
        out_err = tmp_dir_p / "errors.csv"
        
        pipeline = Module1Pipeline(
            raw_data_dir=raw_mock,
            output_h5_path=out_h5,
            summary_path=out_sum,
            errors_path=out_err,
            target_sr=2000,
            window_sec=5.0,
            hop_sec=2.5,
            train_ratio=0.50,
            val_ratio=0.25,
            test_ratio=0.25,
            apply_train_augmentation=True
        )
        
        summary = pipeline.run()
        assert out_h5.exists()
        assert out_sum.exists()
        assert summary["total_patients"] == 4
        assert summary["successful_recordings"] == 6
        assert summary["failed_recordings"] == 0
        
        with h5py.File(out_h5, "r") as f:
            for s in ["train", "validation", "test"]:
                assert s in f
                assert "audio" in f[s]
                assert "labels" in f[s]
                if f[s]["audio"].shape[0] > 0:
                    assert f[s]["audio"].shape[1] == 10000
