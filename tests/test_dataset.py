"""
Unit tests for CirCorDataset data loader and parser.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.dataset import CirCorDataset
from src.utils.config import RAW_DATA_DIR, SAMPLE_RATE

@pytest.fixture
def dataset():
    return CirCorDataset(data_dir=RAW_DATA_DIR)

def test_dataset_metadata_loading(dataset):
    patients_df = dataset.get_patients_df()
    assert isinstance(patients_df, pd.DataFrame)
    assert len(patients_df) == 942
    assert "Patient ID" in patients_df.columns
    assert "Murmur" in patients_df.columns
    assert "Outcome" in patients_df.columns
    assert "Murmur_Binary" in patients_df.columns
    assert "Outcome_Binary" in patients_df.columns

def test_recordings_dataframe(dataset):
    recs_df = dataset.get_recordings_df()
    assert isinstance(recs_df, pd.DataFrame)
    assert len(recs_df) > 3000
    assert "recording_id" in recs_df.columns
    assert "wav_path" in recs_df.columns
    assert "location" in recs_df.columns
    assert "patient_murmur_binary" in recs_df.columns

def test_audio_loading(dataset):
    recs_df = dataset.get_recordings_df()
    sample_rec = recs_df.iloc[0]
    
    audio, sr = dataset.load_audio(sample_rec["wav_path"])
    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert sr == SAMPLE_RATE
    assert len(audio) > 0
    assert not np.isnan(audio).any()
    assert np.max(np.abs(audio)) <= 1.05

def test_segmentation_loading(dataset):
    recs_df = dataset.get_recordings_df()
    # Find a recording with valid TSV
    sample_with_tsv = recs_df[recs_df["tsv_path"].notnull()].iloc[0]
    
    seg_df = dataset.load_segmentation(sample_with_tsv["tsv_path"])
    assert isinstance(seg_df, pd.DataFrame)
    assert len(seg_df) > 0
    assert set(["start", "end", "state", "state_name"]).issubset(seg_df.columns)
    assert seg_df["state"].isin([0, 1, 2, 3, 4]).all()
