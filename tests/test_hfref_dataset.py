"""
Unit tests for HFrEF dataset loader, subject indexing, and valve mapping.
"""
import pytest
from pathlib import Path
from src.hfref.config import (
    AUSCULTATION_CODE_MAP,
    REVERSE_CODE_MAP,
    ORDERED_HFREF_LOCATIONS
)
from src.hfref.dataset_loader import (
    HFrEFDatasetLoader,
    HFrEFSubjectRecord,
    HFrEFRecordingMeta,
    get_dataset_loader
)

def test_valve_mappings():
    """Verify standard mapping between official HFrEF site codes and LubDub valve codes."""
    assert AUSCULTATION_CODE_MAP["APEX"]["lubdub_code"] == "MV"
    assert AUSCULTATION_CODE_MAP["LLSB"]["lubdub_code"] == "TV"
    assert AUSCULTATION_CODE_MAP["LUSB"]["lubdub_code"] == "PV"
    assert AUSCULTATION_CODE_MAP["RUSB"]["lubdub_code"] == "AV"

    assert REVERSE_CODE_MAP["MV"] == "APEX"
    assert REVERSE_CODE_MAP["TV"] == "LLSB"
    assert REVERSE_CODE_MAP["PV"] == "LUSB"
    assert REVERSE_CODE_MAP["AV"] == "RUSB"

def test_dataset_loader_detection_graceful(tmp_path):
    """Verify dataset loader handles missing or unplaced dataset gracefully without crashing."""
    loader = HFrEFDatasetLoader(data_dir=tmp_path)
    assert not loader.is_dataset_present()
    summary = loader.get_dataset_summary()
    assert summary["is_present"] is False
    assert summary["status"] == "dataset_pending_placement"
    assert summary["total_subjects"] == 0
    assert loader.list_subjects() == []
    assert loader.get_subject("nonexistent") is None

def test_subject_record_properties():
    """Verify HFrEFSubjectRecord property logic and location verification."""
    sub = HFrEFSubjectRecord(
        subject_id="SUBJ_001",
        fold=1,
        lvef=35.0,
        is_hfref=True,
        demographics={"Age": 58, "Sex": "M"}
    )
    
    assert sub.is_hfref is True
    assert not sub.has_all_pcg_locations
    assert sub.available_pcg_locations == []

    # Add mock recording metadata
    sub.pcg_recordings["RUSB"] = HFrEFRecordingMeta(
        filepath=Path("mock_rusb.wav"),
        filename="SUBJ_001_RUSB_PCG.wav",
        modality="PCG",
        site_code="RUSB",
        valve_code="AV",
        sample_rate=4000,
        duration_sec=30.0,
        num_samples=120000,
        file_size_bytes=240000
    )
    sub.pcg_recordings["LUSB"] = HFrEFRecordingMeta(
        filepath=Path("mock_lusb.wav"),
        filename="SUBJ_001_LUSB_PCG.wav",
        modality="PCG",
        site_code="LUSB",
        valve_code="PV",
        sample_rate=4000,
        duration_sec=30.0,
        num_samples=120000,
        file_size_bytes=240000
    )
    sub.pcg_recordings["LLSB"] = HFrEFRecordingMeta(
        filepath=Path("mock_llsb.wav"),
        filename="SUBJ_001_LLSB_PCG.wav",
        modality="PCG",
        site_code="LLSB",
        valve_code="TV",
        sample_rate=4000,
        duration_sec=30.0,
        num_samples=120000,
        file_size_bytes=240000
    )
    sub.pcg_recordings["APEX"] = HFrEFRecordingMeta(
        filepath=Path("mock_apex.wav"),
        filename="SUBJ_001_APEX_PCG.wav",
        modality="PCG",
        site_code="APEX",
        valve_code="MV",
        sample_rate=4000,
        duration_sec=30.0,
        num_samples=120000,
        file_size_bytes=240000
    )

    assert sub.has_all_pcg_locations
    assert len(sub.available_pcg_locations) == 4
    
    summary_dict = sub.to_summary_dict()
    assert summary_dict["subject_id"] == "SUBJ_001"
    assert summary_dict["lvef"] == 35.0
    assert summary_dict["is_hfref"] is True
    assert "AV" in summary_dict["available_pcg_locations"]
    assert "MV" in summary_dict["available_pcg_locations"]
