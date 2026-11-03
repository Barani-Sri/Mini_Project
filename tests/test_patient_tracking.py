"""
Unit and integration tests for Feature 3: Patient Heart-Sound Longitudinal Tracking.
Tests multi-visit storage, mathematical trend calculation, safety guardrails, patient isolation, and API endpoints.
"""
import pytest
from pathlib import Path
import tempfile
import json
from fastapi.testclient import TestClient

from src.services.patient_tracking_service import PatientTrackingService
from src.web.app import app

client = TestClient(app)


@pytest.fixture
def temp_tracking_service():
    """Creates a PatientTrackingService instance backed by a temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        temp_path = Path(tf.name)
    
    # Remove file so service initializes it
    if temp_path.exists():
        temp_path.unlink()
        
    service = PatientTrackingService(storage_path=temp_path)
    yield service
    
    if temp_path.exists():
        temp_path.unlink()


def test_tracking_single_visit_baseline(temp_tracking_service):
    """Verifies that a single visit produces baseline status without misleading trend slope."""
    history = temp_tracking_service.add_visit(
        patient_id="PT_SINGLE_001",
        patient_name="Single Visit Patient",
        murmur_prediction="Absent",
        murmur_prediction_class=0,
        murmur_probability=0.18,
        murmur_probability_percent=18.0,
        clinical_outcome="Normal",
        clinical_outcome_class=0,
        location_attention={"AV": {"percent": 25.0, "is_active": True}}
    )
    
    assert history["patient_id"] == "PT_SINGLE_001"
    assert history["total_visits"] == 1
    assert history["has_multiple_visits"] is False
    
    trend = history["trend"]
    assert trend["status"] == "single_visit"
    assert trend["direction"] == "baseline"
    assert trend["latest_percent"] == 18.0
    assert trend["previous_percent"] is None
    assert trend["change_previous_pp"] is None
    assert "Only one analysis is currently available" in trend["interpretation"]


def test_tracking_multiple_visits_increasing_trend(temp_tracking_service):
    """Verifies trend calculation across multiple visits with increasing AI measurements."""
    pid = "PT_MULTI_001"
    
    # Visit 1
    temp_tracking_service.add_visit(
        patient_id=pid,
        patient_name="Multi Visit Patient",
        murmur_prediction="Absent",
        murmur_prediction_class=0,
        murmur_probability=0.35,
        murmur_probability_percent=35.0,
        clinical_outcome="Normal",
        clinical_outcome_class=0
    )
    
    # Visit 2
    temp_tracking_service.add_visit(
        patient_id=pid,
        patient_name="Multi Visit Patient",
        murmur_prediction="Present",
        murmur_prediction_class=1,
        murmur_probability=0.52,
        murmur_probability_percent=52.0,
        clinical_outcome="Abnormal",
        clinical_outcome_class=1
    )
    
    # Visit 3
    history = temp_tracking_service.add_visit(
        patient_id=pid,
        patient_name="Multi Visit Patient",
        murmur_prediction="Present",
        murmur_prediction_class=1,
        murmur_probability=0.68,
        murmur_probability_percent=68.0,
        clinical_outcome="Abnormal",
        clinical_outcome_class=1
    )
    
    assert history["total_visits"] == 3
    assert history["has_multiple_visits"] is True
    
    trend = history["trend"]
    assert trend["status"] == "multi_visit"
    assert trend["direction"] == "increasing"
    assert trend["direction_symbol"] == "↑"
    assert trend["latest_percent"] == 68.0
    assert trend["previous_percent"] == 52.0
    assert trend["change_previous_pp"] == 16.0
    assert trend["overall_change_pp"] == 33.0
    
    # Verify safety: no claim of disease worsening
    interp = trend["interpretation"]
    assert "AI acoustic indicator has increased" in interp
    assert "reflect acoustic pattern variations and do not indicate clinical disease progression" in interp
    assert "worsening" not in interp.lower()
    assert "deteriorating" not in interp.lower()


def test_tracking_decreasing_and_stable_trends(temp_tracking_service):
    """Verifies decreasing and relatively stable trajectory classifications."""
    pid_dec = "PT_DEC_001"
    temp_tracking_service.add_visit(
        patient_id=pid_dec, patient_name="Dec Patient",
        murmur_prediction="Present", murmur_prediction_class=1,
        murmur_probability=0.75, murmur_probability_percent=75.0,
        clinical_outcome="Abnormal", clinical_outcome_class=1
    )
    hist_dec = temp_tracking_service.add_visit(
        patient_id=pid_dec, patient_name="Dec Patient",
        murmur_prediction="Present", murmur_prediction_class=1,
        murmur_probability=0.61, murmur_probability_percent=61.0,
        clinical_outcome="Abnormal", clinical_outcome_class=1
    )
    assert hist_dec["trend"]["direction"] == "decreasing"
    assert hist_dec["trend"]["change_previous_pp"] == -14.0
    assert hist_dec["trend"]["direction_symbol"] == "↓"
    assert "improving" not in hist_dec["trend"]["interpretation"].lower()
    
    pid_stab = "PT_STAB_001"
    temp_tracking_service.add_visit(
        patient_id=pid_stab, patient_name="Stab Patient",
        murmur_prediction="Present", murmur_prediction_class=1,
        murmur_probability=0.552, murmur_probability_percent=55.2,
        clinical_outcome="Normal", clinical_outcome_class=0
    )
    hist_stab = temp_tracking_service.add_visit(
        patient_id=pid_stab, patient_name="Stab Patient",
        murmur_prediction="Present", murmur_prediction_class=1,
        murmur_probability=0.554, murmur_probability_percent=55.4,
        clinical_outcome="Normal", clinical_outcome_class=0
    )
    assert hist_stab["trend"]["direction"] == "stable"
    assert hist_stab["trend"]["direction_symbol"] == "→"


def test_tracking_patient_isolation(temp_tracking_service):
    """Verifies that multiple patients have strictly isolated histories."""
    temp_tracking_service.add_visit(
        patient_id="PATIENT_A", patient_name="Alice",
        murmur_prediction="Absent", murmur_prediction_class=0,
        murmur_probability=0.20, murmur_probability_percent=20.0,
        clinical_outcome="Normal", clinical_outcome_class=0
    )
    
    temp_tracking_service.add_visit(
        patient_id="PATIENT_B", patient_name="Bob",
        murmur_prediction="Present", murmur_prediction_class=1,
        murmur_probability=0.85, murmur_probability_percent=85.0,
        clinical_outcome="Abnormal", clinical_outcome_class=1
    )
    
    hist_a = temp_tracking_service.get_patient_history("PATIENT_A")
    hist_b = temp_tracking_service.get_patient_history("PATIENT_B")
    
    assert hist_a["total_visits"] == 1
    assert hist_a["visits"][0]["patient_name"] == "Alice"
    assert hist_a["visits"][0]["murmur_probability_percent"] == 20.0
    
    assert hist_b["total_visits"] == 1
    assert hist_b["visits"][0]["patient_name"] == "Bob"
    assert hist_b["visits"][0]["murmur_probability_percent"] == 85.0


def test_api_patient_history_and_record_endpoints():
    """Tests FastAPI endpoints for patient history retrieval and manual recording."""
    test_pid = "API_TEST_PT_99"
    
    # 1. Clear any existing
    client.delete(f"/api/patient/{test_pid}/history")
    
    # 2. Record Visit 1
    payload1 = {
        "patient_id": test_pid,
        "patient_name": "API Test Patient",
        "murmur_prediction": "Absent",
        "murmur_prediction_class": 0,
        "murmur_probability": 0.22,
        "murmur_probability_percent": 22.0,
        "clinical_outcome": "Normal",
        "clinical_outcome_class": 0,
        "location_attention": {"AV": {"percent": 25.0, "is_active": True}}
    }
    res1 = client.post(f"/api/patient/{test_pid}/record", json=payload1)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["total_visits"] == 1
    assert data1["trend"]["status"] == "single_visit"
    
    # 3. Record Visit 2
    payload2 = {
        "patient_id": test_pid,
        "patient_name": "API Test Patient",
        "murmur_prediction": "Present",
        "murmur_prediction_class": 1,
        "murmur_probability": 0.44,
        "murmur_probability_percent": 44.0,
        "clinical_outcome": "Abnormal",
        "clinical_outcome_class": 1
    }
    res2 = client.post(f"/api/patient/{test_pid}/record", json=payload2)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["total_visits"] == 2
    assert data2["trend"]["direction"] == "increasing"
    assert data2["trend"]["change_previous_pp"] == 22.0
    
    # 4. Get History via GET endpoint
    get_res = client.get(f"/api/patient/{test_pid}/history")
    assert get_res.status_code == 200
    get_data = get_res.json()
    assert get_data["total_visits"] == 2
    assert len(get_data["visits"]) == 2
    
    # 5. Clean up
    del_res = client.delete(f"/api/patient/{test_pid}/history")
    assert del_res.status_code == 200
