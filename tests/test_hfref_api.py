"""
Integration tests for HFrEF and Integrated Result FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from src.web.app import app

client = TestClient(app)

def test_api_hfref_status():
    """Verify GET /api/hfref/status returns valid schema."""
    response = client.get("/api/hfref/status")
    assert response.status_code == 200
    data = response.json()
    assert "dataset_name" in data
    assert "is_present" in data
    assert "status" in data
    assert "supported_modalities" in data or "sampling_rates" in data

def test_api_hfref_subjects():
    """Verify GET /api/hfref/subjects returns list."""
    response = client.get("/api/hfref/subjects")
    assert response.status_code == 200
    data = response.json()
    assert "is_present" in data
    assert "subjects" in data
    assert isinstance(data["subjects"], list)

def test_api_hfref_demo():
    """Verify POST /api/hfref/demo executes without crashing."""
    response = client.post("/api/hfref/demo")
    assert response.status_code in [200, 404]
    data = response.json()
    assert "status" in data or "prediction" in data

def test_api_integrated_result():
    """Verify GET /api/integrated/result returns consolidated payload with both systems."""
    response = client.get("/api/integrated/result")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "existing_lubdub_result" in data
    assert "new_hfref_ecg_pcg_result" in data
    assert "specialist_recommendation" in data
    assert "architecture_summary" in data
    
    # Verify existing LubDub data is intact in the consolidated response
    lubdub = data["existing_lubdub_result"]
    assert "murmur_detection" in lubdub
    assert "clinical_outcome" in lubdub
    assert "location_attention" in lubdub
    assert "acoustic_similarity" in lubdub

    # Verify recommendation structure
    rec = data["specialist_recommendation"]
    assert "suggested_specialist" in rec
    assert "recommendation_text" in rec
    assert "disclaimer" in rec

def test_api_integrated_recommendation_post():
    """Verify POST /api/integrated/recommendation dynamically computes specialist referral."""
    payload = {
        "hfref_result": {
            "prediction": "HFrEF",
            "prediction_class": 1,
            "hfref_probability": 0.88
        },
        "lubdub_result": {
            "murmur_detection": {
                "prediction": "Present",
                "prediction_class": 1
            }
        }
    }
    response = client.post("/api/integrated/recommendation", json=payload)
    assert response.status_code == 200
    rec = response.json()
    assert "Cardiologist" in rec["suggested_specialist"]
    assert rec["priority_class"] == "urgent"

def test_api_integrated_report():
    """Verify POST /api/integrated/report produces valid PDF bytes."""
    payload = {
        "existing_lubdub_result": {
            "patient_id": "40840",
            "patient_name": "Test Patient",
            "murmur_detection": {"prediction": "Present", "probability_percent": 52.8},
            "clinical_outcome": {"prediction": "Abnormal", "probability_percent": 51.8}
        },
        "new_hfref_ecg_pcg_result": {
            "prediction": "Non-HFrEF",
            "estimated_lvef_range": "LVEF > 40%",
            "hfref_probability_percent": 25.0,
            "ecg_analysis": {"heart_rate_bpm": 74.0}
        },
        "specialist_recommendation": {
            "suggested_specialist": "Cardiologist",
            "priority_level": "Cardiology Review",
            "recommendation_text": "Further cardiac evaluation is recommended."
        }
    }
    response = client.post("/api/integrated/report", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content[:4] == b"%PDF"

def test_api_integrated_report_dataset_pending():
    """Verify POST /api/integrated/report produces valid PDF bytes when HFrEF is pending/None."""
    payload = {
        "existing_lubdub_result": {
            "patient_id": "40840",
            "patient_name": "Test Patient",
            "murmur_detection": {"prediction": "Present", "probability_percent": 52.8},
            "clinical_outcome": {"prediction": "Abnormal", "probability_percent": 51.8}
        },
        "new_hfref_ecg_pcg_result": None,
        "specialist_recommendation": {
            "suggested_specialist": "Cardiologist / Valvular Specialist",
            "priority_level": "Cardiology Auscultation Review",
            "recommendation_text": "Based on the available acoustic analysis, further evaluation by a cardiologist is recommended. HFrEF analysis is pending because the required dataset/model is not currently available.",
            "disclaimer": "Research prototype — not a clinical diagnostic system. Findings must be evaluated by a certified medical practitioner."
        }
    }
    response = client.post("/api/integrated/report", json=payload)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content[:4] == b"%PDF"

