"""
Unit and Integration Tests for PDF Patient Report Generation Service and API.
"""
import pytest
from fastapi.testclient import TestClient

from src.web.app import app
from src.services.report_generation_service import (
    generate_patient_pdf_report,
    sanitize_filename
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_analysis_result():
    return {
        "patient_name": "CirCor Test Patient 40840",
        "patient_id": "40840",
        "is_demo": True,
        "available_locations": ["AV", "MV", "PV", "TV"],
        "demo_metadata": {
            "age": "Adolescent",
            "sex": "Female",
            "height": "147.0 cm",
            "weight": "36.1 kg"
        },
        "murmur_detection": {
            "prediction": "Present",
            "prediction_class": 1,
            "probability": 0.5275,
            "probability_percent": 52.75,
            "threshold": 0.5,
            "status_label": "Murmur Detected"
        },
        "clinical_outcome": {
            "prediction": "Abnormal",
            "prediction_class": 1,
            "probability": 0.6810,
            "probability_percent": 68.10,
            "threshold": 0.5,
            "status_label": "Abnormal Clinical Finding"
        },
        "location_attention": {
            "AV": {"weight": 0.3120, "percent": 31.20, "is_active": True},
            "MV": {"weight": 0.3480, "percent": 34.80, "is_active": True},
            "PV": {"weight": 0.1890, "percent": 18.90, "is_active": True},
            "TV": {"weight": 0.1510, "percent": 15.10, "is_active": True},
            "Phc": {"weight": 0.0, "percent": 0.0, "is_active": False}
        },
        "recordings_info": {
            "AV": {"filename": "40840_AV.wav", "duration_sec": 15.0, "sample_rate": 2000, "available": True},
            "MV": {"filename": "40840_MV.wav", "duration_sec": 15.0, "sample_rate": 2000, "available": True},
            "PV": {"filename": "40840_PV.wav", "duration_sec": 15.0, "sample_rate": 2000, "available": True},
            "TV": {"filename": "40840_TV.wav", "duration_sec": 15.0, "sample_rate": 2000, "available": True},
            "Phc": {"filename": None, "duration_sec": 0.0, "sample_rate": 0, "available": False}
        },
        "acoustic_similarity": {
            "fingerprint": {
                "id": "HS-C68EB9E0",
                "dimension": 1264,
                "status": "Generated ✓"
            },
            "matches": [
                {
                    "rank": 1,
                    "reference_id": "REF-13918",
                    "similarity_percent": 89.4,
                    "location": "Multi-Location",
                    "murmur_status": "Murmur Present"
                },
                {
                    "rank": 2,
                    "reference_id": "REF-29378",
                    "similarity_percent": 85.1,
                    "location": "Multi-Location",
                    "murmur_status": "Murmur Present"
                }
            ]
        },
        "patient_tracking": {
            "visits": [
                {
                    "visit_id": "V-1",
                    "murmur_probability_percent": 52.75,
                    "clinical_outcome": "Abnormal"
                }
            ]
        },
        "ai_explanation": {
            "headline": "AI Diagnostic Finding: Murmur Present (52.75% confidence)",
            "bullets": [
                "Mitral Valve (MV) showed highest relative attention weighting (34.8%).",
                "Phonocardiographic features indicate high-frequency acoustic turbulence."
            ]
        }
    }


def test_sanitize_filename():
    assert sanitize_filename("40840") == "40840"
    assert sanitize_filename("Patient John Doe #123") == "Patient_John_Doe_123"
    assert sanitize_filename("   ") == "Patient"
    assert sanitize_filename("TEST/PATIENT:99") == "TEST_PATIENT_99"


def test_generate_patient_pdf_report_valid(sample_analysis_result):
    pdf_bytes = generate_patient_pdf_report(sample_analysis_result)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF-")


def test_generate_patient_pdf_report_empty():
    pdf_bytes = generate_patient_pdf_report({})
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF-")


def test_api_report_endpoint_success(client, sample_analysis_result):
    res = client.post("/api/report", json=sample_analysis_result)
    assert res.status_code == 200
    assert res.headers.get("content-type") == "application/pdf"
    assert "attachment; filename=" in res.headers.get("content-disposition", "")
    assert res.content.startswith(b"%PDF-")


def test_api_report_endpoint_empty_body(client):
    res = client.post("/api/report", json={})
    assert res.status_code in [200, 400] # Handled cleanly without crash
