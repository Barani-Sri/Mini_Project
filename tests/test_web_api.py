"""
Unit and API integration tests for LubDub Murmur FastAPI application.
"""
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from src.web.app import app
from src.utils.config import RAW_DATA_DIR

client = TestClient(app)

def test_api_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["app_name"] == "LubDub Murmur"

def test_api_demo_inference_patient_40840():
    response = client.post("/api/demo")
    assert response.status_code == 200
    data = response.json()
    
    assert data["patient_id"] == "40840"
    assert data["is_demo"] is True
    assert data["tensor_shape"] == [1, 5, 120, 501]
    assert data["location_mask"] == [1.0, 1.0, 1.0, 1.0, 0.0]
    
    # Check predictions match verified manual test
    murmur = data["murmur_detection"]
    outcome = data["clinical_outcome"]
    
    assert murmur["prediction"] == "Present"
    assert murmur["prediction_class"] == 1
    assert abs(murmur["probability_percent"] - 52.75) < 0.1
    
    assert outcome["prediction"] == "Abnormal"
    assert outcome["prediction_class"] == 1
    assert abs(outcome["probability_percent"] - 51.83) < 0.1
    
    # Check location attention weights
    attn = data["location_attention"]
    assert "AV" in attn and "MV" in attn and "PV" in attn and "TV" in attn and "Phc" in attn
    assert abs(attn["AV"]["percent"] - 23.71) < 0.1
    assert abs(attn["MV"]["percent"] - 26.39) < 0.1
    assert abs(attn["PV"]["percent"] - 27.38) < 0.1
    assert abs(attn["TV"]["percent"] - 22.51) < 0.1
    assert attn["Phc"]["percent"] == 0.0

def test_api_analyze_with_files():
    training_data_dir = Path(RAW_DATA_DIR) / "training_data"
    av_path = training_data_dir / "40840_AV.wav"
    mv_path = training_data_dir / "40840_MV.wav"
    
    if not av_path.exists() or not mv_path.exists():
        pytest.skip("CirCor test audio files not found.")
        
    with open(av_path, "rb") as f_av, open(mv_path, "rb") as f_mv:
        files = {
            "av_file": ("40840_AV.wav", f_av, "audio/wav"),
            "mv_file": ("40840_MV.wav", f_mv, "audio/wav")
        }
        data = {
            "patient_name": "Test Subject A",
            "patient_id": "SUBJ-001"
        }
        response = client.post("/api/analyze", data=data, files=files)
        
    assert response.status_code == 200
    res = response.json()
    assert res["patient_name"] == "Test Subject A"
    assert res["patient_id"] == "SUBJ-001"
    assert res["available_locations"] == ["AV", "MV"]
    assert res["tensor_shape"] == [1, 5, 120, 501]
    assert res["location_mask"] == [1.0, 1.0, 0.0, 0.0, 0.0]
    assert "probability" in res["murmur_detection"]
    assert "probability" in res["clinical_outcome"]
    assert "ai_explanation" in res
    assert res["ai_explanation"]["status_tag"] in ["NORMAL", "MURMUR DETECTED"]

def test_api_demo_ai_explanation():
    response = client.post("/api/demo")
    assert response.status_code == 200
    data = response.json()
    assert "ai_explanation" in data
    expl = data["ai_explanation"]
    assert "status_title" in expl
    assert "what_this_means" in expl
    assert "confidence_percent" in expl
    assert "disclaimer" in expl
    assert "location_insight" in expl

def test_api_chat_flow():
    session_id = "test_api_session_001"
    
    # 1. Send normal result question
    payload = {
        "session_id": session_id,
        "message": "Why is my result normal?",
        "context": {
            "prediction": "NORMAL",
            "confidence": 0.82
        }
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert "NORMAL" in data["response"]
    assert "82.0%" in data["response"]
    assert len(data["suggested_questions"]) > 0
    
    # 2. Reset session
    reset_res = client.post("/api/chat/reset", json={"session_id": session_id})
    assert reset_res.status_code == 200
    assert reset_res.json()["status"] == "session_reset"

