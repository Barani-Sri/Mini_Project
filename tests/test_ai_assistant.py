"""
Unit tests for LubDub AI — Result Explanation Engine and Ask LubDub AI Chatbot Service.
"""
import pytest
from src.services.result_explanation_service import generate_result_explanation
from src.services.chatbot_service import LubDubChatbotService, get_chatbot_service


def test_result_explainer_normal():
    explanation = generate_result_explanation(
        murmur_pred="Absent",
        murmur_prob=0.18,  # -> 82% confidence normal
        outcome_pred="Normal",
        outcome_prob=0.15,
        available_locations=["AV", "MV", "PV", "TV"],
        location_attention={
            "AV": {"percent": 25.0, "is_active": True},
            "MV": {"percent": 30.0, "is_active": True},
            "PV": {"percent": 25.0, "is_active": True},
            "TV": {"percent": 20.0, "is_active": True},
            "Phc": {"percent": 0.0, "is_active": False}
        }
    )
    assert explanation["status_tag"] == "NORMAL"
    assert explanation["confidence_percent"] == 82.0
    assert "not detect acoustic patterns" in explanation["what_this_means"]
    assert "screening tool" in explanation["disclaimer"].lower()
    assert "Mitral Valve (MV)" in explanation["location_insight"]


def test_result_explainer_murmur():
    explanation = generate_result_explanation(
        murmur_pred="Present",
        murmur_prob=0.78,
        outcome_pred="Abnormal",
        outcome_prob=0.75,
        available_locations=["AV", "MV"],
        location_attention={
            "AV": {"percent": 60.0, "is_active": True},
            "MV": {"percent": 40.0, "is_active": True},
            "PV": {"percent": 0.0, "is_active": False},
            "TV": {"percent": 0.0, "is_active": False},
            "Phc": {"percent": 0.0, "is_active": False}
        }
    )
    assert explanation["status_tag"] == "MURMUR DETECTED"
    assert explanation["confidence_percent"] == 78.0
    assert "turbulent blood flow" in explanation["what_this_means"]
    assert "Aortic Valve (AV)" in explanation["location_insight"]
    assert "echocardiogram" in explanation["next_steps"].lower()


def test_chatbot_case_1_normal_result():
    bot = LubDubChatbotService()
    session_id = "test_sess_normal"
    ctx = {"prediction": "NORMAL", "confidence": 0.82}
    
    res = bot.chat(session_id, "Why is my result normal?", context=ctx)
    assert res["session_id"] == session_id
    assert "NORMAL" in res["response"]
    assert "82.0%" in res["response"]
    assert "screening assessment" in res["response"] or "screening" in res["response"]


def test_chatbot_case_2_murmur_result():
    bot = LubDubChatbotService()
    session_id = "test_sess_murmur"
    ctx = {"prediction": "MURMUR", "confidence": 0.78}
    
    res = bot.chat(session_id, "What does my result mean?", context=ctx)
    assert "MURMUR DETECTED" in res["response"]
    assert "78.0%" in res["response"]
    assert "turbulent blood flow" in res["response"] or "acoustic vibrations" in res["response"]


def test_chatbot_case_3_confidence_meaning():
    bot = LubDubChatbotService()
    session_id = "test_sess_conf"
    ctx = {"prediction": "NORMAL", "confidence": 0.82}
    
    res = bot.chat(session_id, "What does confidence mean?", context=ctx)
    assert "statistical certainty" in res["response"]
    assert "NOT" in res["response"]
    assert "medical probability" in res["response"]


def test_chatbot_case_4_next_steps():
    bot = LubDubChatbotService()
    session_id = "test_sess_steps"
    ctx = {"prediction": "MURMUR", "confidence": 0.78}
    
    res = bot.chat(session_id, "What should I do next?", context=ctx)
    assert "Consult a Doctor" in res["response"]
    assert "echocardiogram" in res["response"].lower()


def test_chatbot_case_5_diagnosis_guardrail():
    bot = LubDubChatbotService()
    session_id = "test_sess_diag"
    ctx = {"prediction": "NORMAL", "confidence": 0.82}
    
    res = bot.chat(session_id, "Can this result diagnose my heart condition?", context=ctx)
    assert "screening tool, not a diagnostic" in res["response"]
    assert "cannot diagnose" in res["response"].lower()
    assert "physician" in res["response"].lower() or "cardiologist" in res["response"].lower()


def test_chatbot_emergency_red_flag():
    bot = LubDubChatbotService()
    session_id = "test_sess_emergency"
    ctx = {"prediction": "MURMUR", "confidence": 0.78}
    
    res = bot.chat(session_id, "I have severe chest pain and feel like I am suffocating", context=ctx)
    assert res["safety_flag"] is True
    assert "URGENT MEDICAL NOTICE" in res["response"]
    assert "911" in res["response"] or "emergency" in res["response"].lower()


def test_chatbot_medication_guardrail():
    bot = LubDubChatbotService()
    session_id = "test_sess_meds"
    ctx = {"prediction": "MURMUR", "confidence": 0.78}
    
    res = bot.chat(session_id, "What medicine or pills should I take for this murmur?", context=ctx)
    assert "cannot recommend, prescribe, or alter any medications" in res["response"]


def test_chatbot_multi_turn_memory_and_reset():
    bot = LubDubChatbotService()
    session_id = "multi_turn_session"
    ctx1 = {"prediction": "NORMAL", "confidence": 0.82}
    
    # First turn
    bot.chat(session_id, "Why is my result normal?", context=ctx1)
    
    # Second turn
    res2 = bot.chat(session_id, "What about my confidence?")
    assert "82.0%" in res2["response"]
    
    # Check history count
    sess = bot.get_or_create_session(session_id)
    assert len(sess["history"]) == 4  # 2 user + 2 assistant
    
    # Reset session for new patient
    ctx2 = {"prediction": "MURMUR", "confidence": 0.78}
    bot.reset_session(session_id, new_context=ctx2)
    sess_reset = bot.get_or_create_session(session_id)
    assert len(sess_reset["history"]) == 0
    assert sess_reset["context"]["prediction"] == "MURMUR"
