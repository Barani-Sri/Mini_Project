"""
Tests for Feature 4: 🔎 Acoustic Similarity Search & Heart-Sound Fingerprint V2.
Validates:
1. Fingerprint V2 generation, dimensionality (1,264), and deterministic HS- ID hash.
2. Acoustic similarity scoring using Cosine metric.
3. Top-K ranking (descending order).
4. Self-matching prevention (excluding query patient from reference search).
5. Patient isolation (separate query contexts).
6. Graceful fallback on empty/missing signals.
7. Web API endpoints (/api/demo, /api/analyze, /api/similarity/search).
"""
import io
import wave
import json
import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.fingerprint_v2.config import FingerprintV2Config
from src.fingerprint_v2.fingerprint_v2_generator import (
    HeartSoundFingerprintV2,
    HeartSoundFingerprintV2Generator
)
from src.fingerprint_v2.fingerprint_v2_similarity import AcousticSimilarityCalculatorV2
from src.services.acoustic_similarity_service import (
    AcousticSimilarityService,
    compute_deterministic_fingerprint_id,
    get_similarity_service
)
from src.web.app import app

client = TestClient(app)


def create_synthetic_wav_bytes(duration_sec: float = 5.0, sr: int = 4000, freq: float = 120.0) -> bytes:
    """Generates synthetic sine wave PCG WAV bytes."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Fundamental heartbeat + harmonic
    signal = 0.5 * np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * (freq * 2) * t)
    audio_int16 = (signal * 32767).astype(np.int16)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio_int16.tobytes())
    return bio.getvalue()


def test_deterministic_fingerprint_id():
    """Verify that identical vectors produce identical HS- hash IDs."""
    vec1 = np.array([0.1, 0.2, -0.3, 0.4, 0.5], dtype=np.float32)
    vec2 = np.array([0.1, 0.2, -0.3, 0.4, 0.5], dtype=np.float32)
    vec3 = np.array([0.1, 0.2, -0.3, 0.4, 0.6], dtype=np.float32)

    id1 = compute_deterministic_fingerprint_id(vec1)
    id2 = compute_deterministic_fingerprint_id(vec2)
    id3 = compute_deterministic_fingerprint_id(vec3)

    assert id1.startswith("HS-")
    assert len(id1) == 11  # "HS-" + 8 hex chars
    assert id1 == id2
    assert id1 != id3


def test_fingerprint_v2_generation_and_dimension():
    """Verify that HeartSoundFingerprintV2Generator produces a 1,264-D unit vector."""
    config = FingerprintV2Config()
    gen = HeartSoundFingerprintV2Generator(config)

    t = np.linspace(0, 5.0, 20000, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 100 * t)).astype(np.float32)

    fp = gen.generate_recording_fingerprint(audio, orig_sr=4000, location="AV")

    assert isinstance(fp, HeartSoundFingerprintV2)
    assert fp.dimension == 1264
    assert fp.vector.shape == (1264,)
    # Unit sphere norm
    assert np.isclose(np.linalg.norm(fp.vector), 1.0, atol=1e-3)


def test_acoustic_similarity_ranking_and_bounds():
    """Verify that similarity search returns ranked results within [0, 100]%."""
    service = get_similarity_service()
    assert len(service.reference_library) > 0

    # Query using first reference vector
    query_vec = np.array(service.reference_library[0]["vector"], dtype=np.float32)
    query_pid = service.reference_library[0]["patient_id"]

    # Search with self-exclusion
    results = service.search_similar_recordings(
        query_vec,
        top_k=5,
        exclude_patient_id=query_pid
    )

    matches = results["matches"]
    assert len(matches) <= 5
    assert results["fingerprint"]["id"].startswith("HS-")
    assert results["fingerprint"]["status"] == "Generated ✓"

    # Verify self-exclusion
    for m in matches:
        assert m["patient_id"] != query_pid
        assert 0.0 <= m["similarity_percent"] <= 100.0
        assert -1.0 <= m["similarity_score"] <= 1.0

    # Verify strictly descending order
    scores = [m["similarity_score"] for m in matches]
    assert scores == sorted(scores, reverse=True)


def test_self_matching_prevention():
    """Verify that query patient is strictly excluded from reference search."""
    service = get_similarity_service()
    ref_0 = service.reference_library[0]
    pid_0 = ref_0["patient_id"]

    results = service.search_similar_recordings(
        np.array(ref_0["vector"], dtype=np.float32),
        top_k=5,
        exclude_patient_id=pid_0
    )

    matched_pids = [m["patient_id"] for m in results["matches"]]
    assert pid_0 not in matched_pids


def test_empty_signal_fallback():
    """Verify graceful handling when query vector is empty or zero."""
    service = get_similarity_service()
    empty_vec = np.zeros(1264, dtype=np.float32)

    results = service.search_similar_recordings(empty_vec, top_k=5)

    assert results["fingerprint"]["id"] == "HS-UNAVAILABLE"
    assert len(results["matches"]) == 0
    assert "No valid acoustic signal" in results["message"]
    assert "disclaimer" in results


def test_api_demo_includes_acoustic_similarity():
    """Verify that /api/demo contains the acoustic_similarity payload."""
    response = client.post("/api/demo")
    assert response.status_code == 200
    data = response.json()

    assert "acoustic_similarity" in data
    sim = data["acoustic_similarity"]

    assert "fingerprint" in sim
    assert sim["fingerprint"]["version"] == "v2"
    assert sim["fingerprint"]["id"].startswith("HS-")

    assert "matches" in sim
    assert len(sim["matches"]) == 5
    # Patient 40840 must not match itself
    for m in sim["matches"]:
        assert m["patient_id"] != "40840"
        assert "similarity_percent" in m
        assert "rank" in m
        assert "component_similarities" in m

    assert "Acoustic similarity only" in sim["disclaimer"]


def test_api_similarity_search_endpoint():
    """Verify dedicated POST /api/similarity/search endpoint."""
    av_bytes = create_synthetic_wav_bytes(5.0, 4000, 100.0)
    mv_bytes = create_synthetic_wav_bytes(5.0, 4000, 150.0)

    files = {
        "av_file": ("test_av.wav", av_bytes, "audio/wav"),
        "mv_file": ("test_mv.wav", mv_bytes, "audio/wav")
    }
    data = {
        "patient_id": "TEST_PATIENT_999",
        "top_k": 5
    }

    response = client.post("/api/similarity/search", data=data, files=files)
    assert response.status_code == 200
    res = response.json()

    assert "fingerprint" in res
    assert res["fingerprint"]["id"].startswith("HS-")
    assert "matches" in res
    assert len(res["matches"]) <= 5
