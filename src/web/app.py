"""
LubDub Murmur — FastAPI Web Application Backend.
Connects the web user interface to the existing Module 1, 2, and 3 ML inference pipeline.
"""
import io
import os
import sys
import wave
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import librosa
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Response, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_RATE,
    MODULE1_TARGET_SAMPLE_RATE,
    MODULE1_WINDOW_SEC,
    MODULE1_HOP_SEC
)
from src.audio.preprocessing import (
    preprocess_module1_audio,
    segment_and_pad_waveform
)
from src.features.config import FeatureConfig
from src.features.melspectrogram import MelSpectrogramExtractor
from src.features.mfcc import MFCCExtractor
from pydantic import BaseModel
from src.features.location import (
    MultiLocationPatientTensor,
    ORDERED_LOCATIONS,
    encode_location
)
from src.models.config import Module3Config, MODULE3_MODEL_PATH
from src.models.multi_location_net import MultiLocationHeartSoundNet
from src.services import (
    generate_result_explanation,
    get_chatbot_service,
    get_tracking_service,
    get_similarity_service,
    generate_patient_pdf_report,
    generate_integrated_pdf_report,
    sanitize_filename
)
from src.hfref import (
    get_dataset_loader,
    get_hfref_analyzer,
    get_specialist_recommender,
    extract_multimodal_features,
    HFREF_ECG_SAMPLE_RATE,
    HFREF_PCG_TARGET_SR,
    AUSCULTATION_CODE_MAP
)

class ChatRequest(BaseModel):
    session_id: str = "default_session"
    message: str
    context: Optional[Dict[str, Any]] = None

class ChatResetRequest(BaseModel):
    session_id: str = "default_session"
    context: Optional[Dict[str, Any]] = None

class PatientRecordRequest(BaseModel):
    patient_id: str
    patient_name: Optional[str] = "Anonymous Patient"
    murmur_prediction: str
    murmur_prediction_class: int
    murmur_probability: float
    murmur_probability_percent: float
    clinical_outcome: str
    clinical_outcome_class: int
    location_attention: Optional[Dict[str, Any]] = None
    is_demo: Optional[bool] = False


app = FastAPI(
    title="LubDub Murmur API",
    description="Heart Sound Analysis System API powered by multi-location deep learning.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Model & Feature Extractors Cache
MODEL_CACHE = {
    "model": None,
    "model_config": None,
    "feature_config": None,
    "mel_extractor": None,
    "mfcc_extractor": None,
    "loc_builder": None,
    "circor_df": None
}

def get_pipeline():
    """Initializes and caches ML pipeline components."""
    if MODEL_CACHE["model"] is None:
        feat_config = FeatureConfig()
        mel_ext = MelSpectrogramExtractor(feat_config)
        mfcc_ext = MFCCExtractor(feat_config)
        loc_b = MultiLocationPatientTensor(
            num_channels=feat_config.combined_mfcc_channels,
            num_time_frames=feat_config.num_time_frames
        )
        
        mod3_config = Module3Config()
        model = MultiLocationHeartSoundNet(mod3_config)
        
        # Load weights
        ckpt_path = Path(MODULE3_MODEL_PATH)
        if not ckpt_path.exists():
            # Check data/processed as fallback
            ckpt_path = PROCESSED_DATA_DIR / "module3_best_model.pt"
            
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {ckpt_path}")
            
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(ckpt_path), map_location="cpu")
            
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        
        # Metadata CSV
        csv_path = Path(RAW_DATA_DIR) / "training_data.csv"
        df = pd.read_csv(csv_path) if csv_path.exists() else None
        
        MODEL_CACHE["model"] = model
        MODEL_CACHE["model_config"] = mod3_config
        MODEL_CACHE["feature_config"] = feat_config
        MODEL_CACHE["mel_extractor"] = mel_ext
        MODEL_CACHE["mfcc_extractor"] = mfcc_ext
        MODEL_CACHE["loc_builder"] = loc_b
        MODEL_CACHE["circor_df"] = df

    return MODEL_CACHE

def read_audio_bytes(file_bytes: bytes) -> Tuple[np.ndarray, int, float]:
    """Decodes WAV bytes into float32 array, returns (audio, sr, duration)."""
    with io.BytesIO(file_bytes) as bio:
        with wave.open(bio, "rb") as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            raw_bytes = wf.readframes(n_frames)
            
            if sampwidth == 2:
                audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio = np.frombuffer(raw_bytes, dtype=np.float32)
                
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
                
            duration = n_frames / sr if sr > 0 else 0.0
            return audio, sr, duration

def process_single_location(
    audio: np.ndarray,
    sr: int,
    pipeline: Dict[str, Any]
) -> np.ndarray:
    """Preprocesses audio using Module 1 and extracts Module 2 features (120, 501)."""
    feat_config = pipeline["feature_config"]
    mel_extractor = pipeline["mel_extractor"]
    mfcc_extractor = pipeline["mfcc_extractor"]

    # Module 1 DSP: 4000Hz -> 2000Hz, 25-400Hz Order 5 Butterworth, Peak Norm 0.95
    proc_audio, current_sr = preprocess_module1_audio(
        audio,
        orig_sr=sr,
        target_sr=MODULE1_TARGET_SAMPLE_RATE,
        norm_method="peak",
        target_peak=0.95
    )

    # 5.0s windowing (10,000 samples) with 2.5s hop
    windows = segment_and_pad_waveform(
        proc_audio,
        sr=current_sr,
        window_sec=MODULE1_WINDOW_SEC,
        hop_sec=MODULE1_HOP_SEC
    ) # (N_windows, 10000)

    # Module 2 Feature Extraction per window
    win_features = []
    for w in windows:
        p_mel = mel_extractor.extract_power_mel(w)
        if np.max(p_mel) < 1e-12:
            log_mel_unnorm = np.zeros((feat_config.n_mels, feat_config.num_time_frames), dtype=np.float32)
        else:
            log_mel_unnorm = librosa.power_to_db(p_mel, ref=np.max, top_db=feat_config.top_db)
        log_mel = mel_extractor.standardize_feature_map(log_mel_unnorm)

        mfcc, delta, delta2, comb = mfcc_extractor.extract_all(
            w,
            log_mel=log_mel_unnorm,
            standardize=True
        )
        win_features.append(comb)

    return np.mean(np.array(win_features), axis=0) # (120, 501)

@app.get("/api/health")
@app.get("/api/status")
def health_check():
    """Health check endpoint."""
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "model_loaded": pipeline["model"] is not None,
        "app_name": "LubDub Murmur",
        "target_sample_rate": MODULE1_TARGET_SAMPLE_RATE,
        "supported_locations": ORDERED_LOCATIONS
    }

@app.post("/api/analyze")
async def analyze_heart_sounds(
    patient_name: str = Form("Anonymous Patient"),
    patient_id: Optional[str] = Form(None),
    av_file: Optional[UploadFile] = File(None),
    mv_file: Optional[UploadFile] = File(None),
    pv_file: Optional[UploadFile] = File(None),
    tv_file: Optional[UploadFile] = File(None),
    phc_file: Optional[UploadFile] = File(None)
):
    """
    Performs patient-level multi-location inference from uploaded WAV files.
    """
    pipeline = get_pipeline()
    
    file_map = {
        "AV": av_file,
        "MV": mv_file,
        "PV": pv_file,
        "TV": tv_file,
        "Phc": phc_file
    }
    
    available_locs = []
    location_features = {}
    location_audios = {}
    recordings_info = {}
    
    for loc, ufile in file_map.items():
        if ufile is not None and ufile.filename:
            content = await ufile.read()
            if len(content) > 0:
                try:
                    audio, sr, dur = read_audio_bytes(content)
                    feat = process_single_location(audio, sr, pipeline)
                    location_features[loc] = feat
                    location_audios[loc] = audio
                    available_locs.append(loc)
                    recordings_info[loc] = {
                        "filename": ufile.filename,
                        "duration_sec": round(dur, 2),
                        "sample_rate": sr,
                        "available": True
                    }
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to process {loc} audio file '{ufile.filename}': {str(e)}"
                    )
        else:
            recordings_info[loc] = {
                "filename": None,
                "duration_sec": 0.0,
                "sample_rate": 0,
                "available": False
            }

    if len(location_features) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one auscultation recording (AV, MV, PV, TV, or Phc) must be provided."
        )

    # Build (5, 120, 501) tensor and (5,) mask
    loc_builder = pipeline["loc_builder"]
    patient_tensor_np, location_mask_np = loc_builder.build_patient_tensor(location_features)
    
    patient_tensor = torch.from_numpy(patient_tensor_np).unsqueeze(0).float()
    location_mask = torch.from_numpy(location_mask_np).unsqueeze(0).float()

    # Inference
    model = pipeline["model"]
    mod3_config = pipeline["model_config"]
    
    with torch.no_grad():
        outputs = model(patient_tensor, location_mask)

    murmur_prob = float(outputs["murmur_probs"].item())
    outcome_prob = float(outputs["outcome_probs"].item())
    murmur_pred = int(murmur_prob >= mod3_config.murmur_threshold)
    outcome_pred = int(outcome_prob >= mod3_config.outcome_threshold)
    attn_weights = outputs["attention_weights"].squeeze(0).numpy() # (5,)

    location_names = ORDERED_LOCATIONS # ["AV", "MV", "PV", "TV", "Phc"]

    location_attention_dict = {
        loc: {
            "weight": float(attn_weights[i]),
            "percent": round(float(attn_weights[i]) * 100, 2),
            "is_active": loc in available_locs
        }
        for i, loc in enumerate(location_names)
    }

    ai_explanation = generate_result_explanation(
        murmur_pred="Present" if murmur_pred == 1 else "Absent",
        murmur_prob=murmur_prob,
        outcome_pred="Abnormal" if outcome_pred == 1 else "Normal",
        outcome_prob=outcome_prob,
        available_locations=available_locs,
        location_attention=location_attention_dict
    )

    tracking_service = get_tracking_service()
    clean_patient_id = patient_id.strip() if (patient_id and patient_id.strip() and patient_id.strip().upper() not in ["NONE", "N/A", "UNKNOWN"]) else (patient_name.strip() or "Anonymous")
    patient_tracking = tracking_service.add_visit(
        patient_id=clean_patient_id,
        patient_name=patient_name.strip() or "Anonymous Patient",
        murmur_prediction="Present" if murmur_pred == 1 else "Absent",
        murmur_prediction_class=murmur_pred,
        murmur_probability=murmur_prob,
        murmur_probability_percent=round(murmur_prob * 100, 2),
        clinical_outcome="Abnormal" if outcome_pred == 1 else "Normal",
        clinical_outcome_class=outcome_pred,
        location_attention=location_attention_dict,
        is_demo=False
    )

    # Feature 4: Acoustic Similarity Search & Fingerprint V2
    similarity_service = get_similarity_service()
    try:
        query_fp = similarity_service.generate_fingerprint_from_audio(location_audios, orig_sr=MODULE1_TARGET_SAMPLE_RATE)
        acoustic_similarity = similarity_service.search_similar_recordings(
            query_fp,
            top_k=5,
            exclude_patient_id=clean_patient_id
        )
    except Exception as e:
        acoustic_similarity = {
            "fingerprint": {
                "version": "v2",
                "id": "HS-UNAVAILABLE",
                "dimension": 1264,
                "status": "Fingerprint unavailable",
                "metadata": {}
            },
            "matches": [],
            "reference_library_size": 0,
            "message": f"Acoustic similarity search error: {str(e)}",
            "disclaimer": "Acoustic similarity only — this result does not indicate medical similarity, disease similarity, or clinical diagnosis."
        }

    return {
        "patient_name": patient_name.strip() or "Anonymous Patient",
        "patient_id": patient_id.strip() if patient_id else "N/A",
        "is_demo": False,
        "available_locations": available_locs,
        "missing_locations": [loc for loc in location_names if loc not in available_locs],
        "recordings_info": recordings_info,
        "tensor_shape": list(patient_tensor.shape),
        "location_mask": location_mask_np.tolist(),
        "murmur_detection": {
            "prediction": "Present" if murmur_pred == 1 else "Absent",
            "prediction_class": murmur_pred,
            "probability": murmur_prob,
            "probability_percent": round(murmur_prob * 100, 2),
            "threshold": mod3_config.murmur_threshold,
            "status_label": "Murmur Detected" if murmur_pred == 1 else "No Murmur Detected"
        },
        "clinical_outcome": {
            "prediction": "Abnormal" if outcome_pred == 1 else "Normal",
            "prediction_class": outcome_pred,
            "probability": outcome_prob,
            "probability_percent": round(outcome_prob * 100, 2),
            "threshold": mod3_config.outcome_threshold,
            "status_label": "Abnormal Clinical Finding" if outcome_pred == 1 else "Normal Cardiac Function"
        },
        "location_attention": location_attention_dict,
        "ai_explanation": ai_explanation,
        "patient_tracking": patient_tracking,
        "acoustic_similarity": acoustic_similarity,
        "disclaimer": "This is an AI model prediction for research/demo purposes and is not a medical diagnosis."
    }

@app.post("/api/demo")
def run_demo_patient():
    """
    Executes real inference on CirCor Test Patient 40840 using existing audio files.
    Retrieves metadata directly from training_data.csv.
    """
    pipeline = get_pipeline()
    training_data_dir = Path(RAW_DATA_DIR) / "training_data"
    demo_pid = "40840"
    
    # Retrieve demo metadata from CSV
    df = pipeline["circor_df"]
    demo_meta = {}
    if df is not None:
        p_row = df[df["Patient ID"].astype(str) == demo_pid]
        if len(p_row) > 0:
            row_dict = p_row.iloc[0].to_dict()
            demo_meta = {
                "age": str(row_dict.get("Age", "Adolescent")),
                "sex": str(row_dict.get("Sex", "Female")),
                "height": f"{row_dict.get('Height', 147.0)} cm",
                "weight": f"{row_dict.get('Weight', 36.1)} kg",
                "pregnancy": str(row_dict.get("Pregnancy status", "False")),
                "ground_truth_murmur": str(row_dict.get("Murmur", "Present")),
                "ground_truth_outcome": str(row_dict.get("Outcome", "Abnormal"))
            }

    available_locs = ["AV", "MV", "PV", "TV"]
    location_features = {}
    location_audios = {}
    recordings_info = {}

    for loc in ["AV", "MV", "PV", "TV", "Phc"]:
        wav_path = training_data_dir / f"{demo_pid}_{loc}.wav"
        if wav_path.exists():
            with open(wav_path, "rb") as f:
                content = f.read()
            audio, sr, dur = read_audio_bytes(content)
            feat = process_single_location(audio, sr, pipeline)
            location_features[loc] = feat
            location_audios[loc] = audio
            recordings_info[loc] = {
                "filename": wav_path.name,
                "duration_sec": round(dur, 2),
                "sample_rate": sr,
                "available": True
            }
        else:
            recordings_info[loc] = {
                "filename": None,
                "duration_sec": 0.0,
                "sample_rate": 0,
                "available": False
            }

    # Build tensor
    loc_builder = pipeline["loc_builder"]
    patient_tensor_np, location_mask_np = loc_builder.build_patient_tensor(location_features)
    
    patient_tensor = torch.from_numpy(patient_tensor_np).unsqueeze(0).float()
    location_mask = torch.from_numpy(location_mask_np).unsqueeze(0).float()

    # Inference
    model = pipeline["model"]
    mod3_config = pipeline["model_config"]
    
    with torch.no_grad():
        outputs = model(patient_tensor, location_mask)

    murmur_prob = float(outputs["murmur_probs"].item())
    outcome_prob = float(outputs["outcome_probs"].item())
    murmur_pred = int(murmur_prob >= mod3_config.murmur_threshold)
    outcome_pred = int(outcome_prob >= mod3_config.outcome_threshold)
    attn_weights = outputs["attention_weights"].squeeze(0).numpy()

    location_names = ORDERED_LOCATIONS

    location_attention_dict = {
        loc: {
            "weight": float(attn_weights[i]),
            "percent": round(float(attn_weights[i]) * 100, 2),
            "is_active": loc in available_locs
        }
        for i, loc in enumerate(location_names)
    }

    ai_explanation = generate_result_explanation(
        murmur_pred="Present" if murmur_pred == 1 else "Absent",
        murmur_prob=murmur_prob,
        outcome_pred="Abnormal" if outcome_pred == 1 else "Normal",
        outcome_prob=outcome_prob,
        available_locations=available_locs,
        location_attention=location_attention_dict
    )

    tracking_service = get_tracking_service()
    patient_tracking = tracking_service.add_visit(
        patient_id=demo_pid,
        patient_name=f"CirCor Test Patient {demo_pid}",
        murmur_prediction="Present" if murmur_pred == 1 else "Absent",
        murmur_prediction_class=murmur_pred,
        murmur_probability=murmur_prob,
        murmur_probability_percent=round(murmur_prob * 100, 2),
        clinical_outcome="Abnormal" if outcome_pred == 1 else "Normal",
        clinical_outcome_class=outcome_pred,
        location_attention=location_attention_dict,
        is_demo=True
    )

    # Feature 4: Acoustic Similarity Search & Fingerprint V2
    similarity_service = get_similarity_service()
    try:
        query_fp = similarity_service.generate_fingerprint_from_audio(location_audios, orig_sr=SAMPLE_RATE)
        acoustic_similarity = similarity_service.search_similar_recordings(
            query_fp,
            top_k=5,
            exclude_patient_id=demo_pid
        )
    except Exception as e:
        acoustic_similarity = {
            "fingerprint": {
                "version": "v2",
                "id": "HS-UNAVAILABLE",
                "dimension": 1264,
                "status": "Fingerprint unavailable",
                "metadata": {}
            },
            "matches": [],
            "reference_library_size": 0,
            "message": f"Acoustic similarity search error: {str(e)}",
            "disclaimer": "Acoustic similarity only — this result does not indicate medical similarity, disease similarity, or clinical diagnosis."
        }

    return {
        "patient_name": f"CirCor Test Patient {demo_pid}",
        "patient_id": demo_pid,
        "is_demo": True,
        "demo_metadata": demo_meta,
        "available_locations": available_locs,
        "missing_locations": ["Phc"],
        "recordings_info": recordings_info,
        "tensor_shape": list(patient_tensor.shape),
        "location_mask": location_mask_np.tolist(),
        "murmur_detection": {
            "prediction": "Present" if murmur_pred == 1 else "Absent",
            "prediction_class": murmur_pred,
            "probability": murmur_prob,
            "probability_percent": round(murmur_prob * 100, 2),
            "threshold": mod3_config.murmur_threshold,
            "status_label": "Murmur Detected" if murmur_pred == 1 else "No Murmur Detected"
        },
        "clinical_outcome": {
            "prediction": "Abnormal" if outcome_pred == 1 else "Normal",
            "prediction_class": outcome_pred,
            "probability": outcome_prob,
            "probability_percent": round(outcome_prob * 100, 2),
            "threshold": mod3_config.outcome_threshold,
            "status_label": "Abnormal Clinical Finding" if outcome_pred == 1 else "Normal Cardiac Function"
        },
        "location_attention": location_attention_dict,
        "ai_explanation": ai_explanation,
        "patient_tracking": patient_tracking,
        "acoustic_similarity": acoustic_similarity,
        "disclaimer": "This is an AI model prediction for research/demo purposes and is not a medical diagnosis."
    }

@app.post("/api/similarity/search")
async def search_similarity_endpoint(
    patient_id: Optional[str] = Form(None),
    top_k: int = Form(5),
    av_file: Optional[UploadFile] = File(None),
    mv_file: Optional[UploadFile] = File(None),
    pv_file: Optional[UploadFile] = File(None),
    tv_file: Optional[UploadFile] = File(None),
    phc_file: Optional[UploadFile] = File(None)
):
    """
    Computes Heart Sound Fingerprint V2 and searches for top-K acoustically similar reference recordings.
    """
    file_map = {
        "AV": av_file,
        "MV": mv_file,
        "PV": pv_file,
        "TV": tv_file,
        "Phc": phc_file
    }
    location_audios = {}
    for loc, ufile in file_map.items():
        if ufile is not None and ufile.filename:
            content = await ufile.read()
            if len(content) > 0:
                audio, sr, dur = read_audio_bytes(content)
                location_audios[loc] = audio

    clean_pid = patient_id.strip() if patient_id else None
    if len(location_audios) == 0 and clean_pid:
        training_dir = Path(RAW_DATA_DIR) / "training_data"
        for loc in ["AV", "MV", "PV", "TV"]:
            wpath = training_dir / f"{clean_pid}_{loc}.wav"
            if wpath.exists():
                with open(wpath, "rb") as f:
                    content = f.read()
                audio, sr, dur = read_audio_bytes(content)
                location_audios[loc] = audio

    if len(location_audios) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one heart sound recording file or valid patient ID must be provided."
        )

    similarity_service = get_similarity_service()
    query_fp = similarity_service.generate_fingerprint_from_audio(location_audios, orig_sr=SAMPLE_RATE)
    results = similarity_service.search_similar_recordings(
        query_fp,
        top_k=top_k,
        exclude_patient_id=clean_pid
    )
    return results

@app.get("/api/patient/{patient_id}/history")
def get_patient_history_endpoint(patient_id: str):
    """
    Retrieves longitudinal analysis history and AI acoustic trend metrics for a patient.
    """
    tracking_service = get_tracking_service()
    return tracking_service.get_patient_history(patient_id)

@app.post("/api/patient/{patient_id}/record")
def record_patient_visit_endpoint(patient_id: str, req: PatientRecordRequest):
    """
    Manually records a visit for a patient.
    """
    tracking_service = get_tracking_service()
    return tracking_service.add_visit(
        patient_id=patient_id,
        patient_name=req.patient_name or "Anonymous Patient",
        murmur_prediction=req.murmur_prediction,
        murmur_prediction_class=req.murmur_prediction_class,
        murmur_probability=req.murmur_probability,
        murmur_probability_percent=req.murmur_probability_percent,
        clinical_outcome=req.clinical_outcome,
        clinical_outcome_class=req.clinical_outcome_class,
        location_attention=req.location_attention,
        is_demo=req.is_demo or False
    )

@app.delete("/api/patient/{patient_id}/history")
def clear_patient_history_endpoint(patient_id: str):
    """
    Clears analysis history for a patient.
    """
    tracking_service = get_tracking_service()
    success = tracking_service.clear_patient_history(patient_id)
    return {"status": "cleared" if success else "not_found", "patient_id": patient_id}


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Context-aware conversational assistant endpoint.
    Answers patient questions dynamically grounded in the current heart sound screening prediction.
    """
    chatbot = get_chatbot_service()
    res = chatbot.chat(
        session_id=req.session_id,
        user_message=req.message,
        context=req.context
    )
    return res

@app.post("/api/chat/reset")
async def chat_reset_endpoint(req: ChatResetRequest):
    """
    Resets the conversation history for a new screening session.
    """
    chatbot = get_chatbot_service()
    chatbot.reset_session(session_id=req.session_id, new_context=req.context)
    return {"status": "session_reset", "session_id": req.session_id}

@app.get("/api/audio/demo/{location}")
def get_demo_audio(location: str):
    """Streams existing audio recording for demo patient 40840."""
    loc = location.upper()
    wav_path = Path(RAW_DATA_DIR) / "training_data" / f"40840_{loc}.wav"
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail=f"Demo recording for {loc} not found")
    return FileResponse(
        path=str(wav_path),
        media_type="audio/wav",
        filename=f"40840_{loc}.wav"
    )

@app.post("/api/report")
async def generate_report_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    Generates a downloadable clinical PDF report from a completed patient analysis result.
    Uses the existing analysis result and does NOT rerun ML inference.
    """
    if not payload:
        raise HTTPException(
            status_code=400,
            detail="Analysis result payload is required to generate report."
        )
    
    try:
        pdf_bytes = generate_patient_pdf_report(payload)
        p_id = payload.get("patient_id") or payload.get("patient_name") or "Analysis"
        safe_id = sanitize_filename(str(p_id))
        filename = f"LubDub_Murmur_Report_{safe_id}.pdf"
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "application/pdf"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate patient PDF report: {str(e)}"
        )

# =====================================================================
# NEW FEATURE: HFrEF Multi-Modal ECG + PCG & Integrated Result Endpoints
# Completely independent from existing LubDub model and CirCor dataset
# =====================================================================

@app.get("/api/hfref/status")
def get_hfref_status_endpoint():
    """Returns dataset presence, summary, and model status for the HFrEF module."""
    loader = get_dataset_loader()
    analyzer = get_hfref_analyzer()
    summary = loader.get_dataset_summary()
    summary["model_trained"] = analyzer.is_model_trained()
    summary["cv_metrics"] = analyzer.get_cv_metrics()
    return summary

@app.get("/api/hfref/subjects")
def get_hfref_subjects_endpoint(limit: Optional[int] = 50):
    """Returns indexed subjects from the HFrEF dataset if present."""
    loader = get_dataset_loader()
    return {
        "is_present": loader.is_dataset_present(),
        "total_indexed": len(loader.list_subjects()),
        "subjects": loader.list_subjects(limit=limit)
    }

@app.post("/api/hfref/analyze")
async def analyze_hfref_endpoint(
    subject_id: Optional[str] = Form(None),
    ecg_file: Optional[UploadFile] = File(None),
    rusb_file: Optional[UploadFile] = File(None),
    lusb_file: Optional[UploadFile] = File(None),
    llsb_file: Optional[UploadFile] = File(None),
    apex_file: Optional[UploadFile] = File(None)
):
    """
    Analyzes an HFrEF subject by ID from local dataset or processes uploaded ECG and PCG WAV files.
    """
    loader = get_dataset_loader()
    analyzer = get_hfref_analyzer()
    recommender = get_specialist_recommender()

    # Case 1: Analyze indexed subject by subject_id
    if subject_id and subject_id.strip():
        sid = subject_id.strip()
        analysis = analyzer.analyze_subject_by_id(sid, loader=loader)
        if analysis:
            rec = recommender.generate_recommendation(hfref_result=analysis)
            analysis["specialist_recommendation"] = rec
            return analysis
        # If subject ID not found in indexed dataset
        if not loader.is_dataset_present():
            raise HTTPException(
                status_code=404,
                detail="HFrEF dataset archive not detected in data/hfref/. Please place HFrEF_5_folds.zip in data/hfref/."
            )
        raise HTTPException(
            status_code=404,
            detail=f"Subject '{sid}' not found in HFrEF dataset."
        )

    # Case 2: Analyze uploaded audio files
    ecg_signals = {}
    pcg_signals = {}

    if ecg_file and ecg_file.filename:
        content = await ecg_file.read()
        if len(content) > 0:
            arr, sr, dur = read_audio_bytes(content)
            ecg_signals["APEX"] = {"signal": arr, "sr": sr}

    file_site_map = {
        "RUSB": rusb_file,
        "LUSB": lusb_file,
        "LLSB": llsb_file,
        "APEX": apex_file
    }

    for site, ufile in file_site_map.items():
        if ufile and ufile.filename:
            content = await ufile.read()
            if len(content) > 0:
                arr, sr, dur = read_audio_bytes(content)
                pcg_signals[site] = {"signal": arr, "sr": sr}

    if len(ecg_signals) == 0 and len(pcg_signals) == 0:
        raise HTTPException(
            status_code=400,
            detail="Either a valid HFrEF subject_id or uploaded ECG/PCG WAV files must be provided."
        )

    subject_signals = {
        "subject_id": "Uploaded_Subject",
        "lvef": None,
        "is_hfref": None,
        "fold": None,
        "ecg_signals": ecg_signals,
        "pcg_signals": pcg_signals
    }

    analysis = analyzer.analyze_signals(subject_signals)
    rec = recommender.generate_recommendation(hfref_result=analysis)
    analysis["specialist_recommendation"] = rec
    return analysis

@app.post("/api/hfref/demo")
def run_hfref_demo_endpoint():
    """
    Executes real inference on the first available HFrEF dataset subject if present.
    If dataset is pending placement, returns status information.
    """
    loader = get_dataset_loader()
    analyzer = get_hfref_analyzer()
    recommender = get_specialist_recommender()

    if not loader.is_dataset_present():
        return {
            "is_demo_ready": False,
            "status": "dataset_pending_placement",
            "message": (
                "HFrEF dataset not detected in data/hfref/. "
                "Place HFrEF_5_folds.zip in data/hfref/ to activate full subject demo."
            ),
            "expected_path": "data/hfref/HFrEF_5_folds.zip"
        }

    subjects = loader.list_subjects(limit=1)
    if len(subjects) == 0:
        return {
            "is_demo_ready": False,
            "status": "no_subjects_indexed",
            "message": "No indexed subjects found in extracted HFrEF dataset."
        }

    first_sub = subjects[0]
    analysis = analyzer.analyze_subject_by_id(first_sub["subject_id"], loader=loader)
    if analysis:
        rec = recommender.generate_recommendation(hfref_result=analysis)
        analysis["specialist_recommendation"] = rec
        analysis["is_demo"] = True
        analysis["demo_notice"] = "DEMO / UI WORKFLOW ONLY — NOT A REAL HFrEF PREDICTION"
        return analysis

    raise HTTPException(status_code=500, detail="Failed to run HFrEF demo analysis.")

@app.get("/api/integrated/result")
def get_integrated_result_endpoint():
    """
    Consolidated View endpoint:
    Returns the combined outputs of BOTH systems without modifying either system:
    1. Existing LubDub Result (Murmur detection, Clinical outcome, Attention, Fingerprint, Similarity)
    2. New ECG + Multi-Location PCG Analysis (ECG metrics, 4-site PCG features, HFrEF/LVEF status)
    3. Consolidated Specialist Recommendation
    """
    # 1. Existing LubDub result
    lubdub_data = run_demo_patient()

    # 2. HFrEF status / analysis
    loader = get_dataset_loader()
    analyzer = get_hfref_analyzer()
    recommender = get_specialist_recommender()

    hfref_status = loader.get_dataset_summary()
    hfref_analysis = None

    if loader.is_dataset_present():
        subjects = loader.list_subjects(limit=1)
        if len(subjects) > 0:
            hfref_analysis = analyzer.analyze_subject_by_id(subjects[0]["subject_id"], loader=loader)
            if hfref_analysis:
                hfref_analysis["is_demo"] = True
                hfref_analysis["demo_notice"] = "DEMO / UI WORKFLOW ONLY — NOT A REAL HFrEF PREDICTION"

    # 3. Consolidated Specialist Recommendation
    specialist_rec = recommender.generate_recommendation(
        hfref_result=hfref_analysis,
        lubdub_result=lubdub_data
    )

    is_present = loader.is_dataset_present()

    return {
        "status": "success",
        "timestamp": pd.Timestamp.now().isoformat(),
        "patient_overview": {
            "primary_patient_id": lubdub_data.get("patient_id", "40840"),
            "display_name": "Integrated Clinical Screening Profile",
            "note": "Software-level consolidated view of independent diagnostic modalities."
        },
        "existing_lubdub_result": lubdub_data,
        "new_hfref_ecg_pcg_result": hfref_analysis,
        "hfref_dataset_status": hfref_status,
        "hfref_dataset_available": is_present,
        "hfref_display_state": {
            "dataset_availability": "HFrEF Dataset: Not Available" if not is_present else "Available",
            "analysis_status": "HFrEF Analysis: Pending Dataset" if not is_present else "Completed",
            "prediction_status": "No clinical prediction generated" if not is_present else "Clinical prediction generated",
            "demo_notice": "DEMO / UI WORKFLOW ONLY — NOT A REAL HFrEF PREDICTION" if (hfref_analysis and hfref_analysis.get("is_demo")) else None
        },
        "specialist_recommendation": specialist_rec,
        "architecture_summary": {
            "existing_system": "LubDub Murmur — CirCor PCG Multi-Location Attention Net (Murmur + Outcome + Fingerprint)",
            "new_feature": "HFrEF Multi-Modal — EKO DUO Synchronized ECG + 4-Valve PCG Fusion Net (LVEF/HFrEF Screening)",
            "data_separation": "Datasets remain strictly separate. CirCor and HFrEF are not merged into a single dataset."
        }
    }

@app.post("/api/integrated/recommendation")
def generate_integrated_recommendation_endpoint(payload: Dict[str, Any] = Body(...)):
    """Generates dynamic specialist recommendation given combined analysis results."""
    recommender = get_specialist_recommender()
    hfref_res = payload.get("hfref_result")
    lubdub_res = payload.get("lubdub_result")
    return recommender.generate_recommendation(hfref_result=hfref_res, lubdub_result=lubdub_res)

@app.post("/api/integrated/report")
async def generate_integrated_report_endpoint(payload: Dict[str, Any] = Body(...)):
    """
    Generates a downloadable clinical PDF report from consolidated Integrated Results.
    Existing patient PDF report endpoint remains completely available.
    """
    if not payload:
        raise HTTPException(
            status_code=400,
            detail="Consolidated result payload is required to generate integrated report."
        )
    
    try:
        pdf_bytes = generate_integrated_pdf_report(payload)
        p_id = "Integrated"
        if "existing_lubdub_result" in payload and payload["existing_lubdub_result"]:
            p_id = payload["existing_lubdub_result"].get("patient_id", "40840")
        safe_id = sanitize_filename(str(p_id))
        filename = f"LubDub_Integrated_Report_{safe_id}.pdf"
        
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Type": "application/pdf"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate integrated PDF report: {str(e)}"
        )

# Mount static directory for frontend
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
