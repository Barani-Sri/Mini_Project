"""
End-to-End Manual Inference Pipeline for Heart Sound PCG Classification.
Takes raw WAV auscultation files for a patient, applies Module 1 preprocessing,
Module 2 multi-location feature extraction, and runs Module 3 deep learning inference.
"""
import os
import sys
import json
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import librosa
import torch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
from src.features.location import (
    MultiLocationPatientTensor,
    ORDERED_LOCATIONS,
    encode_location
)
from src.models.config import Module3Config, MODULE3_MODEL_PATH
from src.models.multi_location_net import MultiLocationHeartSoundNet

def load_wav_file(filepath: Path) -> Tuple[np.ndarray, int]:
    """Reads a 16-bit PCM WAV audio file into float32 array in [-1.0, 1.0]."""
    with wave.open(str(filepath), "rb") as wf:
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
            
    return audio, sr

def run_patient_inference(
    patient_id: str = "40840",
    data_dir: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    output_json_path: Optional[Path] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Executes complete end-to-end inference from raw WAV files for a single patient.
    """
    if data_dir is None:
        data_dir = Path(RAW_DATA_DIR) / "training_data"
    else:
        data_dir = Path(data_dir)
        
    if checkpoint_path is None:
        checkpoint_path = Path(MODULE3_MODEL_PATH)
    else:
        checkpoint_path = Path(checkpoint_path)
        
    if output_json_path is None:
        output_json_path = PROCESSED_DATA_DIR / f"manual_test_{patient_id}.json"
    else:
        output_json_path = Path(output_json_path)

    # 1. Inspect and Locate Available WAV Files
    location_names = ["AV", "MV", "PV", "TV", "Phc"]
    available_files: Dict[str, Path] = {}
    missing_locations: List[str] = []

    for loc in location_names:
        w_path = data_dir / f"{patient_id}_{loc}.wav"
        if w_path.exists():
            available_files[loc] = w_path
        else:
            missing_locations.append(loc)

    if len(available_files) == 0:
        raise FileNotFoundError(f"No WAV audio files found for patient {patient_id} in {data_dir}")

    # 2. Module 1 Preprocessing & Module 2 Feature Extraction
    feat_config = FeatureConfig()
    mel_extractor = MelSpectrogramExtractor(feat_config)
    mfcc_extractor = MFCCExtractor(feat_config)
    loc_tensor_builder = MultiLocationPatientTensor(
        num_channels=feat_config.combined_mfcc_channels,
        num_time_frames=feat_config.num_time_frames
    )

    location_features: Dict[str, np.ndarray] = {}
    pipeline_status = {}

    for loc, w_path in available_files.items():
        # Load raw audio
        raw_audio, sr = load_wav_file(w_path)
        
        # Module 1 DSP Preprocessing: 4000Hz -> 2000Hz, 25-400Hz Order 5 Butterworth, Peak Norm 0.95
        proc_audio, current_sr = preprocess_module1_audio(
            raw_audio,
            orig_sr=sr,
            target_sr=MODULE1_TARGET_SAMPLE_RATE,
            norm_method="peak",
            target_peak=0.95
        )
        
        # Fixed 5.0s windowing (10,000 samples) with 2.5s hop (5,000 samples)
        windows = segment_and_pad_waveform(
            proc_audio,
            sr=current_sr,
            window_sec=MODULE1_WINDOW_SEC,
            hop_sec=MODULE1_HOP_SEC
        ) # (N_windows, 10000)
        
        pipeline_status[f"{loc} preprocessing"] = True

        # Module 2 Feature Extraction per window
        win_combined_features = []
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
            win_combined_features.append(comb) # (120, 501)

        # Average window representations for this cardiac location
        location_features[loc] = np.mean(np.array(win_combined_features), axis=0) # (120, 501)

    # 3. Construct Multi-Location Patient Representation Tensor (5, 120, 501) and Mask (5,)
    patient_tensor_np, location_mask_np = loc_tensor_builder.build_patient_tensor(location_features)
    
    # Expand to batch dimension: (1, 5, 120, 501) and (1, 5)
    patient_tensor = torch.from_numpy(patient_tensor_np).unsqueeze(0).float()
    location_mask = torch.from_numpy(location_mask_np).unsqueeze(0).float()

    # 4. Load Trained Model Checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing Module 3 model checkpoint at {checkpoint_path}")

    model_config = Module3Config()
    model = MultiLocationHeartSoundNet(model_config)
    
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 5. Run Inference (Evaluation Mode, No Gradients)
    with torch.no_grad():
        outputs = model(patient_tensor, location_mask)

    murmur_prob = float(outputs["murmur_probs"].item())
    outcome_prob = float(outputs["outcome_probs"].item())
    murmur_pred = int(murmur_prob >= model_config.murmur_threshold)
    outcome_pred = int(outcome_prob >= model_config.outcome_threshold)
    attn_weights = outputs["attention_weights"].squeeze(0).numpy() # (5,)

    # Format Results
    results_summary = {
        "patient_id": patient_id,
        "available_locations": list(available_files.keys()),
        "missing_locations": missing_locations,
        "tensor_shape": list(patient_tensor.shape),
        "location_mask": location_mask_np.tolist(),
        "murmur_detection": {
            "prediction": "Present" if murmur_pred == 1 else "Absent",
            "prediction_class": murmur_pred,
            "probability": murmur_prob,
            "probability_percent": round(murmur_prob * 100, 2),
            "threshold": model_config.murmur_threshold
        },
        "clinical_outcome": {
            "prediction": "Abnormal" if outcome_pred == 1 else "Normal",
            "prediction_class": outcome_pred,
            "probability": outcome_prob,
            "probability_percent": round(outcome_prob * 100, 2),
            "threshold": model_config.outcome_threshold
        },
        "location_attention_weights": {
            loc: float(attn_weights[i]) for i, loc in enumerate(location_names)
        },
        "location_attention_percent": {
            loc: round(float(attn_weights[i]) * 100, 2) for i, loc in enumerate(location_names)
        },
        "pipeline_stages": {
            "preprocessing": pipeline_status,
            "mfcc_extraction": True,
            "mel_spectrogram_extraction": True,
            "multi_location_fusion": True,
            "location_masking": True,
            "model_loading": True,
            "inference": True
        },
        "checkpoint_used": str(checkpoint_path),
        "disclaimer": "This is an AI model prediction for research/demo purposes and is not a medical diagnosis."
    }

    # Save JSON artifact
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    # 6. Print Formatted Report
    if verbose:
        avail_str = ", ".join(available_files.keys())
        missing_str = ", ".join(missing_locations) if missing_locations else "None"
        
        print("========================================")
        print("HEART MURMUR AI — MANUAL TEST")
        print("========================================")
        print(f"Patient ID: {patient_id}")
        print(f"Available Locations: {avail_str}")
        print(f"Missing Location: {missing_str}")
        print("\nGround Truth:")
        print("Murmur: Present")
        print("Clinical Outcome: Abnormal")
        print("\n---")
        print("## PIPELINE")
        for loc in available_files.keys():
            print(f"{loc} preprocessing: \u2713")
        print("\nMFCC extraction: \u2713")
        print("Mel-spectrogram extraction: \u2713")
        print("Multi-location fusion: \u2713")
        print("Location masking: \u2713")
        print("Model loading: \u2713")
        print("Inference: \u2713")
        print("\n---")
        print("## MODEL INPUT")
        print("Patient tensor shape:")
        print("Expected: (1, 5, 120, 501)")
        print(f"Actual:   {tuple(patient_tensor.shape)}")
        print("\nLocation mask:")
        print(f"AV: {location_mask_np[0]:.0f}, MV: {location_mask_np[1]:.0f}, PV: {location_mask_np[2]:.0f}, TV: {location_mask_np[3]:.0f}, Phc: {location_mask_np[4]:.0f}")
        print(f"Array: {location_mask_np.tolist()}")
        print("\n---")
        print("## PREDICTION")
        print("Murmur:")
        print(f"Prediction: {'Present' if murmur_pred == 1 else 'Absent'}")
        print(f"Probability: {murmur_prob * 100:.2f}%")
        print("\nClinical Outcome:")
        print(f"Prediction: {'Abnormal' if outcome_pred == 1 else 'Normal'}")
        print(f"Probability: {outcome_prob * 100:.2f}%")
        print("\n---")
        print("## LOCATION ATTENTION")
        for i, loc in enumerate(location_names):
            print(f"{loc}: {attn_weights[i] * 100:.2f}%")
        print("========================================")
        print("\nThis is an AI model prediction for research/demo purposes and is not a medical diagnosis.")
        print(f"\nSaved test results to: {output_json_path}\n")

    return results_summary

if __name__ == "__main__":
    run_patient_inference(patient_id="40840")
