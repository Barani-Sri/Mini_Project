"""
Heart Sound Fingerprint Generator.
Generates fixed-dimensional, normalized, reproducible acoustic fingerprints combining
MFCC/Delta dynamics, Log-Mel frequency energy, and deep Audio Spectrogram Transformer (AST) embeddings.
"""
import json
import base64
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import librosa

from src.fingerprint.config import FingerprintConfig, FINGERPRINT_VERSION
from src.audio.preprocessing import preprocess_module1_audio, segment_and_pad_waveform
from src.features.config import FeatureConfig
from src.features.melspectrogram import MelSpectrogramExtractor
from src.features.mfcc import MFCCExtractor
from src.ast.ast_embedding_extractor import ASTEmbeddingExtractor

@dataclass
class HeartSoundFingerprint:
    """
    Structured container for a standardized Heart Sound Acoustic Fingerprint.
    """
    vector: np.ndarray                        # 1D Float32 array of shape (dimension,)
    dimension: int                            # Total dimension (e.g. 1016 for v1)
    version: str = FINGERPRINT_VERSION        # Version tag (e.g. "v1")
    location: Optional[str] = None            # Valve area ("AV", "MV", etc.) or "multi_location"
    component_dims: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, include_vector: bool = True) -> Dict[str, Any]:
        """Serializes fingerprint to a JSON-safe dictionary (excluding personal/PII data)."""
        res = {
            "version": self.version,
            "dimension": int(self.dimension),
            "location": self.location,
            "component_dims": self.component_dims,
            "metadata": self.metadata
        }
        if include_vector:
            res["vector"] = self.vector.tolist()
        return res

    def to_json(self, include_vector: bool = True) -> str:
        """Serializes to JSON string."""
        return json.dumps(self.to_dict(include_vector=include_vector), indent=2)

    def to_base64(self) -> str:
        """Encodes raw float32 vector as base64 string for efficient transmission."""
        return base64.b64encode(self.vector.astype(np.float32).tobytes()).decode("utf-8")

    @classmethod
    def from_base64(
        cls,
        b64_str: str,
        dimension: int = 1016,
        version: str = FINGERPRINT_VERSION,
        location: Optional[str] = None
    ) -> "HeartSoundFingerprint":
        """Reconstructs fingerprint from base64 string."""
        raw_bytes = base64.b64decode(b64_str.encode("utf-8"))
        vec = np.frombuffer(raw_bytes, dtype=np.float32)
        return cls(
            vector=vec,
            dimension=len(vec),
            version=version,
            location=location
        )


class HeartSoundFingerprintGenerator:
    """
    Generates standardized acoustic fingerprints from heart sound PCG recordings.
    Combines:
    1. Module 2 MFCC Summary (120 dims)
    2. Module 2 Log-Mel Summary (128 dims)
    3. AST Pretrained Embedding (768 dims)
    """
    def __init__(
        self,
        config: Optional[FingerprintConfig] = None,
        ast_extractor: Optional[ASTEmbeddingExtractor] = None
    ):
        self.config = config if config is not None else FingerprintConfig()
        self.feature_config = FeatureConfig()
        self.mel_extractor = MelSpectrogramExtractor(self.feature_config)
        self.mfcc_extractor = MFCCExtractor(self.feature_config)
        self.ast_extractor = ast_extractor if ast_extractor is not None else ASTEmbeddingExtractor()

    def generate_recording_fingerprint(
        self,
        audio: np.ndarray,
        orig_sr: int = 4000,
        location: Optional[str] = None
    ) -> HeartSoundFingerprint:
        """
        Generates a heart sound fingerprint from a single PCG audio recording.
        
        Args:
            audio: Raw or preprocessed 1D PCG waveform.
            orig_sr: Original sampling rate of audio (default 4000 Hz).
            location: Auscultation location (e.g. 'AV', 'MV', 'PV', 'TV', 'Phc').
            
        Returns:
            HeartSoundFingerprint instance with fixed-dimensional L2-normalized vector.
        """
        if len(audio) == 0:
            zeros = np.zeros(self.config.total_dimension, dtype=np.float32)
            return HeartSoundFingerprint(
                vector=zeros,
                dimension=self.config.total_dimension,
                version=self.config.version,
                location=location
            )

        # 1. Module 1 DSP Preprocessing: resample to 2000Hz, bandpass 25-400Hz, peak norm 0.95
        proc_audio, current_sr = preprocess_module1_audio(
            audio,
            orig_sr=orig_sr,
            target_sr=self.feature_config.sample_rate,
            norm_method="peak",
            target_peak=0.95
        )

        # 2. Fixed 5.0-second windowing
        windows = segment_and_pad_waveform(
            proc_audio,
            sr=current_sr,
            window_sec=5.0,
            hop_sec=2.5
        )

        # 3. Extract Traditional Acoustic Features across windows
        win_mfccs = []
        win_mels = []

        for w in windows:
            p_mel = self.mel_extractor.extract_power_mel(w)
            if np.max(p_mel) < 1e-12:
                log_mel_unnorm = np.zeros((self.feature_config.n_mels, self.feature_config.num_time_frames), dtype=np.float32)
            else:
                log_mel_unnorm = librosa.power_to_db(p_mel, ref=np.max, top_db=self.feature_config.top_db)
            log_mel = self.mel_extractor.standardize_feature_map(log_mel_unnorm)

            mfcc, delta, delta2, comb = self.mfcc_extractor.extract_all(
                w,
                log_mel=log_mel_unnorm,
                standardize=True
            )
            # Summarize window across time: mean over frames -> (120,)
            win_mfccs.append(np.mean(comb, axis=1))
            # Summarize mel across time: mean over frames -> (128,)
            win_mels.append(np.mean(log_mel, axis=1))

        # Mean across all windows for this recording
        mfcc_vec = np.mean(np.array(win_mfccs), axis=0).astype(np.float32) # (120,)
        mel_vec = np.mean(np.array(win_mels), axis=0).astype(np.float32)   # (128,)

        # 4. Extract AST Embedding on the primary 5.0-second standardized window
        # (or average AST embeddings across 5s windows)
        ast_embeddings = []
        for w in windows:
            ast_emb = self.ast_extractor.extract_embedding(w, orig_sr=current_sr)
            ast_embeddings.append(ast_emb)
        ast_vec = np.mean(np.array(ast_embeddings), axis=0).astype(np.float32) # (768,)

        # 5. Component-Level Normalization
        if self.config.normalize_components:
            mfcc_vec = self._l2_normalize(mfcc_vec)
            mel_vec = self._l2_normalize(mel_vec)
            ast_vec = self._l2_normalize(ast_vec)

        # 6. Weighted Fusion & Concatenation
        components = []
        comp_dims = {}

        if self.config.use_mfcc:
            components.append(mfcc_vec * self.config.weight_mfcc)
            comp_dims["mfcc"] = len(mfcc_vec)

        if self.config.use_mel:
            components.append(mel_vec * self.config.weight_mel)
            comp_dims["mel"] = len(mel_vec)

        if self.config.use_ast:
            components.append(ast_vec * self.config.weight_ast)
            comp_dims["ast"] = len(ast_vec)

        fused_vec = np.concatenate(components, axis=0).astype(np.float32)

        # 7. Final Global Normalization to Unit Sphere
        if self.config.normalize_output:
            fused_vec = self._l2_normalize(fused_vec)

        return HeartSoundFingerprint(
            vector=fused_vec,
            dimension=len(fused_vec),
            version=self.config.version,
            location=location,
            component_dims=comp_dims,
            metadata={
                "num_windows": len(windows),
                "duration_sec": round(len(proc_audio) / current_sr, 2),
                "sample_rate": current_sr
            }
        )

    def generate_patient_fingerprint(
        self,
        location_audios: Dict[str, np.ndarray],
        orig_sr: int = 4000
    ) -> HeartSoundFingerprint:
        """
        Generates a composite multi-location fingerprint for a patient from available valve recordings.
        """
        if not location_audios:
            zeros = np.zeros(self.config.total_dimension, dtype=np.float32)
            return HeartSoundFingerprint(
                vector=zeros,
                dimension=self.config.total_dimension,
                version=self.config.version,
                location="multi_location"
            )

        loc_fps = []
        active_locs = []

        for loc, audio in location_audios.items():
            if audio is not None and len(audio) > 0:
                fp = self.generate_recording_fingerprint(audio, orig_sr=orig_sr, location=loc)
                loc_fps.append(fp.vector)
                active_locs.append(loc)

        if not loc_fps:
            zeros = np.zeros(self.config.total_dimension, dtype=np.float32)
            return HeartSoundFingerprint(
                vector=zeros,
                dimension=self.config.total_dimension,
                version=self.config.version,
                location="multi_location"
            )

        # Mean across available valve site fingerprints
        patient_vec = np.mean(np.array(loc_fps), axis=0).astype(np.float32)
        if self.config.normalize_output:
            patient_vec = self._l2_normalize(patient_vec)

        return HeartSoundFingerprint(
            vector=patient_vec,
            dimension=len(patient_vec),
            version=self.config.version,
            location="multi_location",
            component_dims={
                "mfcc": self.config.mfcc_dim if self.config.use_mfcc else 0,
                "mel": self.config.mel_dim if self.config.use_mel else 0,
                "ast": self.config.ast_dim if self.config.use_ast else 0
            },
            metadata={
                "active_locations": active_locs,
                "num_locations": len(active_locs)
            }
        )

    @staticmethod
    def _l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < eps:
            return np.zeros_like(vec, dtype=np.float32)
        return (vec / (norm + eps)).astype(np.float32)


def generate_fingerprint(
    audio: np.ndarray,
    orig_sr: int = 4000,
    location: Optional[str] = None,
    config: Optional[FingerprintConfig] = None
) -> HeartSoundFingerprint:
    """Convenience function: generates a heart sound fingerprint from PCG audio."""
    gen = HeartSoundFingerprintGenerator(config=config)
    return gen.generate_recording_fingerprint(audio, orig_sr=orig_sr, location=location)
