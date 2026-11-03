"""
Acoustic Similarity Engine.
Computes technical acoustic similarity between heart sound fingerprints using normalized Cosine Similarity.

IMPORTANT CLINICAL NOTICE:
Acoustic similarity evaluates signal-level and spectral-temporal acoustic waveform closeness.
It is NOT a 'medical match' or clinical diagnosis and does not imply that subjects share the same
pathology or medical condition.
"""
from typing import Dict, Any, Union, Optional
import numpy as np

from src.fingerprint.generator import HeartSoundFingerprint
from src.fingerprint.config import FingerprintConfig

class AcousticSimilarityCalculator:
    """
    Computes acoustic similarity metrics between two heart sound fingerprints.
    """
    def __init__(self, config: Optional[FingerprintConfig] = None):
        self.config = config if config is not None else FingerprintConfig()

    def compute_similarity(
        self,
        fp_a: Union[HeartSoundFingerprint, np.ndarray],
        fp_b: Union[HeartSoundFingerprint, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Computes Acoustic Similarity between two heart sound fingerprints.
        
        Args:
            fp_a: First HeartSoundFingerprint or 1D array.
            fp_b: Second HeartSoundFingerprint or 1D array.
            
        Returns:
            Dictionary containing:
            - acoustic_similarity_score: float in [-1.0, 1.0] (Cosine similarity)
            - acoustic_similarity_percent: float in [0.0, 100.0]
            - component_similarities: Dict of sub-component similarities (MFCC, Mel, AST)
            - is_identical_audio: bool (similarity >= 0.9999)
            - interpretation: Technical description of acoustic closeness
            - disclaimer: Mandatory non-diagnostic notice
        """
        vec_a = self._extract_vector(fp_a)
        vec_b = self._extract_vector(fp_b)

        if len(vec_a) != len(vec_b):
            raise ValueError(f"Fingerprint dimension mismatch: {len(vec_a)} vs {len(vec_b)}")

        if len(vec_a) == 0:
            return {
                "acoustic_similarity_score": 0.0,
                "acoustic_similarity_percent": 0.0,
                "component_similarities": {},
                "is_identical_audio": False,
                "interpretation": "Empty fingerprint vectors.",
                "disclaimer": self._disclaimer_text()
            }

        # 1. Cosine Similarity on Unit-Normalized Vectors
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a < 1e-8 or norm_b < 1e-8:
            cosine_sim = 0.0
        else:
            cosine_sim = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

        # Clamp to valid [-1.0, 1.0] numerical bounds
        cosine_sim = max(-1.0, min(1.0, cosine_sim))

        # Percentage score mapped from [-1, 1] to [0, 100] (0.0 similarity -> 50%, 1.0 -> 100%)
        # or normalized positive cosine similarity
        sim_percent = max(0.0, min(100.0, ((cosine_sim + 1.0) / 2.0) * 100.0))

        # 2. Sub-Component Similarities (if standard v1 dimension 1016)
        comp_sims = {}
        if len(vec_a) == self.config.total_dimension:
            idx = 0
            if self.config.use_mfcc:
                mfcc_a = vec_a[idx : idx + self.config.mfcc_dim]
                mfcc_b = vec_b[idx : idx + self.config.mfcc_dim]
                comp_sims["mfcc_similarity"] = self._safe_cosine(mfcc_a, mfcc_b)
                idx += self.config.mfcc_dim

            if self.config.use_mel:
                mel_a = vec_a[idx : idx + self.config.mel_dim]
                mel_b = vec_b[idx : idx + self.config.mel_dim]
                comp_sims["mel_similarity"] = self._safe_cosine(mel_a, mel_b)
                idx += self.config.mel_dim

            if self.config.use_ast:
                ast_a = vec_a[idx : idx + self.config.ast_dim]
                ast_b = vec_b[idx : idx + self.config.ast_dim]
                comp_sims["ast_similarity"] = self._safe_cosine(ast_a, ast_b)
                idx += self.config.ast_dim

        # 3. Technical Interpretation (Strictly Acoustic, Non-Medical)
        is_identical = bool(cosine_sim >= 0.9999)
        if is_identical:
            interp = "Identical acoustic profile (exact or near-exact acoustic waveform match)."
        elif cosine_sim >= 0.85:
            interp = "High acoustic similarity (highly congruent spectral envelope and temporal timbre)."
        elif cosine_sim >= 0.60:
            interp = "Moderate acoustic similarity (comparable frequency distribution with distinct temporal variations)."
        else:
            interp = "Low acoustic similarity (distinct acoustic signatures and spectral profiles)."

        return {
            "metric_name": "Acoustic Similarity",
            "acoustic_similarity_score": round(cosine_sim, 4),
            "acoustic_similarity_percent": round(sim_percent, 2),
            "component_similarities": comp_sims,
            "is_identical_audio": is_identical,
            "interpretation": interp,
            "version": self.config.version,
            "disclaimer": self._disclaimer_text()
        }

    @staticmethod
    def _extract_vector(fp: Union[HeartSoundFingerprint, np.ndarray]) -> np.ndarray:
        if isinstance(fp, HeartSoundFingerprint):
            return np.asarray(fp.vector, dtype=np.float32)
        return np.asarray(fp, dtype=np.float32)

    @staticmethod
    def _safe_cosine(u: np.ndarray, v: np.ndarray) -> float:
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-8 or nv < 1e-8:
            return 0.0
        return float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))

    @staticmethod
    def _disclaimer_text() -> str:
        return (
            "Acoustic Similarity measures signal-level and spectral-temporal waveform proximity. "
            "It does NOT indicate a medical diagnosis, clinical match, or shared disease pathology."
        )


def calculate_acoustic_similarity(
    fp_a: Union[HeartSoundFingerprint, np.ndarray],
    fp_b: Union[HeartSoundFingerprint, np.ndarray],
    config: Optional[FingerprintConfig] = None
) -> Dict[str, Any]:
    """Convenience function: computes acoustic similarity between two heart sound fingerprints."""
    calc = AcousticSimilarityCalculator(config=config)
    return calc.compute_similarity(fp_a, fp_b)
