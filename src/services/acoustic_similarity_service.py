"""
Acoustic Similarity Service & Heart-Sound Fingerprint V2 Engine.
Manages deterministic 1,264-D temporal-aware fingerprint generation, reference indexing,
and top-K acoustic nearest-neighbor search.

CLINICAL / MEDICAL SAFETY GUARDRAILS:
Acoustic similarity measures signal-level and spectral-temporal waveform proximity.
It does NOT establish medical similarity, shared pathology, or a clinical diagnosis.
"""
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd

from src.utils.config import (
    BASE_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLE_RATE
)
from src.fingerprint_v2.config import FingerprintV2Config, FINGERPRINT_V2_VERSION
from src.fingerprint_v2.fingerprint_v2_generator import (
    HeartSoundFingerprintV2,
    HeartSoundFingerprintV2Generator,
    generate_fingerprint_v2
)
from src.fingerprint_v2.fingerprint_v2_similarity import AcousticSimilarityCalculatorV2


REFERENCE_INDEX_DIR = PROCESSED_DATA_DIR / "fingerprint_index"
REFERENCE_LIBRARY_FILE = REFERENCE_INDEX_DIR / "reference_library_v2.json"


def compute_deterministic_fingerprint_id(vector: np.ndarray) -> str:
    """
    Computes a stable, deterministic 8-character hex hash identifier from the fingerprint vector.
    Format: 'HS-XXXXXXXX' (e.g. 'HS-A7F3D92C')
    """
    if vector is None or len(vector) == 0:
        return "HS-00000000"
    # Convert rounded float32 array to bytes for numerical hash stability
    vec_bytes = np.round(vector.astype(np.float32), decimals=5).tobytes()
    digest = hashlib.sha256(vec_bytes).hexdigest()[:8].upper()
    return f"HS-{digest}"


class AcousticSimilarityService:
    """
    Acoustic Similarity Service using Heart-Sound Fingerprint V2.
    """
    def __init__(
        self,
        config: Optional[FingerprintV2Config] = None,
        generator: Optional[HeartSoundFingerprintV2Generator] = None,
        reference_file: Optional[Path] = None
    ):
        self.config = config if config is not None else FingerprintV2Config()
        self.generator = generator if generator is not None else HeartSoundFingerprintV2Generator(self.config)
        self.similarity_calculator = AcousticSimilarityCalculatorV2(self.config)
        self.reference_file = Path(reference_file) if reference_file is not None else REFERENCE_LIBRARY_FILE
        self._reference_library: List[Dict[str, Any]] = []
        self._load_or_build_reference_library()

    @property
    def reference_library(self) -> List[Dict[str, Any]]:
        return self._reference_library

    def generate_fingerprint_from_audio(
        self,
        audio_dict: Dict[str, np.ndarray],
        orig_sr: int = 4000
    ) -> HeartSoundFingerprintV2:
        """
        Generates a 1,264-D multi-location patient fingerprint V2 from audio waveforms.
        """
        # Filter active non-empty audios
        valid_audios = {
            loc: audio for loc, audio in audio_dict.items()
            if audio is not None and len(audio) > 0
        }
        if len(valid_audios) == 0:
            zeros = np.zeros(self.config.total_dimension, dtype=np.float32)
            return HeartSoundFingerprintV2(
                vector=zeros,
                dimension=self.config.total_dimension,
                version=self.config.version,
                location="multi_location",
                metadata={"status": "empty_input"}
            )

        if len(valid_audios) == 1:
            loc, audio = next(iter(valid_audios.items()))
            fp = self.generator.generate_recording_fingerprint(audio, orig_sr=orig_sr, location=loc)
            return fp

        fp = self.generator.generate_patient_fingerprint(valid_audios, orig_sr=orig_sr)
        return fp

    def search_similar_recordings(
        self,
        query_fp: Union[HeartSoundFingerprintV2, np.ndarray],
        top_k: int = 5,
        exclude_patient_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Searches the reference library for the top-K acoustically similar recordings.
        Prevents data leakage by excluding the query patient ID if present.
        """
        if isinstance(query_fp, HeartSoundFingerprintV2):
            query_vec = query_fp.vector
            meta = query_fp.metadata
        else:
            query_vec = np.asarray(query_fp, dtype=np.float32)
            meta = {}

        fingerprint_id = compute_deterministic_fingerprint_id(query_vec)

        if len(query_vec) == 0 or np.linalg.norm(query_vec) < 1e-8:
            return {
                "fingerprint": {
                    "version": self.config.version,
                    "id": "HS-UNAVAILABLE",
                    "dimension": self.config.total_dimension,
                    "status": "Fingerprint unavailable",
                    "metadata": meta
                },
                "matches": [],
                "reference_library_size": len(self._reference_library),
                "message": "No valid acoustic signal available for fingerprint extraction.",
                "disclaimer": self._disclaimer_text()
            }

        if len(self._reference_library) == 0:
            return {
                "fingerprint": {
                    "version": self.config.version,
                    "id": fingerprint_id,
                    "dimension": len(query_vec),
                    "status": "Generated ✓",
                    "metadata": meta
                },
                "matches": [],
                "reference_library_size": 0,
                "message": "No reference recordings are currently available for acoustic similarity search.",
                "disclaimer": self._disclaimer_text()
            }

        clean_exclude_id = str(exclude_patient_id).strip() if exclude_patient_id else None

        scored_candidates = []
        for ref_item in self._reference_library:
            ref_pid = str(ref_item.get("patient_id", "")).strip()

            # Self-matching exclusion: Do not compare against the query patient itself
            if clean_exclude_id and ref_pid == clean_exclude_id:
                continue

            ref_vec = np.array(ref_item["vector"], dtype=np.float32)
            sim_result = self.similarity_calculator.compute_similarity(query_vec, ref_vec)

            score = sim_result["acoustic_similarity_score"]
            percent = sim_result["acoustic_similarity_percent"]
            comp_sims = sim_result.get("component_similarities", {})

            scored_candidates.append({
                "reference_id": ref_item.get("reference_id", f"REF-{ref_pid}"),
                "patient_id": ref_pid,
                "similarity_score": score,
                "similarity_percent": percent,
                "component_similarities": comp_sims,
                "available_locations": ref_item.get("locations", ["AV", "MV", "PV", "TV"]),
                "demographics": ref_item.get("demographics", {}),
                "split": ref_item.get("split", "train"),
                "interpretation": sim_result.get("interpretation", "Comparable acoustic profile.")
            })

        # Sort descending by similarity score
        scored_candidates.sort(key=lambda x: x["similarity_score"], reverse=True)

        top_matches = scored_candidates[:top_k]

        # Add 1-based ranks
        for rank, match in enumerate(top_matches, start=1):
            match["rank"] = rank

        return {
            "fingerprint": {
                "version": self.config.version,
                "id": fingerprint_id,
                "dimension": len(query_vec),
                "status": "Generated ✓",
                "metadata": meta
            },
            "matches": top_matches,
            "reference_library_size": len(self._reference_library),
            "total_candidates_searched": len(scored_candidates),
            "disclaimer": self._disclaimer_text()
        }

    def _load_or_build_reference_library(self):
        """Loads existing reference library from JSON or builds and caches a default reference library."""
        if self.reference_file.exists():
            try:
                with open(self.reference_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        self._reference_library = data
                        return
            except Exception:
                pass

        # If not present, build reference library from CirCor training dataset
        self._reference_library = self._build_default_reference_library()
        self._save_reference_library()

    def _build_default_reference_library(self, max_patients: int = 12) -> List[Dict[str, Any]]:
        """
        Builds a high-quality reference library using training split patients from CirCor dataset.
        """
        ref_lib = []
        csv_path = RAW_DATA_DIR / "training_data.csv"
        training_dir = RAW_DATA_DIR / "training_data"

        if not csv_path.exists() or not training_dir.exists():
            return self._build_synthetic_seed_reference_library()

        try:
            df = pd.read_csv(csv_path)
            from src.data.splitting import get_patient_split_sets
            splits = get_patient_split_sets(df, random_state=42)
            train_pids = list(splits.get("train", []))

            # Select diverse training patients
            selected_train_pids = []
            for pid in train_pids:
                all_exist = all((training_dir / f"{pid}_{loc}.wav").exists() for loc in ["AV", "MV", "PV", "TV"])
                if all_exist:
                    selected_train_pids.append(pid)
                if len(selected_train_pids) >= max_patients:
                    break

            for idx, pid in enumerate(selected_train_pids, start=1):
                p_row = df[df["Patient ID"].astype(str) == str(pid)]
                demo = {}
                if len(p_row) > 0:
                    r = p_row.iloc[0]
                    demo = {
                        "age_group": str(r.get("Age", "Adolescent")),
                        "sex": str(r.get("Sex", "Unknown"))
                    }

                # Load recordings (limit to first 10s for fast representation)
                loc_audios = {}
                active_locs = []
                for loc in ["AV", "MV", "PV", "TV"]:
                    wpath = training_dir / f"{pid}_{loc}.wav"
                    if wpath.exists():
                        import wave
                        with wave.open(str(wpath), "rb") as wf:
                            n_frames = min(wf.getnframes(), 4000 * 10) # 10s max
                            sr = wf.getframerate()
                            raw = wf.readframes(n_frames)
                            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                            loc_audios[loc] = audio
                            active_locs.append(loc)

                if loc_audios:
                    fp = self.generator.generate_patient_fingerprint(loc_audios, orig_sr=SAMPLE_RATE)
                    ref_id = f"REF-{str(pid).zfill(5)}"
                    ref_lib.append({
                        "reference_id": ref_id,
                        "patient_id": str(pid),
                        "locations": active_locs,
                        "fingerprint_id": compute_deterministic_fingerprint_id(fp.vector),
                        "vector": fp.vector.tolist(),
                        "demographics": demo,
                        "split": "train"
                    })
        except Exception:
            return self._build_synthetic_seed_reference_library()

        if len(ref_lib) == 0:
            return self._build_synthetic_seed_reference_library()

        return ref_lib

    def _build_synthetic_seed_reference_library(self) -> List[Dict[str, Any]]:
        """Fallback reference library if raw dataset audio is unavailable."""
        seed_refs = []
        rng = np.random.RandomState(42)
        demo_archetypes = [
            ("2530", ["AV", "MV", "PV", "TV"], {"age_group": "Child", "sex": "Female"}),
            ("9979", ["AV", "MV", "PV", "TV"], {"age_group": "Adolescent", "sex": "Male"}),
            ("13918", ["AV", "MV", "PV", "TV"], {"age_group": "Child", "sex": "Male"}),
            ("14241", ["AV", "MV", "PV", "TV"], {"age_group": "Child", "sex": "Female"}),
            ("14998", ["AV", "MV", "PV", "TV"], {"age_group": "Adolescent", "sex": "Female"}),
            ("23625", ["AV", "MV", "PV", "TV"], {"age_group": "Adult", "sex": "Male"}),
            ("24160", ["AV", "MV", "PV", "TV"], {"age_group": "Adolescent", "sex": "Male"}),
            ("29045", ["AV", "MV", "PV", "TV"], {"age_group": "Child", "sex": "Female"}),
            ("29378", ["AV", "MV", "PV", "TV"], {"age_group": "Child", "sex": "Male"}),
            ("30005", ["AV", "MV", "PV", "TV"], {"age_group": "Adolescent", "sex": "Female"})
        ]

        for pid, locs, demo in demo_archetypes:
            vec = rng.randn(self.config.total_dimension).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            fpid = compute_deterministic_fingerprint_id(vec)
            seed_refs.append({
                "reference_id": f"REF-{pid}",
                "patient_id": str(pid),
                "locations": locs,
                "fingerprint_id": fpid,
                "vector": vec.tolist(),
                "demographics": demo,
                "split": "train"
            })
        return seed_refs

    def _save_reference_library(self):
        """Persists reference library to JSON."""
        try:
            REFERENCE_INDEX_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.reference_file, "w", encoding="utf-8") as f:
                json.dump(self._reference_library, f, indent=2)
        except Exception:
            pass

    @staticmethod
    def _disclaimer_text() -> str:
        return (
            "Acoustic similarity only — this result does not indicate medical similarity, "
            "disease similarity, or clinical diagnosis."
        )


_SIMILARITY_SERVICE_INSTANCE: Optional[AcousticSimilarityService] = None

def get_similarity_service() -> AcousticSimilarityService:
    """Singleton getter for AcousticSimilarityService."""
    global _SIMILARITY_SERVICE_INSTANCE
    if _SIMILARITY_SERVICE_INSTANCE is None:
        _SIMILARITY_SERVICE_INSTANCE = AcousticSimilarityService()
    return _SIMILARITY_SERVICE_INSTANCE
