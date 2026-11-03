"""
Dataset loader and parser for the HFrEF Multi-Channel ECG and PCG Dataset.
Reads and validates the official HFrEF dataset structure from Zenodo (Record 16934966).
Correctly associates ECG and multi-location PCG recordings for each individual subject.
Never mixes recordings between different subjects.
"""
import os
import re
import wave
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from src.hfref.config import (
    HFREF_DIR,
    HFREF_ZIP_PATH,
    HFREF_EXTRACTED_DIR,
    HFREF_ECG_SAMPLE_RATE,
    HFREF_PCG_SAMPLE_RATE,
    AUSCULTATION_CODE_MAP,
    ORDERED_HFREF_LOCATIONS,
    LVEF_HFREF_THRESHOLD
)

@dataclass
class HFrEFRecordingMeta:
    """Metadata for a single signal recording file."""
    filepath: Path
    filename: str
    modality: str            # 'ECG' or 'PCG'
    site_code: str           # 'APEX', 'LLSB', 'LUSB', 'RUSB'
    valve_code: str          # 'MV', 'TV', 'PV', 'AV'
    sample_rate: int
    duration_sec: float
    num_samples: int
    file_size_bytes: int

@dataclass
class HFrEFSubjectRecord:
    """Consolidated multimodal record for a single HFrEF dataset subject."""
    subject_id: str
    fold: Optional[int]
    lvef: Optional[float]
    is_hfref: Optional[bool]
    ecg_recordings: Dict[str, HFrEFRecordingMeta] = field(default_factory=dict)
    pcg_recordings: Dict[str, HFrEFRecordingMeta] = field(default_factory=dict)
    demographics: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_all_pcg_locations(self) -> bool:
        """Checks if all 4 PCG auscultation locations are present."""
        return all(loc in self.pcg_recordings for loc in ORDERED_HFREF_LOCATIONS)

    @property
    def available_pcg_locations(self) -> List[str]:
        """Returns list of available PCG location codes."""
        return [loc for loc in ORDERED_HFREF_LOCATIONS if loc in self.pcg_recordings]

    @property
    def primary_ecg_recording(self) -> Optional[HFrEFRecordingMeta]:
        """Returns primary ECG recording (preferring APEX/RUSB or first available)."""
        for loc in ["APEX", "RUSB", "LLSB", "LUSB"]:
            if loc in self.ecg_recordings:
                return self.ecg_recordings[loc]
        return None

    def to_summary_dict(self) -> Dict[str, Any]:
        """Serializes subject metadata to a clean dictionary."""
        return {
            "subject_id": self.subject_id,
            "fold": self.fold,
            "lvef": self.lvef,
            "is_hfref": self.is_hfref,
            "lvef_status": "HFrEF (LVEF <= 40%)" if self.is_hfref else ("Preserved LVEF (> 40%)" if self.is_hfref is False else "Unknown"),
            "ecg_channels_count": len(self.ecg_recordings),
            "pcg_channels_count": len(self.pcg_recordings),
            "available_pcg_locations": [AUSCULTATION_CODE_MAP[loc]["lubdub_code"] for loc in self.available_pcg_locations],
            "demographics": self.demographics
        }


class HFrEFDatasetLoader:
    """
    Manages discovery, validation, and loading of the HFrEF 5-fold dataset.
    Detects whether the dataset is present in data/hfref/ without throwing errors.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else HFREF_DIR
        self.zip_path = self.data_dir / "HFrEF_5_folds.zip"
        self.extracted_dir = self.data_dir / "extracted"
        self._subjects: Dict[str, HFrEFSubjectRecord] = {}
        self._is_indexed = False
        self._summary_cache: Optional[Dict[str, Any]] = None

    def is_dataset_present(self) -> bool:
        """
        Determines whether the dataset is available locally.
        Checks for extracted signal files or a valid complete ZIP archive.
        """
        if self.extracted_dir.exists():
            wav_files = list(self.extracted_dir.glob("**/*.wav"))
            if len(wav_files) > 0:
                return True

        if self.zip_path.exists() and self.zip_path.stat().st_size > 200000000:
            try:
                with zipfile.ZipFile(self.zip_path, 'r') as z:
                    return len(z.namelist()) > 0
            except zipfile.BadZipFile:
                return False

        return False

    def ensure_extracted(self) -> bool:
        """
        Extracts HFrEF_5_folds.zip into data/hfref/extracted if not already extracted.
        Returns True if files are ready, False otherwise.
        """
        if self.extracted_dir.exists():
            wav_files = list(self.extracted_dir.glob("**/*.wav"))
            if len(wav_files) > 0:
                return True

        if self.zip_path.exists() and self.zip_path.stat().st_size > 200000000:
            try:
                self.extracted_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(self.zip_path, 'r') as z:
                    z.extractall(self.extracted_dir)
                return True
            except Exception as e:
                print(f"[WARN] Failed to extract {self.zip_path}: {e}")
                return False

        return False

    def index_dataset(self, force: bool = False) -> Dict[str, HFrEFSubjectRecord]:
        """
        Scans all files and metadata in extracted directory and indexes subjects.
        Only indexes real files present on disk. Does not fabricate records.
        """
        if self._is_indexed and not force:
            return self._subjects

        self._subjects = {}
        self._is_indexed = True

        if not self.ensure_extracted():
            return self._subjects

        # Step 1: Discover all label CSV files (*_label.csv or labels.csv)
        labels_by_subject: Dict[str, Dict[str, Any]] = {}
        label_csvs = list(self.extracted_dir.glob("**/*label*.csv"))
        
        for csv_path in label_csvs:
            try:
                # Detect fold from path if organized by fold (e.g., fold_0_label.csv or fold0/fold0_label.csv)
                fold_num = None
                fold_match = re.search(r"fold[_\-\s]?([0-4])", str(csv_path), re.IGNORECASE)
                if fold_match:
                    fold_num = int(fold_match.group(1))

                # Check if CSV has a header
                df_sample = pd.read_csv(csv_path, nrows=5)
                has_header = any(
                    str(c).strip().lower() in ["id", "subject_id", "patient_id", "lvef", "ef"]
                    for c in df_sample.columns
                )

                if has_header:
                    df = pd.read_csv(csv_path)
                    cols = {str(c).strip().lower(): c for c in df.columns}
                    id_col = cols.get("id") or cols.get("subject_id") or cols.get("patient_id") or df.columns[0]
                    lvef_col = cols.get("lvef") or cols.get("ef") or (df.columns[1] if len(df.columns) > 1 else None)
                else:
                    df = pd.read_csv(csv_path, header=None)
                    id_col = 0
                    lvef_col = 1 if len(df.columns) > 1 else None

                for _, row in df.iterrows():
                    raw_id = str(row[id_col]).strip()
                    # Handle composite IDs like 1005263204_APEX -> 1005263204
                    sid = raw_id.split("_")[0] if "_" in raw_id else raw_id

                    lvef_val = None
                    if lvef_col is not None and pd.notna(row[lvef_col]):
                        try:
                            lvef_raw = float(row[lvef_col])
                            # If given as fraction <= 1.0 (e.g. 0.72), scale to percentage (72.0%)
                            lvef_val = round(lvef_raw * 100.0, 1) if lvef_raw <= 1.0 else round(lvef_raw, 1)
                        except ValueError:
                            pass

                    is_hfref = (lvef_val <= LVEF_HFREF_THRESHOLD) if lvef_val is not None else None
                    labels_by_subject[sid] = {
                        "lvef": lvef_val,
                        "is_hfref": is_hfref,
                        "fold": fold_num,
                        "demographics": {f"col_{k}": row[k] for k in df.columns if k not in [id_col, lvef_col] and pd.notna(row[k])}
                    }
            except Exception as e:
                print(f"[WARN] Failed reading label CSV {csv_path}: {e}")

        # Step 2: Discover and index all audio recordings
        # Standard naming pattern: [ID]_[APEX/LLSB/LUSB/RUSB]_[ECG/PCG].wav
        wav_files = list(self.extracted_dir.glob("**/*.wav"))
        file_pattern = re.compile(
            r"^(?P<subject>[A-Za-z0-9_\-]+)_(?P<site>APEX|LLSB|LUSB|RUSB)_(?P<modality>ECG|PCG)\.wav$",
            re.IGNORECASE
        )

        for wpath in wav_files:
            match = file_pattern.match(wpath.name)
            if not match:
                # Try relaxed match: [ID]_[site]_[modality].wav
                parts = wpath.stem.split("_")
                if len(parts) >= 3 and parts[-1].upper() in ["ECG", "PCG"] and parts[-2].upper() in AUSCULTATION_CODE_MAP:
                    sid = "_".join(parts[:-2])
                    site = parts[-2].upper()
                    mod = parts[-1].upper()
                else:
                    continue
            else:
                sid = match.group("subject")
                site = match.group("site").upper()
                mod = match.group("modality").upper()

            # Read audio metadata quickly using wave module
            try:
                with wave.open(str(wpath), "rb") as wf:
                    sr = wf.getframerate()
                    n_frames = wf.getnframes()
                    dur = n_frames / sr if sr > 0 else 0.0
            except Exception:
                sr = HFREF_ECG_SAMPLE_RATE if mod == "ECG" else HFREF_PCG_SAMPLE_RATE
                dur = 30.0
                n_frames = int(dur * sr)

            rec_meta = HFrEFRecordingMeta(
                filepath=wpath,
                filename=wpath.name,
                modality=mod,
                site_code=site,
                valve_code=AUSCULTATION_CODE_MAP[site]["lubdub_code"],
                sample_rate=sr,
                duration_sec=round(dur, 2),
                num_samples=n_frames,
                file_size_bytes=wpath.stat().st_size
            )

            if sid not in self._subjects:
                lbl = labels_by_subject.get(sid, {})
                fold_match = re.search(r"fold[_\-\s]?([0-4])", str(wpath), re.IGNORECASE)
                assigned_fold = lbl.get("fold") if lbl.get("fold") is not None else (int(fold_match.group(1)) if fold_match else None)

                self._subjects[sid] = HFrEFSubjectRecord(
                    subject_id=sid,
                    fold=assigned_fold,
                    lvef=lbl.get("lvef"),
                    is_hfref=lbl.get("is_hfref"),
                    demographics=lbl.get("demographics", {})
                )

            if mod == "ECG":
                self._subjects[sid].ecg_recordings[site] = rec_meta
            elif mod == "PCG":
                self._subjects[sid].pcg_recordings[site] = rec_meta

        return self._subjects

    def get_dataset_summary(self) -> Dict[str, Any]:
        """
        Returns comprehensive inspection summary of the HFrEF dataset.
        Always reflects actual disk state.
        """
        is_ready = self.is_dataset_present()
        subjects = self.index_dataset() if is_ready else {}

        if not is_ready or len(subjects) == 0:
            return {
                "dataset_name": "HFrEF Multi-Channel ECG and PCG Dataset",
                "zenodo_record": "16934966",
                "is_present": False,
                "status": "dataset_pending_placement",
                "expected_zip_path": str(self.zip_path),
                "expected_extracted_path": str(self.extracted_dir),
                "message": (
                    "HFrEF dataset not detected in data/hfref/. "
                    "Place HFrEF_5_folds.zip into data/hfref/ to activate full training and evaluation."
                ),
                "total_subjects": 0,
                "ecg_recordings_count": 0,
                "pcg_recordings_count": 0,
                "folds_count": 0,
                "supported_modalities": ["ECG", "PCG"],
                "supported_pcg_locations": {
                    site: info["valve_name"] for site, info in AUSCULTATION_CODE_MAP.items()
                }
            }

        ecg_count = sum(len(s.ecg_recordings) for s in subjects.values())
        pcg_count = sum(len(s.pcg_recordings) for s in subjects.values())
        folds = set(s.fold for s in subjects.values() if s.fold is not None)
        lvef_known = sum(1 for s in subjects.values() if s.lvef is not None)
        hfref_count = sum(1 for s in subjects.values() if s.is_hfref is True)
        non_hfref_count = sum(1 for s in subjects.values() if s.is_hfref is False)

        return {
            "dataset_name": "HFrEF Multi-Channel ECG and PCG Dataset",
            "zenodo_record": "16934966",
            "is_present": True,
            "status": "indexed_and_ready",
            "total_subjects": len(subjects),
            "ecg_recordings_count": ecg_count,
            "pcg_recordings_count": pcg_count,
            "total_recordings": ecg_count + pcg_count,
            "folds_count": len(folds),
            "folds": sorted(list(folds)),
            "subjects_with_lvef": lvef_known,
            "hfref_subjects_count": hfref_count,
            "non_hfref_subjects_count": non_hfref_count,
            "sampling_rates": {
                "ECG": HFREF_ECG_SAMPLE_RATE,
                "PCG": HFREF_PCG_SAMPLE_RATE
            },
            "locations": {
                site: {
                    "valve_name": info["valve_name"],
                    "lubdub_code": info["lubdub_code"],
                    "anatomical_site": info["anatomical_site"]
                }
                for site, info in AUSCULTATION_CODE_MAP.items()
            }
        }

    def list_subjects(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns list of indexed subjects with their available recordings."""
        subjects = self.index_dataset()
        res = [s.to_summary_dict() for s in subjects.values()]
        return res[:limit] if limit else res

    def get_subject(self, subject_id: str) -> Optional[HFrEFSubjectRecord]:
        """Retrieves a specific subject record by ID."""
        subjects = self.index_dataset()
        return subjects.get(str(subject_id))

    def load_subject_signals(
        self,
        subject_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Loads the decoded audio waveforms for a single subject.
        Returns a dictionary with ECG and PCG signals per location.
        """
        sub = self.get_subject(subject_id)
        if not sub:
            return None

        def _read_wav(filepath: Path) -> Tuple[np.ndarray, int]:
            with wave.open(str(filepath), "rb") as wf:
                sr = wf.getframerate()
                n = wf.getnframes()
                sampwidth = wf.getsampwidth()
                raw = wf.readframes(n)
                if sampwidth == 2:
                    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                elif sampwidth == 4:
                    data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                else:
                    data = np.frombuffer(raw, dtype=np.float32)
                if wf.getnchannels() > 1:
                    data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
                return data, sr

        signals = {
            "subject_id": sub.subject_id,
            "fold": sub.fold,
            "lvef": sub.lvef,
            "is_hfref": sub.is_hfref,
            "ecg_signals": {},
            "pcg_signals": {}
        }

        for site, meta in sub.ecg_recordings.items():
            if meta.filepath.exists():
                arr, sr = _read_wav(meta.filepath)
                signals["ecg_signals"][site] = {"signal": arr, "sr": sr, "meta": meta}

        for site, meta in sub.pcg_recordings.items():
            if meta.filepath.exists():
                arr, sr = _read_wav(meta.filepath)
                signals["pcg_signals"][site] = {"signal": arr, "sr": sr, "meta": meta}

        return signals


# Global loader cache
_DATASET_LOADER_INSTANCE: Optional[HFrEFDatasetLoader] = None

def get_dataset_loader() -> HFrEFDatasetLoader:
    """Returns singleton HFrEF dataset loader instance."""
    global _DATASET_LOADER_INSTANCE
    if _DATASET_LOADER_INSTANCE is None:
        _DATASET_LOADER_INSTANCE = HFrEFDatasetLoader()
    return _DATASET_LOADER_INSTANCE
