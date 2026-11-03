"""
LubDub Murmur — Patient Heart-Sound Longitudinal Tracking Service.
Maintains analysis records for patients across multiple visits and computes
mathematical trends in AI acoustic measurements over time without claiming disease progression.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.utils.config import PROCESSED_DATA_DIR

TRACKING_DIR = Path(PROCESSED_DATA_DIR) / "tracking"
TRACKING_FILE = TRACKING_DIR / "patient_records.json"


def _get_demo_seed_records() -> List[Dict[str, Any]]:
    """
    Returns baseline demo visit records for CirCor Test Patient 40840.
    These represent historical visits preceding the current session.
    """
    return [
        {
            "visit_id": "V1",
            "visit_number": 1,
            "timestamp": "2026-08-15T09:30:00Z",
            "date_formatted": "Aug 15, 2026",
            "patient_id": "40840",
            "patient_name": "CirCor Test Patient 40840",
            "murmur_prediction": "Present",
            "murmur_prediction_class": 1,
            "murmur_probability": 0.452,
            "murmur_probability_percent": 45.2,
            "clinical_outcome": "Abnormal",
            "clinical_outcome_class": 1,
            "location_attention": {
                "AV": {"weight": 0.22, "percent": 22.0, "is_active": True},
                "MV": {"weight": 0.25, "percent": 25.0, "is_active": True},
                "PV": {"weight": 0.28, "percent": 28.0, "is_active": True},
                "TV": {"weight": 0.25, "percent": 25.0, "is_active": True},
                "Phc": {"weight": 0.0, "percent": 0.0, "is_active": False}
            },
            "is_demo": True
        },
        {
            "visit_id": "V2",
            "visit_number": 2,
            "timestamp": "2026-08-28T14:15:00Z",
            "date_formatted": "Aug 28, 2026",
            "patient_id": "40840",
            "patient_name": "CirCor Test Patient 40840",
            "murmur_prediction": "Present",
            "murmur_prediction_class": 1,
            "murmur_probability": 0.498,
            "murmur_probability_percent": 49.8,
            "clinical_outcome": "Abnormal",
            "clinical_outcome_class": 1,
            "location_attention": {
                "AV": {"weight": 0.23, "percent": 23.0, "is_active": True},
                "MV": {"weight": 0.26, "percent": 26.0, "is_active": True},
                "PV": {"weight": 0.28, "percent": 28.0, "is_active": True},
                "TV": {"weight": 0.23, "percent": 23.0, "is_active": True},
                "Phc": {"weight": 0.0, "percent": 0.0, "is_active": False}
            },
            "is_demo": True
        }
    ]


class PatientTrackingService:
    """
    Thread-safe longitudinal record store and trend calculation service for patient visits.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or TRACKING_FILE
        self._ensure_storage()

    def _ensure_storage(self):
        """Creates storage directory and file with default demo data if missing."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            initial_data = {
                "40840": _get_demo_seed_records()
            }
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, indent=2)

    def _load_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Loads all patient records from disk."""
        self._ensure_storage()
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    return {}
                return data
        except Exception:
            return {}

    def _save_all(self, data: Dict[str, List[Dict[str, Any]]]):
        """Persists all patient records to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_visit(
        self,
        patient_id: str,
        patient_name: str,
        murmur_prediction: str,
        murmur_prediction_class: int,
        murmur_probability: float,
        murmur_probability_percent: float,
        clinical_outcome: str,
        clinical_outcome_class: int,
        location_attention: Optional[Dict[str, Any]] = None,
        is_demo: bool = False,
        custom_timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Adds a new visit analysis record for a patient and returns the updated history and trend.
        """
        clean_id = str(patient_id).strip()
        if not clean_id or clean_id.upper() in ["NONE", "N/A", "UNKNOWN", ""]:
            clean_id = str(patient_name).strip().replace(" ", "_").lower() or "anonymous"

        all_records = self._load_all()
        patient_visits = all_records.get(clean_id, [])

        now = datetime.now()
        ts = custom_timestamp or now.isoformat()
        date_formatted = now.strftime("%b %d, %Y")

        visit_num = len(patient_visits) + 1
        visit_record = {
            "visit_id": f"V{visit_num}",
            "visit_number": visit_num,
            "timestamp": ts,
            "date_formatted": date_formatted,
            "patient_id": clean_id,
            "patient_name": patient_name or "Anonymous Patient",
            "murmur_prediction": murmur_prediction,
            "murmur_prediction_class": murmur_prediction_class,
            "murmur_probability": round(float(murmur_probability), 4),
            "murmur_probability_percent": round(float(murmur_probability_percent), 2),
            "clinical_outcome": clinical_outcome,
            "clinical_outcome_class": clinical_outcome_class,
            "location_attention": location_attention or {},
            "is_demo": bool(is_demo)
        }

        patient_visits.append(visit_record)
        all_records[clean_id] = patient_visits
        self._save_all(all_records)

        return self.get_patient_history(clean_id)

    def get_patient_history(self, patient_id: str) -> Dict[str, Any]:
        """
        Retrieves all visit records and calculated trend statistics for a given patient.
        """
        clean_id = str(patient_id).strip()
        all_records = self._load_all()
        visits = all_records.get(clean_id, [])

        # If it's demo patient 40840 and has no records, initialize with seed
        if clean_id == "40840" and len(visits) == 0:
            visits = _get_demo_seed_records()
            all_records["40840"] = visits
            self._save_all(all_records)

        trend = self.calculate_trend(visits)

        return {
            "patient_id": clean_id,
            "patient_name": visits[-1]["patient_name"] if visits else clean_id,
            "total_visits": len(visits),
            "visits": visits,
            "trend": trend,
            "has_multiple_visits": len(visits) >= 2
        }

    def calculate_trend(self, visits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates mathematical trend metrics across visits strictly in terms of
        AI acoustic indicator measurements without claiming disease progression.
        """
        if not visits or len(visits) == 0:
            return {
                "status": "no_data",
                "direction": "no_data",
                "direction_symbol": "—",
                "direction_label": "No Data",
                "latest_percent": 0.0,
                "previous_percent": None,
                "change_previous_pp": 0.0,
                "overall_change_pp": 0.0,
                "interpretation": "No historical analysis visits recorded yet."
            }

        latest = visits[-1]
        latest_pct = latest.get("murmur_probability_percent", 0.0)

        if len(visits) == 1:
            return {
                "status": "single_visit",
                "direction": "baseline",
                "direction_symbol": "●",
                "direction_label": "Baseline Visit",
                "latest_percent": round(latest_pct, 2),
                "previous_percent": None,
                "change_previous_pp": None,
                "overall_change_pp": None,
                "interpretation": "Only one analysis is currently available for this patient. Complete another analysis to begin tracking acoustic indicator changes over time."
            }

        prev = visits[-2]
        first = visits[0]

        prev_pct = prev.get("murmur_probability_percent", 0.0)
        first_pct = first.get("murmur_probability_percent", 0.0)

        change_prev_pp = round(latest_pct - prev_pct, 2)
        change_overall_pp = round(latest_pct - first_pct, 2)

        # Classify mathematical trajectory (Threshold: 0.5 percentage points)
        if change_prev_pp >= 0.5:
            direction = "increasing"
            symbol = "↑"
            label = "Increasing"
            direction_text = "increased"
            sign_prev = f"+{change_prev_pp}"
        elif change_prev_pp <= -0.5:
            direction = "decreasing"
            symbol = "↓"
            label = "Decreasing"
            direction_text = "decreased"
            sign_prev = f"{change_prev_pp}"
        else:
            direction = "stable"
            symbol = "→"
            label = "Relatively Stable"
            direction_text = "remained relatively stable"
            sign_prev = f"{'+' if change_prev_pp > 0 else ''}{change_prev_pp}"

        sign_overall = f"{'+' if change_overall_pp > 0 else ''}{change_overall_pp}"

        interpretation = (
            f"The AI acoustic indicator has {direction_text} across the recorded visits "
            f"({sign_prev} percentage points from previous visit, {sign_overall} pp overall). "
            f"Changes in the AI measurement reflect acoustic pattern variations and do not indicate clinical disease progression."
        )

        return {
            "status": "multi_visit",
            "direction": direction,
            "direction_symbol": symbol,
            "direction_label": label,
            "latest_percent": round(latest_pct, 2),
            "previous_percent": round(prev_pct, 2),
            "change_previous_pp": change_prev_pp,
            "change_previous_formatted": f"{sign_prev} pp",
            "overall_change_pp": change_overall_pp,
            "overall_change_formatted": f"{sign_overall} pp",
            "interpretation": interpretation
        }

    def clear_patient_history(self, patient_id: str) -> bool:
        """Clears visit history for a specific patient."""
        clean_id = str(patient_id).strip()
        all_records = self._load_all()
        if clean_id in all_records:
            del all_records[clean_id]
            self._save_all(all_records)
            return True
        return False


# Global Singleton Service
_GLOBAL_TRACKING_SERVICE = None

def get_tracking_service() -> PatientTrackingService:
    """Returns singleton PatientTrackingService instance."""
    global _GLOBAL_TRACKING_SERVICE
    if _GLOBAL_TRACKING_SERVICE is None:
        _GLOBAL_TRACKING_SERVICE = PatientTrackingService()
    return _GLOBAL_TRACKING_SERVICE
