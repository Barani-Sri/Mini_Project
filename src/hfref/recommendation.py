"""
Specialist Clinical Recommendation Module.
Provides rule-grounded, non-diagnostic specialist referral suggestions
based on findings from ECG + Multi-Location PCG analysis and existing LubDub Murmur analysis.
Clearly frames all suggestions as decision support for research prototypes, not medical diagnoses.
"""
from typing import Dict, Optional, Any, List


class SpecialistRecommender:
    """
    Evaluates multimodality findings and produces calibrated specialist recommendations.
    """

    def generate_recommendation(
        self,
        hfref_result: Optional[Dict[str, Any]] = None,
        lubdub_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes recommendations from both LubDub (CirCor PCG) and HFrEF (ECG + PCG) analyses.
        """
        # Determine if real HFrEF analysis is available
        has_real_hfref = False
        is_hfref_positive = False
        hfref_prob = 0.0

        if hfref_result and isinstance(hfref_result, dict):
            # Check if prediction is present and not pending
            pred = hfref_result.get("prediction")
            pred_class = hfref_result.get("prediction_class")
            if pred in ["HFrEF", "Non-HFrEF"] or pred_class in [0, 1]:
                has_real_hfref = True
                is_hfref_positive = (pred_class == 1 or pred == "HFrEF")
                hfref_prob = float(hfref_result.get("hfref_probability", 0.0))

        # Parse LubDub findings if present
        murmur_detected = False
        outcome_abnormal = False
        if lubdub_result:
            m_det = lubdub_result.get("murmur_detection", {})
            c_out = lubdub_result.get("clinical_outcome", {})
            murmur_detected = (m_det.get("prediction") == "Present" or m_det.get("prediction_class") == 1)
            outcome_abnormal = (c_out.get("prediction") == "Abnormal" or c_out.get("prediction_class") == 1)

        # Decision matrix for specialist recommendation
        if has_real_hfref:
            if is_hfref_positive and murmur_detected:
                specialist = "Cardiologist / Heart Failure Specialist"
                department = "Cardiology & Heart Failure Center"
                urgency = "Priority Cardiac Evaluation"
                urgency_class = "urgent"
                action_summary = (
                    "Based on the available acoustic and multimodal analysis, further evaluation by a cardiologist is recommended. "
                    "Both acoustic murmur patterns and electrical-acoustic signatures consistent with "
                    "reduced left ventricular ejection fraction (LVEF <= 40%) were identified."
                )
                next_steps = [
                    "Comprehensive Transthoracic Echocardiogram (TTE) with Doppler",
                    "Standard 12-lead diagnostic electrocardiogram (ECG)",
                    "Clinical serum BNP / NT-proBNP biomarker assessment",
                    "Specialist cardiology consultation for combined valvular and ventricular assessment"
                ]
            elif is_hfref_positive:
                specialist = "Cardiologist"
                department = "Adult Cardiology Clinic"
                urgency = "Prompt Cardiac Evaluation"
                urgency_class = "warning"
                action_summary = (
                    "Based on the available analysis, further evaluation by a cardiologist is recommended. "
                    "Multi-channel ECG rhythm and multi-location phonocardiogram analysis suggests "
                    "patterns associated with reduced left ventricular systolic function."
                )
                next_steps = [
                    "Transthoracic Echocardiogram (TTE) to confirm left ventricular ejection fraction",
                    "Clinical history assessment for dyspnea, fatigue, or orthopnea",
                    "Review of cardiac medications and blood pressure management"
                ]
            elif murmur_detected:
                specialist = "Cardiologist / Valvular Specialist"
                department = "Cardiology Auscultation Clinic"
                urgency = "Cardiology Auscultation Review"
                urgency_class = "warning"
                action_summary = (
                    "Based on the available acoustic analysis, further evaluation by a cardiologist is recommended. "
                    "LubDub acoustic analysis identified significant murmurs across auscultation valve sites, "
                    "while ventricular ejection fraction indicators remain preserved."
                )
                next_steps = [
                    "Detailed cardiac auscultation and Doppler echocardiography",
                    "Auscultation site focal localization review",
                    "Longitudinal acoustic monitoring with LubDub tracking"
                ]
            elif outcome_abnormal:
                specialist = "Cardiologist / General Physician"
                department = "Cardiovascular Medicine"
                urgency = "Clinical Review"
                urgency_class = "info"
                action_summary = (
                    "Clinical consultation with a cardiologist or general physician is recommended "
                    "to review atypical acoustic findings."
                )
                next_steps = [
                    "Routine clinical cardiovascular examination",
                    "Review of patient baseline vital signs and cardiovascular risk profile"
                ]
            else:
                specialist = "General Practitioner / Primary Care Physician"
                department = "Primary Care & Preventive Medicine"
                urgency = "Routine Preventive Follow-up"
                urgency_class = "normal"
                action_summary = (
                    "Based on the available multi-location PCG and ECG analysis, findings are within normal ranges. "
                    "Routine preventive wellness checkups are recommended."
                )
                next_steps = [
                    "Continue standard cardiovascular wellness checkups",
                    "Periodic acoustic screening if any new cardiopulmonary symptoms arise"
                ]
            
            rationale = (
                f"Recommendation derived from multi-location PCG analysis "
                f"(Murmur: {'Detected' if murmur_detected else 'Absent'}) "
                f"and ECG-PCG analysis (HFrEF finding: {'Detected' if is_hfref_positive else 'Preserved'})."
            )
        else:
            # HFrEF analysis is pending/unavailable - base recommendation ONLY on LubDub
            pending_msg = "HFrEF analysis is pending because the required dataset/model is not currently available."
            if murmur_detected:
                specialist = "Cardiologist / Valvular Specialist"
                department = "Cardiology Auscultation Clinic"
                urgency = "Cardiology Auscultation Review"
                urgency_class = "warning"
                action_summary = (
                    f"Based on the available acoustic analysis, further evaluation by a cardiologist is recommended. "
                    f"Significant murmur patterns were identified across auscultation valve sites. "
                    f"{pending_msg}"
                )
                next_steps = [
                    "Detailed cardiac auscultation and Doppler echocardiography",
                    "Auscultation site focal localization review",
                    "Clinical evaluation by a certified medical practitioner"
                ]
            elif outcome_abnormal:
                specialist = "Cardiologist / General Physician"
                department = "Cardiovascular Medicine"
                urgency = "Clinical Review"
                urgency_class = "info"
                action_summary = (
                    f"Based on the available acoustic analysis, further evaluation by a cardiologist or general physician is recommended "
                    f"to review atypical acoustic findings. {pending_msg}"
                )
                next_steps = [
                    "Routine clinical cardiovascular examination",
                    "Review of patient baseline vital signs and cardiovascular risk profile",
                    "Clinical evaluation by a certified medical practitioner"
                ]
            else:
                specialist = "General Practitioner / Primary Care Physician"
                department = "Primary Care & Preventive Medicine"
                urgency = "Routine Preventive Follow-up"
                urgency_class = "normal"
                action_summary = (
                    f"Based on the available acoustic analysis, cardiac sound findings are within normal ranges. "
                    f"Routine preventive wellness checkups are recommended. {pending_msg}"
                )
                next_steps = [
                    "Continue standard cardiovascular wellness checkups",
                    "Periodic acoustic screening if any new cardiopulmonary symptoms arise"
                ]
            
            rationale = (
                f"Recommendation derived from available multi-location PCG acoustic analysis "
                f"(Murmur: {'Detected' if murmur_detected else 'Absent'}). {pending_msg}"
            )

        return {
            "suggested_specialist": specialist,
            "department": department,
            "priority_level": urgency,
            "priority_class": urgency_class,
            "recommendation_text": action_summary,
            "recommended_investigations": next_steps,
            "clinical_rationale": rationale,
            "disclaimer": (
                "Research prototype — not a clinical diagnostic system. "
                "Findings must be evaluated by a certified medical practitioner."
            )
        }


# Global recommender singleton
_RECOMMENDER_INSTANCE: Optional[SpecialistRecommender] = None

def get_specialist_recommender() -> SpecialistRecommender:
    """Returns singleton SpecialistRecommender instance."""
    global _RECOMMENDER_INSTANCE
    if _RECOMMENDER_INSTANCE is None:
        _RECOMMENDER_INSTANCE = SpecialistRecommender()
    return _RECOMMENDER_INSTANCE
