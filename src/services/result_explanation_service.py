"""
LubDub AI — Automated Result Explanation Engine.
Generates patient-friendly, clinically cautious explanations dynamically derived from
the actual ML model predictions, confidence scores, and multi-location acoustic attention.
"""
from typing import Dict, List, Optional, Any, Tuple


def generate_result_explanation(
    murmur_pred: str,
    murmur_prob: float,
    outcome_pred: Optional[str] = None,
    outcome_prob: Optional[float] = None,
    available_locations: Optional[List[str]] = None,
    location_attention: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Generates a structured, patient-accessible explanation of the AI screening results.

    Args:
        murmur_pred: 'Present' or 'Absent' (or 'Murmur Detected' / 'Normal')
        murmur_prob: Probability float in [0.0, 1.0]
        outcome_pred: 'Abnormal' or 'Normal' (optional)
        outcome_prob: Outcome probability float in [0.0, 1.0] (optional)
        available_locations: List of auscultation locations provided (e.g. ['AV', 'MV', 'PV', 'TV'])
        location_attention: Dictionary of location attention weights/percentages

    Returns:
        Structured explanation dictionary with plain-language sections.
    """
    is_murmur = murmur_pred.strip().lower() in ["present", "murmur", "murmur detected", "1", "abnormal"]
    conf_pct = round(murmur_prob * 100, 1) if is_murmur else round((1.0 - murmur_prob) * 100, 1)
    # If murmur_prob was already mapped to class confidence:
    if murmur_prob > 0.5 and not is_murmur:
        conf_pct = round(murmur_prob * 100, 1)
    elif murmur_prob <= 0.5 and is_murmur:
        conf_pct = round(murmur_prob * 100, 1)

    # 1. Headline & Status
    if is_murmur:
        status_title = "Murmur Pattern Detected"
        status_tag = "MURMUR DETECTED"
        icon = "⚠️"
        color_theme = "pink"
        what_this_means = (
            f"The AI detected acoustic sound patterns in your phonocardiogram recording that are "
            f"statistically associated with a heart murmur (model confidence: {conf_pct}%). "
            f"Heart murmurs are sounds—such as whooshing, swishing, or clicking—produced by turbulent "
            f"blood flow inside or near the heart valves."
        )
        medical_context = (
            "This screening result does not identify the specific anatomical cause, valve defect, or "
            "severity of the condition. Murmurs can be 'innocent' (benign, common in children and healthy individuals) "
            "or 'abnormal' (associated with heart valve stenosis, regurgitation, or congenital heart conditions)."
        )
        next_steps = (
            "We recommend scheduling a follow-up consultation with a qualified physician or cardiologist. "
            "They can perform a clinical stethoscope examination and, if warranted, order an echocardiogram (ultrasound of the heart) "
            "to accurately evaluate your cardiac structure and valve function."
        )
    else:
        status_title = "Normal Heart Sound Pattern"
        status_tag = "NORMAL"
        icon = "🟢"
        color_theme = "teal"
        what_this_means = (
            f"The AI did not detect acoustic patterns strongly associated with a heart murmur in this "
            f"recording (model confidence: {conf_pct}%). The detected acoustic signatures reflect typical "
            f"rhythmic 'lub-dub' (S1 and S2) heart valve closure sounds."
        )
        medical_context = (
            "A normal screening result is encouraging, but it does not guarantee the absence of all cardiac "
            "conditions. Some heart conditions do not generate prominent acoustic murmurs or may only be audible "
            "under specific physical conditions or physiological stress."
        )
        next_steps = (
            "Continue regular routine health check-ups. If you ever experience symptoms such as chest pain, "
            "unexplained shortness of breath, dizziness, palpitations, or fainting, always seek prompt medical evaluation "
            "regardless of any screening tool result."
        )

    # 2. Confidence Interpretation
    if conf_pct >= 85:
        conf_desc = (
            f"The model has high statistical confidence ({conf_pct}%) in its acoustic pattern classification "
            f"for this recording. This reflects strong alignment between your audio patterns and the model's trained representations, "
            f"but is not a medical probability of disease."
        )
    elif conf_pct >= 65:
        conf_desc = (
            f"The model has moderate statistical confidence ({conf_pct}%) in this classification. "
            f"Some background acoustic noise or subtle audio variations may be present in the recording."
        )
    else:
        conf_desc = (
            f"The model classified this recording with borderline confidence ({conf_pct}%). "
            f"A repeat recording in a quiet environment or clinical auscultation is recommended for clearer assessment."
        )

    # 3. Location Insights
    loc_insight = "Multi-location valve contribution data is not available for this session."
    if location_attention:
        active_locs = [
            (loc, data.get("percent", 0.0))
            for loc, data in location_attention.items()
            if data.get("is_active", False)
        ]
        if active_locs:
            active_locs.sort(key=lambda x: x[1], reverse=True)
            top_loc, top_pct = active_locs[0]
            loc_names = {
                "AV": "Aortic Valve (AV)",
                "MV": "Mitral Valve (MV)",
                "PV": "Pulmonary Valve (PV)",
                "TV": "Tricuspid Valve (TV)",
                "Phc": "Other Auscultation Site (Phc)"
            }
            top_name = loc_names.get(top_loc, top_loc)
            if len(active_locs) > 1:
                loc_insight = (
                    f"The neural attention mechanism assigned the highest relative acoustic weighting ({top_pct:.1f}%) to the "
                    f"{top_name} recording among the {len(active_locs)} active auscultation locations evaluated."
                )
            else:
                loc_insight = f"Analysis was performed using single-site auscultation at {top_name} (100.0% weighting)."

    # 4. Technical Pipeline Details
    technical_details = (
        f"• Feature Extraction: 120-dimensional representation (60-dim MFCC + 60-dim Log-Mel Spectrogram with Delta/Delta-Delta).\n"
        f"• Neural Architecture: MultiLocationHeartSoundNet (Shared 2D CNN Encoder + Masked Scaled Dot-Product Attention Fusion).\n"
        f"• Multi-Task Heads: Simultaneous optimization for Murmur Detection (Present/Absent) and Clinical Outcome (Normal/Abnormal).\n"
        f"• Location Masking: Zero-padded tensor with dynamic boolean mask vector across AV, MV, PV, TV, and Phc sites."
    )

    return {
        "status_title": status_title,
        "status_tag": status_tag,
        "icon": icon,
        "color_theme": color_theme,
        "confidence_percent": conf_pct,
        "what_this_means": what_this_means,
        "medical_context": medical_context,
        "confidence_explanation": conf_desc,
        "location_insight": loc_insight,
        "technical_details": technical_details,
        "next_steps": next_steps,
        "disclaimer": (
            "IMPORTANT: LubDub Murmur is an AI-assisted screening tool. This is not a medical diagnosis "
            "and does not replace professional clinical evaluation by a physician."
        )
    }

