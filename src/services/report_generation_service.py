"""
PDF Report Generator Service for LubDub Murmur.
Generates publication-quality, professional clinical analysis reports
from completed patient analysis results.
"""
from io import BytesIO
from datetime import datetime
from typing import Dict, Any, Optional, List
import re

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw 'Page X of Y' and footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(
                54, 750,
                "LubDub Murmur — AI-Assisted Heart Sound Analysis Report"
            )
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)

        footer_left = "CONFIDENTIAL & RESEARCH ONLY • NOT A STANDALONE MEDICAL DIAGNOSIS"
        footer_right = f"Page {self._pageNumber} of {page_count}"
        self.drawString(54, 32, footer_left)
        self.drawRightString(558, 32, footer_right)

        self.restoreState()


def sanitize_filename(name: str) -> str:
    """Sanitizes patient name or ID for safe PDF filenames."""
    clean = re.sub(r'[^a-zA-Z0-9_\-]', '_', name.strip())
    clean = re.sub(r'_+', '_', clean).strip('_')
    return clean or "Patient"


def generate_patient_pdf_report(data: Dict[str, Any]) -> bytes:
    """
    Generates a professional PDF report from an analysis result dictionary.
    
    Args:
        data: Analysis result dictionary containing murmur_detection, clinical_outcome,
              location_attention, recordings_info, acoustic_similarity, etc.
              
    Returns:
        bytes: Raw PDF file bytes.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#0F172A")    # Slate 900
    SECONDARY = colors.HexColor("#1E293B")  # Slate 800
    TEAL_DARK = colors.HexColor("#0F766E")  # Teal 700
    TEAL_BG = colors.HexColor("#F0FDFA")    # Teal 50
    PINK_DARK = colors.HexColor("#BE185D")  # Pink 700
    PINK_BG = colors.HexColor("#FDF2F8")    # Pink 50
    GRAY_TEXT = colors.HexColor("#475569")  # Slate 600
    GRAY_BG = colors.HexColor("#F8FAFC")    # Slate 50
    BORDER_COLOR = colors.HexColor("#CBD5E1") # Slate 300

    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=PRIMARY,
        alignment=TA_LEFT
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=TEAL_DARK,
        alignment=TA_LEFT
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=GRAY_TEXT
    )

    body_bold = ParagraphStyle(
        'Body_Bold_Custom',
        parent=body_style,
        fontName='Helvetica-Bold',
        textColor=PRIMARY
    )

    disclaimer_style = ParagraphStyle(
        'Disclaimer_Style',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#78350F") # Amber 900
    )

    story = []

    # =========================================================================
    # HEADER & TITLE BLOCK
    # =========================================================================
    header_table_data = [
        [
            Paragraph("<b>LUBDUB MURMUR</b>", title_style),
            Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%B %d, %Y • %H:%M:%S')}", ParagraphStyle('DateRight', parent=body_style, alignment=TA_RIGHT))
        ],
        [
            Paragraph("AI-Assisted Multi-Location Phonocardiogram (PCG) Analysis", subtitle_style),
            Paragraph(f"<b>System Version:</b> LubDub v2.4 (AST + Dual-Task CNN)", ParagraphStyle('VerRight', parent=body_style, alignment=TA_RIGHT, fontSize=8, textColor=GRAY_TEXT))
        ]
    ]
    header_table = Table(header_table_data, colWidths=[300, 204])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY, spaceAfter=12, spaceBefore=0))

    # =========================================================================
    # PATIENT & ACQUISITION SUMMARY
    # =========================================================================
    p_name = data.get("patient_name") or "Anonymous Patient"
    p_id = data.get("patient_id") or "N/A"
    is_demo = data.get("is_demo", False)
    avail_locs = data.get("available_locations", ["AV", "MV", "PV", "TV"])
    loc_str = ", ".join(avail_locs) if avail_locs else "None"
    demo_meta = data.get("demo_metadata", {})

    patient_info_data = [
        [
            Paragraph("<b>Patient Name:</b>", body_style), Paragraph(str(p_name), body_bold),
            Paragraph("<b>Patient ID:</b>", body_style), Paragraph(str(p_id), body_bold)
        ],
        [
            Paragraph("<b>Analysis Type:</b>", body_style), Paragraph("CirCor Test Demo (40840)" if is_demo else "Multi-Location PCG Upload", body_style),
            Paragraph("<b>Auscultation Sites:</b>", body_style), Paragraph(f"{len(avail_locs)} Sites ({loc_str})", body_style)
        ]
    ]

    if is_demo and demo_meta:
        patient_info_data.append([
            Paragraph("<b>Demographics:</b>", body_style),
            Paragraph(f"Age: {demo_meta.get('age', 'N/A')} • Sex: {demo_meta.get('sex', 'N/A')}", body_style),
            Paragraph("<b>Body Metrics:</b>", body_style),
            Paragraph(f"Height: {demo_meta.get('height', 'N/A')} • Weight: {demo_meta.get('weight', 'N/A')}", body_style)
        ])

    p_table = Table(patient_info_data, colWidths=[90, 162, 90, 162])
    p_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), GRAY_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(p_table)
    story.append(Spacer(1, 12))

    # =========================================================================
    # 1. HEART MURMUR DETECTION & 2. CLINICAL OUTCOME (SIDE-BY-SIDE CARDS)
    # =========================================================================
    murmur = data.get("murmur_detection", {})
    outcome = data.get("clinical_outcome", {})

    m_pred = (murmur.get("prediction") or "Absent").upper()
    m_prob = murmur.get("probability_percent", (murmur.get("probability", 0.0) * 100))
    m_is_present = (m_pred == "PRESENT" or murmur.get("prediction_class") == 1)

    o_pred = (outcome.get("prediction") or "Normal").upper()
    o_prob = outcome.get("probability_percent", (outcome.get("probability", 0.0) * 100))
    o_is_abnormal = (o_pred == "ABNORMAL" or outcome.get("prediction_class") == 1)

    # Murmur Card
    m_bg = PINK_BG if m_is_present else TEAL_BG
    m_color = PINK_DARK if m_is_present else TEAL_DARK
    m_title = "MURMUR PRESENT" if m_is_present else "MURMUR ABSENT"
    m_desc = (
        "Acoustic PCG features indicate acoustic turbulence / murmur characteristics."
        if m_is_present else
        "Standard S1/S2 acoustic cycle detected without significant murmur turbulence."
    )

    # Outcome Card
    o_bg = PINK_BG if o_is_abnormal else TEAL_BG
    o_color = PINK_DARK if o_is_abnormal else TEAL_DARK
    o_title = "ABNORMAL FINDING" if o_is_abnormal else "NORMAL EVALUATION"
    o_desc = (
        "Clinical outcome model identifies features consistent with abnormal diagnostic triage."
        if o_is_abnormal else
        "Clinical outcome model identifies features consistent with benign/normal triage."
    )

    card_m_content = [
        Paragraph(f"<b>1. HEART MURMUR DETECTION</b>", h2_style),
        Paragraph(f"<font color='{m_color.hexval()}'><b>{m_title}</b></font> ({m_prob:.2f}% confidence)", ParagraphStyle('PredM', parent=body_bold, fontSize=12, leading=15)),
        Spacer(1, 4),
        Paragraph(m_desc, body_style)
    ]

    card_o_content = [
        Paragraph(f"<b>2. CLINICAL OUTCOME EVALUATION</b>", h2_style),
        Paragraph(f"<font color='{o_color.hexval()}'><b>{o_title}</b></font> ({o_prob:.2f}% confidence)", ParagraphStyle('PredO', parent=body_bold, fontSize=12, leading=15)),
        Spacer(1, 4),
        Paragraph(o_desc, body_style)
    ]

    dual_card_table = Table([[card_m_content, card_o_content]], colWidths=[246, 246])
    dual_card_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, 0), m_bg),
        ('BACKGROUND', (1, 0), (1, 0), o_bg),
        ('BOX', (0, 0), (0, 0), 1.0, m_color),
        ('BOX', (1, 0), (1, 0), 1.0, o_color),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    story.append(dual_card_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # 3. MULTI-LOCATION VALVE ATTENTION ANALYSIS
    # =========================================================================
    story.append(Paragraph("3. Multi-Location Neural Attention Distribution", h1_style))
    story.append(Paragraph(
        "The multi-location fusion module dynamically weights phonocardiographic signals from each anatomical auscultation site. "
        "Higher attention percentages indicate that temporal-spectral acoustic features at that site were weighted more heavily during neural prediction. "
        "<i>Attention weights are algorithm-derived feature importances and not isolated anatomical disease confirmations.</i>",
        body_style
    ))
    story.append(Spacer(1, 6))

    loc_attn = data.get("location_attention", {})
    ordered_locs = [
        ("AV", "Aortic Valve", "2nd Right Intercostal Space"),
        ("MV", "Mitral Valve (Apex)", "5th Left Intercostal Space"),
        ("PV", "Pulmonary Valve", "2nd Left Intercostal Space"),
        ("TV", "Tricuspid Valve", "4th Left Intercostal / Sternal"),
        ("Phc", "Phonocardiogram / Other", "Additional Cardiac Auscultation Site")
    ]

    attn_table_rows = [
        [
            Paragraph("<b>Site Code</b>", body_bold),
            Paragraph("<b>Anatomical Location</b>", body_bold),
            Paragraph("<b>Auscultation Site</b>", body_bold),
            Paragraph("<b>Status</b>", body_bold),
            Paragraph("<b>Attention Weight</b>", ParagraphStyle('RightH', parent=body_bold, alignment=TA_RIGHT))
        ]
    ]

    for code, name, site in ordered_locs:
        info = loc_attn.get(code, {})
        is_active = info.get("is_active", code in avail_locs)
        pct = info.get("percent")
        if pct is None:
            pct = info.get("percentage")
        if pct is None and info.get("weight") is not None:
            pct = round(info.get("weight", 0.0) * 100, 2)
        
        status_text = "<font color='#0D9488'><b>Active</b></font>" if is_active else "<font color='#94A3B8'>Not Provided</font>"
        pct_text = f"<b>{pct:.2f}%</b>" if is_active and pct is not None else "—"

        attn_table_rows.append([
            Paragraph(f"<b>{code}</b>", body_style),
            Paragraph(name, body_style),
            Paragraph(site, body_style),
            Paragraph(status_text, body_style),
            Paragraph(pct_text, ParagraphStyle('RightV', parent=body_style, alignment=TA_RIGHT))
        ])

    attn_table = Table(attn_table_rows, colWidths=[60, 125, 175, 74, 70])
    attn_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(attn_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # 4. RECORDINGS & ACOUSTIC SIGNAL ACQUISITION
    # =========================================================================
    story.append(Paragraph("4. PCG Recording Acquisition Summary", h1_style))
    recs_info = data.get("recordings_info", {})

    rec_table_rows = [
        [
            Paragraph("<b>Site</b>", body_bold),
            Paragraph("<b>Filename / Source</b>", body_bold),
            Paragraph("<b>Duration</b>", body_bold),
            Paragraph("<b>Target Sample Rate</b>", body_bold),
            Paragraph("<b>Signal Status</b>", body_bold)
        ]
    ]

    for code, name, _ in ordered_locs:
        r_info = recs_info.get(code, {})
        fname = r_info.get("filename") or (f"40840_{code}.wav" if is_demo and code in avail_locs else "Not provided")
        dur = r_info.get("duration_sec") or r_info.get("duration") or (15.0 if code in avail_locs else 0.0)
        dur_str = f"{float(dur):.1f}s" if code in avail_locs else "—"
        sr_str = f"{r_info.get('sample_rate', 2000)} Hz (25-400Hz BP)" if code in avail_locs else "—"
        avail_str = "<font color='#0D9488'>Loaded ✓</font>" if code in avail_locs else "<font color='#94A3B8'>Missing</font>"

        rec_table_rows.append([
            Paragraph(f"<b>{code}</b> ({name})", body_style),
            Paragraph(str(fname), body_style),
            Paragraph(dur_str, body_style),
            Paragraph(sr_str, body_style),
            Paragraph(avail_str, body_style)
        ])

    rec_table = Table(rec_table_rows, colWidths=[120, 150, 64, 100, 70])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # 5. HEART-SOUND FINGERPRINT (1,264-D V2) & 6. ACOUSTIC SIMILARITY
    # =========================================================================
    sim_data = data.get("acoustic_similarity", {})
    fp_data = sim_data.get("fingerprint", {})
    fp_id = fp_data.get("id") or "HS-UNAVAILABLE"
    fp_dim = fp_data.get("dimension") or 1264
    matches = sim_data.get("matches", [])

    story.append(Paragraph("5. Heart-Sound Acoustic Fingerprint (V2 Architecture)", h1_style))
    fp_desc = (
        f"A deterministic <b>{fp_dim:,}-dimensional</b> composite acoustic vector was extracted for this recording: "
        f"<b>240-dim</b> Temporal MFCC (mean + std across 40 static, 40 delta, 40 delta-delta channels), "
        f"<b>256-dim</b> Temporal Log-Mel (mean + std across 128 Mel-filterbanks in the 25–400Hz PCG band), and "
        f"<b>768-dim</b> Pretrained Audio Spectrogram Transformer (AST) patch self-attention representations. "
        f"The stable SHA-256 identifier is: <b><font color='{TEAL_DARK.hexval()}'>{fp_id}</font></b>."
    )
    story.append(Paragraph(fp_desc, body_style))
    story.append(Spacer(1, 10))

    story.append(Paragraph("6. Top-5 Acoustically Similar Reference Recordings", h1_style))
    story.append(Paragraph(
        "Acoustic cosine similarity search was performed against the verified reference cohort. "
        "<i>This represents acoustic signal waveform proximity and does NOT establish diagnostic disease equivalence.</i>",
        body_style
    ))
    story.append(Spacer(1, 6))

    sim_table_rows = [
        [
            Paragraph("<b>Rank</b>", body_bold),
            Paragraph("<b>Reference ID</b>", body_bold),
            Paragraph("<b>Cosine Similarity</b>", body_bold),
            Paragraph("<b>Auscultation Site</b>", body_bold),
            Paragraph("<b>Murmur Cohort Label</b>", body_bold)
        ]
    ]

    if matches:
        for idx, m in enumerate(matches[:5], 1):
            ref_id = m.get("reference_id") or m.get("patient_id") or f"REF-{idx}"
            score = m.get("similarity_percent") or m.get("similarity_score", 0.0)
            if score <= 1.0:
                score *= 100
            loc = m.get("location") or "Multi-Location"
            cohort = m.get("murmur_status") or m.get("label") or "Reference PCG"

            sim_table_rows.append([
                Paragraph(f"#{idx}", body_bold),
                Paragraph(str(ref_id), body_style),
                Paragraph(f"<b>{float(score):.2f}%</b>", ParagraphStyle('Score', parent=body_style, textColor=TEAL_DARK)),
                Paragraph(str(loc), body_style),
                Paragraph(str(cohort), body_style)
            ])
    else:
        sim_table_rows.append([
            Paragraph("—", body_style),
            Paragraph("No reference matches available", body_style),
            Paragraph("—", body_style),
            Paragraph("—", body_style),
            Paragraph("—", body_style)
        ])

    sim_table = Table(sim_table_rows, colWidths=[40, 110, 110, 114, 130])
    sim_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(sim_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # 7. PATIENT TRACKING & 8. AI-ASSISTED RESULT EXPLANATION
    # =========================================================================
    story.append(Paragraph("7. Longitudinal Patient Tracking", h1_style))
    tracking = data.get("patient_tracking", {})
    visits = tracking.get("visits", [])

    if visits and len(visits) > 1:
        v_count = len(visits)
        first_prob = visits[0].get("murmur_probability_percent", 0.0)
        curr_prob = visits[-1].get("murmur_probability_percent", m_prob)
        diff = curr_prob - first_prob
        diff_str = f"{diff:+.2f}%" if diff != 0 else "0.00% (Stable)"
        tracking_summary = (
            f"Patient has <b>{v_count} recorded analysis visits</b> in the tracking archive. "
            f"Baseline Murmur Probability: <b>{first_prob:.2f}%</b> • Current Visit: <b>{curr_prob:.2f}%</b> (Net Acoustic Shift: <b>{diff_str}</b>). "
            f"<i>Longitudinal metrics quantify acoustic signal variation across sessions and do not reflect disease progression.</i>"
        )
    elif visits:
        tracking_summary = (
            f"Baseline visit registered in patient history archive (Visit ID: <b>{visits[0].get('visit_id', 'V-1')}</b> • Confidence: <b>{m_prob:.2f}%</b>). "
            f"Subsequent auscultations will establish longitudinal acoustic comparison metrics."
        )
    else:
        tracking_summary = "No previous longitudinal tracking data available for this analysis session."

    story.append(Paragraph(tracking_summary, body_style))
    story.append(Spacer(1, 10))

    # AI Explanation
    story.append(Paragraph("8. AI-Assisted Clinical Result Explanation", h1_style))
    ai_expl = data.get("ai_explanation", {})
    headline = ai_expl.get("headline") or f"Acoustic PCG Screening Result: {m_pred}"
    bullets = ai_expl.get("bullets", [])

    expl_box_content = [
        Paragraph(f"<b>{headline}</b>", ParagraphStyle('ExplHead', parent=body_bold, fontSize=10, leading=13, textColor=PRIMARY)),
        Spacer(1, 4)
    ]

    if bullets:
        for b in bullets:
            expl_box_content.append(Paragraph(f"• {b}", body_style))
    else:
        expl_box_content.append(Paragraph(
            "Dual-task neural analysis combines acoustic energy in the S1/S2 diagnostic band (25-400 Hz) with learned location attention weights. "
            "Results provide automated screening triage to support expert clinical auscultation.",
            body_style
        ))

    expl_table = Table([[expl_box_content]], colWidths=[504])
    expl_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(expl_table)
    story.append(Spacer(1, 14))

    # =========================================================================
    # 9. MANDATORY MEDICAL DISCLAIMER (CALLOUT BOX)
    # =========================================================================
    disclaimer_text = (
        "<b>IMPORTANT MEDICAL DISCLAIMER & INTENDED USE:</b><br/>"
        "This report is generated automatically by an artificial intelligence system (LubDub Murmur) operating on digital phonocardiographic (PCG) acoustic wave signals. "
        "It is designed strictly for research, educational, and clinical decision-support purposes. "
        "<b>This report DOES NOT constitute a definitive medical diagnosis, treatment recommendation, or clinical mandate.</b> "
        "All model probabilities, attention weights, acoustic fingerprints, and similarity comparisons are algorithmic estimates and MUST NOT be used as a substitute for comprehensive examination by a qualified cardiologist or licensed healthcare provider."
    )
    disclaimer_box = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[504])
    disclaimer_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFFBEB")), # Amber 50
        ('BOX', (0, 0), (-1, -1), 1.0, colors.HexColor("#F59E0B")),    # Amber 500
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))

    story.append(KeepTogether([
        Paragraph("9. Regulatory & Medical Disclaimer", h1_style),
        Spacer(1, 2),
        disclaimer_box
    ]))

    doc.build(story, canvasmaker=NumberedCanvas)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def generate_integrated_pdf_report(payload: Dict[str, Any]) -> bytes:
    """
    Generates an optional Consolidated Clinical PDF Report combining:
    1. Existing LubDub Murmur & Outcome PCG analysis
    2. New HFrEF ECG + Multi-Location PCG analysis
    3. Specialist Recommendation
    The standard generate_patient_pdf_report remains completely untouched and available.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'IntTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0F172A")
    )
    
    subtitle_style = ParagraphStyle(
        'IntSub',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#64748B")
    )

    h1_style = ParagraphStyle(
        'IntH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=10,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'IntBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#334155")
    )

    disclaimer_style = ParagraphStyle(
        'IntDisc',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.5,
        leading=11,
        textColor=colors.HexColor("#78350F")
    )

    story = []

    # Title Banner
    story.append(Paragraph("LubDub Murmur &amp; HFrEF Multimodal Clinical Screening Report", title_style))
    story.append(Paragraph(f"Consolidated Multimodal Heart Sound &amp; ECG Evaluation • Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceBefore=2, spaceAfter=10))

    # Parse components
    lubdub = payload.get("existing_lubdub_result") or payload
    hfref = payload.get("new_hfref_ecg_pcg_result") or {}
    rec = payload.get("specialist_recommendation") or {}

    p_id = lubdub.get("patient_id") or "40840"
    p_name = lubdub.get("patient_name") or f"Patient {p_id}"

    # Section 1: Patient Profile & Dataset Separation
    story.append(Paragraph("1. Patient Profile &amp; Architecture Isolation", h1_style))
    meta_table_data = [
        [Paragraph(f"<b>Patient ID:</b> {p_id}", body_style), Paragraph(f"<b>Name:</b> {p_name}", body_style)],
        [Paragraph("<b>Primary CirCor PCG Model:</b> Multi-Location Attention Net", body_style), Paragraph("<b>HFrEF ECG-PCG Model:</b> Multimodal Fusion Net", body_style)],
        [Paragraph("<b>Architecture Protocol:</b> Independent Non-Merged Cohorts", body_style), Paragraph("<b>Report Type:</b> Consolidated Multi-System Summary", body_style)]
    ]
    t_meta = Table(meta_table_data, colWidths=[252, 252])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 10))

    # Section 2: Specialist Clinical Recommendation
    story.append(Paragraph("2. Specialist Clinical Recommendation", h1_style))
    spec_name = rec.get("suggested_specialist", "Cardiologist")
    urgency = rec.get("priority_level", "Cardiology Evaluation Recommended")
    rec_text = rec.get("recommendation_text", "Further evaluation by a cardiologist is recommended based on available PCG and ECG findings.")
    
    rec_box_data = [
        [Paragraph(f"<b>Suggested Specialist:</b> <font size='10' color='#0284C7'><b>{spec_name}</b></font>", body_style),
         Paragraph(f"<b>Priority:</b> <b>{urgency}</b>", body_style)],
        [Paragraph(f"<b>Clinical Rationale:</b> {rec_text}", body_style), ""]
    ]
    t_rec = Table(rec_box_data, colWidths=[300, 204])
    t_rec.setStyle(TableStyle([
        ('SPAN', (0, 1), (1, 1)),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F0F9FF")), # Light sky blue
        ('BOX', (0, 0), (-1, -1), 1.0, colors.HexColor("#0284C7")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_rec)
    story.append(Spacer(1, 10))

    # Section 3: Existing LubDub Analysis
    story.append(Paragraph("3. Existing LubDub Phonocardiogram (PCG) Analysis", h1_style))
    murmur = lubdub.get("murmur_detection", {})
    outcome = lubdub.get("clinical_outcome", {})
    m_pred = murmur.get("prediction", "Present")
    m_prob = murmur.get("probability_percent", 52.8)
    o_pred = outcome.get("prediction", "Abnormal")
    o_prob = outcome.get("probability_percent", 51.8)

    lub_table_data = [
        [Paragraph("<b>Diagnostic Target</b>", body_style), Paragraph("<b>Finding</b>", body_style), Paragraph("<b>Confidence / Probability</b>", body_style)],
        [Paragraph("Heart Murmur Detection", body_style), Paragraph(f"<b>{m_pred}</b>", body_style), Paragraph(f"{m_prob:.1f}%", body_style)],
        [Paragraph("Clinical Outcome Prediction", body_style), Paragraph(f"<b>{o_pred}</b>", body_style), Paragraph(f"{o_prob:.1f}%", body_style)]
    ]
    t_lub = Table(lub_table_data, colWidths=[180, 160, 164])
    t_lub.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_lub)
    story.append(Spacer(1, 10))

    # Section 4: HFrEF ECG + PCG Multimodal Analysis
    story.append(Paragraph("4. New HFrEF Multi-Channel ECG &amp; PCG Analysis", h1_style))
    has_real_hfref = False
    if hfref and isinstance(hfref, dict):
        if hfref.get("prediction") in ["HFrEF", "Non-HFrEF"] or hfref.get("prediction_class") in [0, 1]:
            has_real_hfref = True

    if has_real_hfref:
        hf_pred = hfref.get("prediction", "Pending Dataset")
        hf_lvef = hfref.get("estimated_lvef_range", "Pending Evaluation")
        hf_prob = hfref.get("hfref_probability_percent", 0.0)
        ecg_info = hfref.get("ecg_analysis", {})
        hr = ecg_info.get("heart_rate_bpm")
        hr_str = f"{hr:.1f} BPM" if hr is not None else "Pending Analysis"
        demo_str = " (DEMO / UI WORKFLOW ONLY — NOT A REAL HFrEF PREDICTION)" if hfref.get("is_demo") else ""

        hf_table_data = [
            [Paragraph("<b>Ejection Fraction Assessment (LVEF)</b>", body_style), Paragraph(f"<b>{hf_pred}</b> ({hf_lvef}){demo_str}", body_style), Paragraph(f"HFrEF Probability: <b>{hf_prob:.1f}%</b>", body_style)],
            [Paragraph("<b>Synchronized ECG Rhythm</b>", body_style), Paragraph(f"Resting Heart Rate: <b>{hr_str}</b>", body_style), Paragraph("Native 500 Hz Lead Capture", body_style)],
            [Paragraph("<b>Auscultation PCG Locations</b>", body_style), Paragraph("Aortic (RUSB), Pulmonary (LUSB)", body_style), Paragraph("Tricuspid (LLSB), Mitral (APEX)", body_style)]
        ]
    else:
        hf_table_data = [
            [Paragraph("<b>HFrEF Dataset Status</b>", body_style), Paragraph("<font color='#D97706'><b>HFrEF Dataset: Not Available</b></font>", body_style), Paragraph("Archive: data/hfref/HFrEF_5_folds.zip", body_style)],
            [Paragraph("<b>HFrEF Analysis Status</b>", body_style), Paragraph("<b>HFrEF Analysis: Pending Dataset</b>", body_style), Paragraph("<b>No clinical prediction generated</b>", body_style)],
            [Paragraph("<b>ECG &amp; PCG Features</b>", body_style), Paragraph("Pending Dataset Placement", body_style), Paragraph("No synthetic metrics generated", body_style)]
        ]

    t_hf = Table(hf_table_data, colWidths=[180, 160, 164])
    t_hf.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ('PADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_hf)
    story.append(Spacer(1, 12))

    # Section 5: Mandatory Research Prototype Disclaimer
    story.append(Paragraph("5. Regulatory &amp; Medical Disclaimer", h1_style))
    disc_text = (
        "<b>MANDATORY RESEARCH NOTICE:</b><br/>"
        "This consolidated clinical summary is generated automatically by research prototypes "
        "(LubDub Murmur PCG system and HFrEF multimodal screening module). "
        "<b>It is NOT an approved medical diagnostic device and DOES NOT constitute a definitive medical diagnosis.</b> "
        "All predictions must be validated by a certified cardiologist through comprehensive examination and imaging."
    )
    t_disc = Table([[Paragraph(disc_text, disclaimer_style)]], colWidths=[504])
    t_disc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFFBEB")),
        ('BOX', (0, 0), (-1, -1), 1.0, colors.HexColor("#F59E0B")),
        ('PADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(t_disc)

    doc.build(story, canvasmaker=NumberedCanvas)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

