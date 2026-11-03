"""
LubDub AI — Context-Aware Patient Assistant & Chatbot Engine.
Provides result-aware, medically responsible, conversational explanations of heart sound screening results.
Maintains session-based conversational memory and strictly adheres to clinical safety guardrails.
"""
import os
import re
from typing import Dict, List, Optional, Any


class LubDubChatbotService:
    """
    Session-aware healthcare educational chatbot for LubDub Murmur.
    Answers patient inquiries dynamically grounded in their specific screening result.
    """

    def __init__(self):
        # In-memory session store: session_id -> { "context": dict, "history": list }
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get_or_create_session(self, session_id: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Retrieves or initializes a chat session with the patient's current screening context."""
        if session_id not in self.sessions or context is not None:
            self.sessions[session_id] = {
                "context": context or {},
                "history": []
            }
        elif context is not None:
            self.sessions[session_id]["context"] = context
        return self.sessions[session_id]

    def reset_session(self, session_id: str, new_context: Optional[Dict[str, Any]] = None):
        """Resets the chat history when a new recording/patient is analyzed."""
        self.sessions[session_id] = {
            "context": new_context or {},
            "history": []
        }

    def chat(
        self,
        session_id: str,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Processes a user message and returns a context-aware response.

        Args:
            session_id: Unique identifier for the screening session.
            user_message: Text prompt from the patient.
            context: Current prediction context (prediction, confidence, locations, etc.)

        Returns:
            Dictionary containing 'response', 'session_id', 'suggested_questions', and 'safety_flag'.
        """
        session = self.get_or_create_session(session_id, context)
        if context:
            session["context"].update(context)

        current_ctx = session["context"]
        history = session["history"]

        # Append user message to history
        history.append({"role": "user", "content": user_message})

        # Generate intelligent response
        bot_response, safety_flag = self._generate_response(user_message, current_ctx, history)

        # Append assistant response to history
        history.append({"role": "assistant", "content": bot_response})

        # Trim history if excessively long
        if len(history) > 20:
            session["history"] = history[-20:]

        suggested = self._get_suggested_followups(user_message, current_ctx)

        return {
            "response": bot_response,
            "session_id": session_id,
            "suggested_questions": suggested,
            "safety_flag": safety_flag,
            "context_summary": {
                "prediction": current_ctx.get("prediction", "Unknown"),
                "confidence": current_ctx.get("confidence", 0.0)
            }
        }

    def _generate_response(
        self,
        query: str,
        ctx: Dict[str, Any],
        history: List[Dict[str, str]]
    ) -> (str, bool):
        """Generates a safe, context-aware answer grounded in the patient's result."""
        q_lower = query.lower().strip()

        # Extract result context
        pred = str(ctx.get("prediction", "UNKNOWN")).upper()
        conf = ctx.get("confidence", None)
        if conf is None:
            conf_pct_str = "80%"
            conf_num = 80.0
        else:
            conf_num = float(conf) * 100 if float(conf) <= 1.0 else float(conf)
            conf_pct_str = f"{conf_num:.1f}%"

        is_murmur = pred in ["MURMUR", "PRESENT", "MURMUR DETECTED", "ABNORMAL"]
        is_normal = pred in ["NORMAL", "ABSENT", "NO MURMUR DETECTED"]

        # ----------------------------------------------------------------------
        # 1. EMERGENCY / RED-FLAG SYMPTOM DETECTION (Highest Priority Safety Rule)
        # ----------------------------------------------------------------------
        red_flags = [
            "chest pain", "faint", "passed out", "syncope", "loss of consciousness",
            "severe shortness of breath", "can't breathe", "cannot breathe",
            "suffocating", "blue lips", "crushing pressure", "emergency", "dying"
        ]
        if any(rf in q_lower for rf in red_flags):
            response = (
                "🚨 **URGENT MEDICAL NOTICE:** If you or someone you are with is experiencing severe symptoms "
                "such as chest pain, pressure or squeezing, fainting, or severe shortness of breath, please call "
                "emergency services (such as 911 or your local emergency number) or go to the nearest emergency room immediately. "
                "Do not wait for AI screening results or online advice when acute cardiac symptoms are present."
            )
            return response, True

        # ----------------------------------------------------------------------
        # 2. DIAGNOSIS & PRESCRIPTION SAFETY GUARDRAIL
        # ----------------------------------------------------------------------
        if any(w in q_lower for w in ["diagnose", "diagnosis", "do i have a disease", "do i have heart failure", "am i sick"]):
            response = (
                "**LubDub Murmur is an AI screening tool, not a diagnostic medical device.**\n\n"
                "The AI analyzes acoustic patterns from heart sound recordings to flag potential murmurs, but it cannot "
                "diagnose underlying heart disease, determine the severity, or identify the specific valve pathology. "
                "Only a qualified physician or cardiologist can provide a medical diagnosis after performing a physical "
                "examination, reviewing your medical history, and conducting tests such as an echocardiogram (cardiac ultrasound)."
            )
            return response, False

        if any(w in q_lower for w in ["medicine", "medication", "pill", "prescription", "drug", "dose", "dosage"]):
            response = (
                "I cannot recommend, prescribe, or alter any medications. Any decisions regarding medications "
                "or treatments must be made in consultation with a licensed healthcare professional who has reviewed your "
                "complete health profile."
            )
            return response, False

        # ----------------------------------------------------------------------
        # 3. RESULT-SPECIFIC: "WHY IS MY RESULT NORMAL / MURMUR?"
        # ----------------------------------------------------------------------
        if ("why" in q_lower and ("normal" in q_lower or "result" in q_lower or "murmur" in q_lower or "absent" in q_lower)) or \
           ("what does my result mean" in q_lower) or ("explain my result" in q_lower) or ("what does this mean" in q_lower):
            if is_normal:
                response = (
                    f"Your heart sound recording was classified as **NORMAL** with a model confidence of **{conf_pct_str}**.\n\n"
                    f"**Why?** The AI analyzed the acoustic frequencies and temporal dynamics of your heart sounds (the standard "
                    f"'lub-dub' valve closure sounds) and did not detect significant acoustic turbulence or murmur patterns.\n\n"
                    f"**Important Context:** An AI screening result indicates that your recording matches healthy sound patterns, "
                    f"but it is a screening assessment and does not replace regular clinical check-ups."
                )
            elif is_murmur:
                response = (
                    f"Your heart sound recording was classified as **MURMUR DETECTED** with a model confidence of **{conf_pct_str}**.\n\n"
                    f"**Why?** The neural network detected acoustic vibrations and frequency patterns in your phonocardiogram "
                    f"that correspond to turbulent blood flow during the cardiac cycle (such as systolic or diastolic murmurs).\n\n"
                    f"**Important Context:** This screening result does not identify the cause or severity. Many heart murmurs are "
                    f"benign ('innocent'), while others may warrant clinical evaluation. We recommend discussing this result with a doctor."
                )
            else:
                response = (
                    f"Your heart sound recording was analyzed with an AI confidence score of **{conf_pct_str}**.\n\n"
                    f"The AI evaluates the acoustic envelope and spectral features across your auscultation sites to differentiate "
                    f"standard valve closure sounds from turbulent acoustic patterns."
                )
            return response, False

        # ----------------------------------------------------------------------
        # 4. CONFIDENCE EXPLANATION: "WHAT DOES CONFIDENCE MEAN?"
        # ----------------------------------------------------------------------
        if any(w in q_lower for w in ["confidence", "probability", "percent", "%", "score"]):
            response = (
                f"Your model confidence score is **{conf_pct_str}** for the **{pred}** classification.\n\n"
                f"**What it means:** A confidence score represents the statistical certainty of the AI algorithm based on how closely "
                f"the spectral-temporal acoustic features of your recording match the thousands of heart sounds in its training dataset.\n\n"
                f"**What it does NOT mean:** It does **NOT** mean there is a {conf_pct_str} medical probability or chance of having a condition. "
                f"Confidence reflects algorithm feature alignment, not clinical likelihood."
            )
            return response, False

        # ----------------------------------------------------------------------
        # 5. NEXT STEPS & PRECAUTIONS: "WHAT SHOULD I DO NEXT?" / "PRECAUTIONS"
        # ----------------------------------------------------------------------
        if any(w in q_lower for w in ["do next", "precautions", "action", "next step", "what now", "lifestyle"]):
            if is_murmur:
                response = (
                    f"Since your AI screening result indicates a **possible heart murmur ({conf_pct_str} confidence)**, here are the sensible next steps:\n\n"
                    f"1. **Consult a Doctor:** Share this screening recording with your primary healthcare provider or cardiologist.\n"
                    f"2. **Physical Auscultation:** A physician can listen directly with an acoustic stethoscope across all valve areas.\n"
                    f"3. **Diagnostic Confirmation:** If indicated, they may order an **echocardiogram** (heart ultrasound) or ECG.\n"
                    f"4. **Monitor Symptoms:** Take note if you notice shortness of breath during exertion, dizziness, unusual fatigue, or chest discomfort."
                )
            else:
                response = (
                    f"Since your AI screening result is **NORMAL ({conf_pct_str} confidence)**, sensible next steps include:\n\n"
                    f"1. **Routine Preventive Care:** Continue with your standard routine medical and health check-ups.\n"
                    f"2. **Heart-Healthy Habits:** Maintain a balanced diet, regular physical exercise, and adequate sleep.\n"
                    f"3. **Listen to Your Body:** If you ever experience new or concerning symptoms (such as palpitations or dizziness), consult a healthcare professional regardless of prior screening results."
                )
            return response, False

        # ----------------------------------------------------------------------
        # 6. "SHOULD I SEE A DOCTOR?"
        # ----------------------------------------------------------------------
        if any(w in q_lower for w in ["see a doctor", "visit a doctor", "cardiologist", "physician", "need a doctor", "hospital"]):
            if is_murmur:
                response = (
                    f"**Yes, it is advisable to consult a healthcare professional.**\n\n"
                    f"Because the screening identified acoustic patterns consistent with a murmur ({conf_pct_str} confidence), "
                    f"having a doctor perform a stethoscope examination is the most reliable way to confirm whether a murmur is present, "
                    f"and if so, whether it is innocent (benign) or requires monitoring."
                )
            else:
                response = (
                    f"An AI screening result alone cannot replace clinical advice. While your result was **NORMAL ({conf_pct_str} confidence)**, "
                    f"you should still see a doctor if you are having any concerning symptoms such as chest tightness, dizziness, shortness of breath, "
                    f"or if you are due for a routine cardiovascular check-up."
                )
            return response, False

        # ----------------------------------------------------------------------
        # 7. "WHAT IS A HEART MURMUR?"
        # ----------------------------------------------------------------------
        if any(w in q_lower for w in ["what is a murmur", "what is a heart murmur", "explain murmur", "murmurs mean"]):
            response = (
                "A **heart murmur** is an unusual sound heard between heartbeats—often described as a whooshing, swishing, or rasping noise. "
                "It is caused by turbulent blood flow in or near the heart valves.\n\n"
                "• **Innocent Murmurs:** Very common, especially in children, adolescents, and athletes. They occur when blood flows rapidly through a completely healthy heart and do not require treatment.\n"
                "• **Abnormal Murmurs:** Occur due to structural valve issues (e.g. valve narrowing/stenosis or leaking/regurgitation) or congenital heart variations.\n\n"
                "A clinical examination by a physician with an echocardiogram is the standard method used to distinguish between innocent and abnormal murmurs."
            )
            return response, False

        # ----------------------------------------------------------------------
        # 8. VALVE LOCATIONS & MULTI-LOCATION AUSCULTATION
        # ----------------------------------------------------------------------
        if any(w in q_lower for w in ["valve", "locations", "av", "mv", "pv", "tv", "auscultation", "sites"]):
            response = (
                "LubDub Murmur analyzes heart sounds across the 4 primary anatomical auscultation valve sites:\n\n"
                "• **AV (Aortic Valve):** Located at the 2nd right intercostal space.\n"
                "• **MV (Mitral Valve):** Located at the 5th left intercostal space (cardiac apex).\n"
                "• **PV (Pulmonary Valve):** Located at the 2nd left intercostal space.\n"
                "• **TV (Tricuspid Valve):** Located at the 4th left intercostal sternal border.\n\n"
                "Multi-location auscultation allows the AI to localize where acoustic turbulence is strongest and combine recordings using attention weighting."
            )
            return response, False

        # ----------------------------------------------------------------------
        # 9. GENERAL / CONTEXTUAL FALLBACK RESPONSE
        # ----------------------------------------------------------------------
        response = (
            f"Regarding your current screening result (**{pred}**, {conf_pct_str} confidence):\n\n"
            f"I can help explain what heart sound murmurs are, how the AI interprets acoustic signals, "
            f"what confidence percentages represent, or what general next steps to consider.\n\n"
            f"Feel free to ask a specific question, such as:\n"
            f"• *Why is my result {pred.lower()}?*\n"
            f"• *What does my {conf_pct_str} confidence mean?*\n"
            f"• *What should I do next?*\n"
            f"• *What is a heart murmur?*"
        )
        return response, False

    def _get_suggested_followups(self, user_query: str, ctx: Dict[str, Any]) -> List[str]:
        """Provides dynamic suggested follow-up questions tailored to the conversation flow."""
        pred = str(ctx.get("prediction", "NORMAL")).upper()
        is_murmur = pred in ["MURMUR", "PRESENT", "MURMUR DETECTED", "ABNORMAL"]

        if is_murmur:
            return [
                "What does my result mean?",
                "What does my confidence mean?",
                "Should I see a doctor?",
                "What is a heart murmur?",
                "What precautions should I take?"
            ]
        else:
            return [
                "Why is my result normal?",
                "What does my confidence mean?",
                "What should I do next?",
                "Can this result diagnose my heart condition?",
                "What is a heart murmur?"
            ]


# Singleton Service Instance
_chatbot_service = LubDubChatbotService()

def get_chatbot_service() -> LubDubChatbotService:
    """Returns the singleton LubDubChatbotService instance."""
    return _chatbot_service
