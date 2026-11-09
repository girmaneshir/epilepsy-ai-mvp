from .config import EMERGENCY_KEYWORDS, MEDICAL_ADVICE_PATTERNS, INJECTION_PATTERNS

FIRST_AID_TEMPLATE = (
    "⚠️ This may be a medical emergency. Call emergency services now "
    "(Ethiopia: 907 / 911). Keep the person safe, do not restrain, time the seizure."
)
REFUSAL_TEMPLATE   = "I cannot provide medical advice or prescriptions. Please consult a qualified healthcare professional."
INJECTION_TEMPLATE = "Request blocked for security reasons."

def check(text: str) -> dict | None:
    low = text.lower()
    if any(k.lower() in low for k in EMERGENCY_KEYWORDS):
        return {"type": "emergency", "response": FIRST_AID_TEMPLATE, "bypass_llm": True}
    if any(p in low for p in MEDICAL_ADVICE_PATTERNS):
        return {"type": "medical_advice", "response": REFUSAL_TEMPLATE, "bypass_llm": True}
    if any(p in low for p in INJECTION_PATTERNS):
        return {"type": "injection", "response": INJECTION_TEMPLATE, "bypass_llm": True}
    return None
