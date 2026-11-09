from pathlib import Path
import os

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CHROMA_DIR = DATA_DIR / "chroma_db"

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
BGE_MODEL  = "BAAI/bge-m3"
CHROMA_COLLECTION = "epilepsy_kb"

OLLAMA_MODEL      = os.getenv("OLLAMA_MODEL",      "qwen2.5:0.5b-instruct")
OLLAMA_URL        = os.getenv("OLLAMA_URL",        "http://localhost:11434")
OLLAMA_KEEP_ALIVE = int(os.getenv("OLLAMA_KEEP_ALIVE", "0"))

NUM_BEAMS = 2

LANG_CODES = {
    "en": "eng_Latn", "am": "amh_Ethi", "om": "gaz_Latn", "ti": "tir_Ethi",
}

EMERGENCY_KEYWORDS = [
    "seizing", "convulsing", "status epilepticus", "not breathing", "unconscious", "can't wake",
    "መንቀጥቀጥ", "ማንቀጥቀጥ", "gaggabaa", "ምንቅጥቃጥ",
]

MEDICAL_ADVICE_PATTERNS = [
    "should i stop", "should i take", "change my dose", "prescribe", "dosage for", "what medication",
]

INJECTION_PATTERNS = [
    "ignore previous", "ignore all instructions", "system prompt", "you are now", "jailbreak",
]
