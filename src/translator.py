import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from .config import NLLB_MODEL, LANG_CODES, NUM_BEAMS

class Translator:
    def __init__(self, model_name: str = NLLB_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, torch_dtype=torch.float32,
        )
        self.model.eval()

    def translate(self, text: str, src: str, tgt: str, max_len: int = 256) -> str:
        if src == tgt:
            return text
        self.tokenizer.src_lang = LANG_CODES[src]
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        bos_id = self.tokenizer.convert_tokens_to_ids(LANG_CODES[tgt])
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                forced_bos_token_id=bos_id,
                max_length=max_len,
                num_beams=NUM_BEAMS,
                early_stopping=True,
            )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)
