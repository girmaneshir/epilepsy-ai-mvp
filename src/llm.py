import requests
from .config import OLLAMA_MODEL, OLLAMA_URL, OLLAMA_KEEP_ALIVE

class OllamaLLM:
    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url   = url
        try:
            requests.get(f"{self.url}/api/tags", timeout=3).raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Ollama not reachable at {self.url}. ({e})")

    def generate(self, system: str, user: str, max_tokens: int = 400) -> str:
        r = requests.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "stream": False,
                "keep_alive": OLLAMA_KEEP_ALIVE,
                "options": {"temperature": 0.1, "num_predict": max_tokens},
            },
            timeout=120,
        )
        r.raise_for_status()
        return r.json()["message"]["content"]

def get_llm():
    return OllamaLLM()
