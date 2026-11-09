from sentence_transformers import SentenceTransformer
from .config import BGE_MODEL

class Embedder:
    def __init__(self, model_name: str = BGE_MODEL):
        self.model = SentenceTransformer(model_name, device="cpu")

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        ).tolist()
