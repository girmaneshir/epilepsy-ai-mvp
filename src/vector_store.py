import chromadb
from chromadb.config import Settings
from .config import CHROMA_DIR, CHROMA_COLLECTION

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids, embeddings, documents, metadatas):
        self.collection.add(ids=ids, embeddings=embeddings,
                            documents=documents, metadatas=metadatas)

    def query(self, embedding: list[float], top_k: int = 5):
        return self.collection.query(
            query_embeddings=[embedding], n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self.collection.count()
