import sys
from pathlib import Path

from sklearn import tests
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pdfplumber
from src.config import CORPUS_DIR
from src.embedder import Embedder
from src.vector_store import VectorStore

CHUNK_SIZE, CHUNK_OVERLAP = 512, 75

def extract_text(pdf_path: Path) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text

def chunk_words(text: str, source: str) -> list[dict]:
    words, out = text.split(), []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if len(chunk.strip()) < 50:
            continue
        out.append({"text": chunk, "source": source})
    return out

def main():
    pdfs = list(CORPUS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs in {CORPUS_DIR}. Add some and rerun.")
        return

    embedder = Embedder()
    store    = VectorStore()
    all_chunks = []
    for pdf in pdfs:
        print(f"Reading {pdf.name} ...")
        all_chunks.extend(chunk_words(extract_text(pdf), pdf.name))

    print(f"Embedding {len(all_chunks)} chunks ...")
    embeddings = embedder.embed([c["text"] for c in all_chunks])

    store.add(
        ids=[f"chunk_{i}" for i in range(len(all_chunks))],
        embeddings=embeddings,
        documents=[c["text"] for c in all_chunks],
        metadatas=[{"source": c["source"]} for c in all_chunks],
    )
    print(f"Done. Chroma now holds {store.count()} chunks.")

if __name__ == "__main__":
    main()

