import gc
from .vector_store import VectorStore
from .guardrails import check
from .llm import get_llm

SYSTEM_PROMPT = """You are an epilepsy awareness assistant for Ethiopia.
Answer ONLY using the provided context. If the context is insufficient, say so.
Never prescribe medication or give clinical advice. Keep answers concise.
Cite sources inline as [Source: filename]."""

class Pipeline:
    def __init__(self):
        self.store = VectorStore()
        self.llm   = get_llm()
        self._translator = None
        self._embedder   = None

    def _get_translator(self):
        if self._translator is None:
            from .translator import Translator
            self._translator = Translator()
        return self._translator

    def _get_embedder(self):
        if self._embedder is None:
            from .embedder import Embedder
            self._embedder = Embedder()
        return self._embedder

    def _drop_translator(self):
        self._translator = None
        gc.collect()

    def query(self, text: str, lang: str = "en") -> dict:
        blocked = check(text)
        if blocked:
            return {"answer": blocked["response"], "english_answer": blocked["response"],
                    "lang": lang, "sources": [], "guardrail": blocked["type"]}

        if lang != "en":
            en_query = self._get_translator().translate(text, lang, "en")
            self._drop_translator()
        else:
            en_query = text

        q_emb   = self._get_embedder().embed([en_query])[0]
        results = self.store.query(q_emb, top_k=5)
        docs    = results["documents"][0]
        metas   = results["metadatas"][0]

        context = "\n\n".join(f"[Source: {m['source']}]\n{d}" for d, m in zip(docs, metas))
        en_answer = self.llm.generate(SYSTEM_PROMPT, f"Context:\n{context}\n\nQuestion: {en_query}")
        final = en_answer if lang == "en" else self._get_translator().translate(en_answer, "en", lang)

        return {"answer": final, "english_answer": en_answer, "lang": lang,
                "sources": sorted({m["source"] for m in metas}), "guardrail": None}
