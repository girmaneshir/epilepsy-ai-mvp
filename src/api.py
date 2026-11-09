from fastapi import FastAPI
from pydantic import BaseModel
from .pipeline import Pipeline

app = FastAPI(title="Epilepsy AI MVP (Low-RAM)")
pipeline: Pipeline | None = None

@app.on_event("startup")
def _load():
    global pipeline
    pipeline = Pipeline()

class Query(BaseModel):
    text: str
    lang: str = "en"

@app.post("/ask")
def ask(q: Query):
    return pipeline.query(q.text, q.lang)

@app.get("/health")
def health():
    return {"status": "ok"}
