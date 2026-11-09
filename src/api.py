# src/api.py
from fastapi import FastAPI
from fastapi.responses import JSONResponse
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
    result = pipeline.query(q.text, q.lang)
    # Return JSON with ensure_ascii=False so Amharic is sent as raw UTF-8
    return JSONResponse(content=result)

@app.get("/health")
def health():
    return {"status": "ok"}