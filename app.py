from fastapi import FastAPI
from pydantic import BaseModel

from src.retrievers import load_store, get_reranked
from src.generate import answer

app = FastAPI(title="FastAPI Docs Q&A")

STORE = load_store()
RETRIEVER = get_reranked(STORE, k_wide=15, top_n=8)

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Query):
    return answer(q.question, RETRIEVER)

@app.get("/health")
def health():
    return {
        "ok": True, "chunks": STORE._collection.count()
    }