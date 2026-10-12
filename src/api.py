from fastapi import FastAPI
from pydantic import BaseModel
from src.graph import ask

app = FastAPI(title="Agentic RAG API")


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_endpoint(q: Query):
    r = ask(q.question)
    return {
        "answer": r["answer"],
        "sources": r["sources"],
        "rewrites": r["retries"],
    }