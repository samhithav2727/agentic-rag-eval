import pickle
from pathlib import Path
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from flashrank import Ranker, RerankRequest
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
INDEX_DIR = Path("data/index")

_emb = OpenAIEmbeddings(model="text-embedding-3-small")
_vs = FAISS.load_local(str(INDEX_DIR), _emb, allow_dangerous_deserialization=True)
with open(INDEX_DIR / "chunks.pkl", "rb") as f:
    _chunks = pickle.load(f)
_bm25 = BM25Okapi([c.page_content.lower().split() for c in _chunks])
_ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="data/models")


def vector_search(query, k=10):
    return _vs.similarity_search(query, k=k)


def bm25_search(query, k=10):
    scores = _bm25.get_scores(query.lower().split())
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [_chunks[i] for i in top]


def hybrid_search(query, k=10):
    """Reciprocal Rank Fusion of vector + BM25 results."""
    fused, docs = {}, {}
    for results in (vector_search(query, k), bm25_search(query, k)):
        for rank, d in enumerate(results):
            cid = d.metadata["chunk_id"]
            docs[cid] = d
            fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
    ranked = sorted(fused, key=fused.get, reverse=True)[:k]
    return [docs[c] for c in ranked]


def rerank(query, docs, top_n=5):
    passages = [{"id": i, "text": d.page_content} for i, d in enumerate(docs)]
    results = _ranker.rerank(RerankRequest(query=query, passages=passages))
    return [docs[r["id"]] for r in results[:top_n]]


def retrieve(query, mode="hybrid_rerank", k=5):
    if mode == "vector":
        return vector_search(query, k)
    if mode == "bm25":
        return bm25_search(query, k)
    if mode == "hybrid":
        return hybrid_search(query, k)
    return rerank(query, hybrid_search(query, 20), top_n=k)


if __name__ == "__main__":
    q = "What is the lost in the middle problem?"
    for mode in ["vector", "bm25", "hybrid", "hybrid_rerank"]:
        print(f"\n=== {mode} ===")
        for d in retrieve(q, mode):
            print(f"- {d.metadata['source']} p{d.metadata.get('page')}: "
                  f"{d.page_content[:90]!r}")