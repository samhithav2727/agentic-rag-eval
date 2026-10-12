# Agentic RAG Eval

An agentic retrieval-augmented generation (RAG) system over research papers, with three retrieval pipelines benchmarked using RAGAS.

## Architecture

PDFs -> chunking -> FAISS (dense) + BM25 (sparse) -> cross-encoder reranker -> LangGraph agent (query rewriting) -> FastAPI `/ask`

## Pipelines compared

1. **Vector**: dense FAISS retrieval only
2. **Hybrid + rerank**: FAISS + BM25, then reranking
3. **Agentic**: hybrid + rerank inside a LangGraph loop that can rewrite the query and retry

## Results (RAGAS, 10 questions)

| Pipeline | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---|---|---|---|
| Vector | 0.900 | 0.901 | 0.650 | 0.900 |
| Hybrid + rerank | 0.930 | 0.877 | 0.842 | 0.800 |
| Agentic | 0.989 | 0.902 | 0.723 | 0.800 |

## Findings and trade-offs

- Hybrid retrieval with reranking raised context precision from 0.65 to 0.84.
- The agentic pipeline had the highest faithfulness (0.99).
- Context recall dropped from 0.90 to 0.80 for both upgraded pipelines.
- The evaluation set is small (10 questions), so treat these numbers as indicative, not definitive.

## Setup

```
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt
```

Create a `.env` file with `OPENAI_API_KEY=your_key`, and download the papers into `data/pdfs/`:

```
curl -L -o data/pdfs/attention.pdf https://arxiv.org/pdf/1706.03762
curl -L -o data/pdfs/rag.pdf https://arxiv.org/pdf/2005.11401
curl -L -o data/pdfs/lost-in-the-middle.pdf https://arxiv.org/pdf/2307.03172
curl -L -o data/pdfs/self-rag.pdf https://arxiv.org/pdf/2310.11511
```

## Run

```
python -m src.ingest          # build the index
uvicorn src.api:app --reload  # API docs at http://127.0.0.1:8000/docs
python -m eval.run_ragas      # run the evaluation
```

Example request:

```
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"question": "What is self-RAG?"}'
```