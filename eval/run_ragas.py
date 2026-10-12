import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness, answer_relevancy, context_precision, context_recall
)
from src.retrieval import retrieve
from src.graph import ask, llm

load_dotenv()
data = json.load(open("eval/dataset.json"))


def simple_rag(question, mode):
    docs = retrieve(question, mode=mode, k=5)
    context = "\n\n".join(d.page_content for d in docs)
    answer = llm.invoke(
        f"Answer using ONLY the context.\n\nQuestion: {question}\n\nContext:\n{context}"
    ).content
    return {"answer": answer, "contexts": [d.page_content for d in docs]}


CONFIGS = {
    "vector": lambda q: simple_rag(q, "vector"),
    "hybrid_rerank": lambda q: simple_rag(q, "hybrid_rerank"),
    "agentic": lambda q: ask(q),
}

judge = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

summary = {}
for name, fn in CONFIGS.items():
    print(f"\nRunning: {name}")
    samples = []
    for item in data:
        out = fn(item["question"])
        samples.append({
            "user_input": item["question"],
            "response": out["answer"],
            "retrieved_contexts": out["contexts"],
            "reference": item["reference"],
        })
    result = evaluate(
        EvaluationDataset.from_list(samples),
        metrics=metrics, llm=judge, embeddings=emb,
    )
    df = result.to_pandas()
    df.to_csv(f"eval/results_{name}.csv", index=False)
    summary[name] = df.select_dtypes("number").mean().round(3).to_dict()

print("\n=== SUMMARY ===")
for name, scores in summary.items():
    print(name, scores)
json.dump(summary, open("eval/summary.json", "w"), indent=2)