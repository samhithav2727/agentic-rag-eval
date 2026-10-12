from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from src.retrieval import retrieve

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
MAX_RETRIES = 2


class State(TypedDict):
    question: str
    query: str
    docs: list
    relevant: bool
    answer: str
    retries: int


def retrieve_node(state):
    return {"docs": retrieve(state["query"], mode="hybrid_rerank", k=5)}


def grade_node(state):
    context = "\n\n".join(d.page_content for d in state["docs"])
    prompt = (
        "Can the question be answered using ONLY the context below?\n"
        "Reply with exactly 'yes' or 'no'.\n\n"
        f"Question: {state['question']}\n\nContext:\n{context}"
    )
    verdict = llm.invoke(prompt).content.strip().lower()
    return {"relevant": verdict.startswith("yes")}


def rewrite_node(state):
    prompt = (
        "Rewrite this question as a better search query for finding passages "
        "in research papers. Return only the query.\n\n"
        f"Question: {state['question']}\nPrevious query: {state['query']}"
    )
    return {
        "query": llm.invoke(prompt).content.strip(),
        "retries": state["retries"] + 1,
    }


def generate_node(state):
    context = "\n\n".join(
        f"[{i + 1}] ({d.metadata['source']} p{d.metadata.get('page')}) {d.page_content}"
        for i, d in enumerate(state["docs"])
    )
    prompt = (
        "Answer the question using ONLY the numbered context. Cite sources "
        "like [1], [2]. If the context is insufficient, say you don't know.\n\n"
        f"Question: {state['question']}\n\nContext:\n{context}"
    )
    return {"answer": llm.invoke(prompt).content}


def route(state):
    if state["relevant"] or state["retries"] >= MAX_RETRIES:
        return "generate"
    return "rewrite"


g = StateGraph(State)
g.add_node("retrieve", retrieve_node)
g.add_node("grade", grade_node)
g.add_node("rewrite", rewrite_node)
g.add_node("generate", generate_node)
g.add_edge(START, "retrieve")
g.add_edge("retrieve", "grade")
g.add_conditional_edges("grade", route, {"generate": "generate", "rewrite": "rewrite"})
g.add_edge("rewrite", "retrieve")
g.add_edge("generate", END)
graph = g.compile()


def ask(question):
    out = graph.invoke(
        {"question": question, "query": question, "retries": 0,
         "docs": [], "relevant": False, "answer": ""}
    )
    sources = [
        {"source": d.metadata["source"], "page": d.metadata.get("page")}
        for d in out["docs"]
    ]
    return {"answer": out["answer"], "sources": sources,
            "contexts": [d.page_content for d in out["docs"]],
            "retries": out["retries"]}


if __name__ == "__main__":
    r = ask("What is the lost in the middle problem?")
    print(r["answer"])
    print("\nSources:", r["sources"])
    print("Rewrites used:", r["retries"])
    