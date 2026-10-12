import pickle
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
DATA_DIR = Path("data/pdfs")
INDEX_DIR = Path("data/index")


def load_docs():
    docs = []
    for pdf in DATA_DIR.glob("*.pdf"):
        pages = PyPDFLoader(str(pdf)).load()
        for p in pages:
            p.metadata["source"] = pdf.name
        docs.extend(pages)
    return docs


def chunk(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = i
    return chunks


def build_index(chunks):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs = FAISS.from_documents(
        chunks, OpenAIEmbeddings(model="text-embedding-3-small")
    )
    vs.save_local(str(INDEX_DIR))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)  # needed later for BM25


if __name__ == "__main__":
    chunks = chunk(load_docs())
    print(f"{len(chunks)} chunks")
    build_index(chunks)
    print("Index saved")