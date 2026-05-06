from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
VECTOR_BACKEND = "chroma"

CHROMA_SETTINGS = {
    "persist_directory": "chroma_db",
    "collection_name": "akos_porocila",
}

RETRIEVER_SETTINGS = {
    "search_type": "mmr",
    "search_kwargs": {"k": 4, "fetch_k": 12},
}

def get_vectorstore(embeddings: Embeddings) -> VectorStore:
    return Chroma(
        persist_directory=CHROMA_SETTINGS["persist_directory"],
        embedding_function=embeddings,
        collection_name=CHROMA_SETTINGS["collection_name"],
    )

def ingest_documents(chunks, embeddings: Embeddings) -> VectorStore:
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_SETTINGS["persist_directory"],
        collection_name=CHROMA_SETTINGS["collection_name"],
    )