from functools import lru_cache

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

from backend.config import RETRIEVAL_K, RERANK_TOP_K
from backend.vector_store import get_vector_store

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

@lru_cache(maxsize=1)

def get_reranker():
    return CrossEncoder(RERANK_MODEL, device="cpu")

def get_indexed_documents():
    """Reconstructs the indexed documents from the vector store."""

    store = get_vector_store()

    data = store.get(include=["documents", "metadatas"])

    documents = []

    for text, metadata in zip(
        data["documents"],
        data["metadatas"],
    ):

        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata or {},
                )
            )


    return documents

def get_hybrid_retriever() -> EnsembleRetriever | None:

    store = get_vector_store()
    corpus = get_indexed_documents()

    if not corpus:
        return None

    vector_retriever = store.as_retriever(

        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_K},
    )

    keyword_retriever = BM25Retriever.from_documents(
        corpus,
        k=RETRIEVAL_K,
    )

    return EnsembleRetriever(
        retrievers=[
            vector_retriever,
            keyword_retriever,
        ],
        weights=[0.65, 0.35],

    )

def rerank_documents(
        question:str,
        documents: list[Document],
) -> list[Document]:
    if not documents:
        return []

    reranker = get_reranker()

    pairs = [
        (question, doc.page_content)
        for doc in documents

    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,

    )

    results = []

    for doc, score in ranked[:RERANK_TOP_K]:
        # Diagnostic score only; not a calibrated probability.

        doc.metadata["rerank_score"] = round(
            float(score), 4
        )

        results.append(doc)

    return results

def retrieve(question:str) -> list[Document]:

    retriever = get_hybrid_retriever()

    if retriever is None:
        return []

    candidates = retriever.invoke(question)

    return rerank_documents(question, candidates)
