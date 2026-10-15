from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from backend.ingestion import process_pdf

from backend.config import CHROMA_DIR, EMBEDDING_MODEL

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vector_store():
    return Chroma(
        collection_name="enterprise_research",
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"}
    )


def index_pdf(pdf_path: str) -> dict:
    chunks = process_pdf(pdf_path)

    store = get_vector_store()

    document_id = chunks[0].metadata["document_id"]

    # Check whether this exact PDF is already indexed.
    existing = store.get(
        where={"document_id": document_id},
        include=["metadatas"],
    )

    if existing["ids"]:
        return {
            "status": "already_indexed",
            "filename": chunks[0].metadata["source"],
            "chunks": len(existing["ids"]),
        }

    ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    # Batch inserts to handle large PDFs without exceeding Chroma batch limits
    batch_size = 250
    for i in range(0, len(chunks), batch_size):
        store.add_documents(
            documents=chunks[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )

    return {
        "status": "indexed",
        "filename": chunks[0].metadata["source"],
        "pages": len({
            chunk.metadata["page"]
            for chunk in chunks
        }),
        "chunks": len(chunks),
    }

