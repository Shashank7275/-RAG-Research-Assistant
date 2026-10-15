import hashlib
import logging
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field

from backend.config import UPLOAD_DIR, MAX_FILE_SIZE_MB
from backend.vector_store import index_pdf
from backend.rag_chain import answer_question
from backend.vector_store import get_vector_store

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Enterprise RAG Research Assistant",
    version="1.0.0",
    description="Document-grounded research API",
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=2000,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    if Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    temp_path = UPLOAD_DIR / f"temp_{uuid.uuid4().hex}.pdf"
    hasher = hashlib.sha256()
    size = 0
    first_chunk = True

    try:
        with open(temp_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break

                if first_chunk:
                    if not chunk.startswith(b"%PDF-"):
                        raise HTTPException(
                            status_code=400,
                            detail="Uploaded file is not a valid PDF header.",
                        )
                    first_chunk = False

                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF exceeds the {MAX_FILE_SIZE_MB} MB limit.",
                    )

                hasher.update(chunk)
                f.write(chunk)

        if size == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        file_hash = hasher.hexdigest()
        safe_path = UPLOAD_DIR / f"{file_hash}.pdf"

        if safe_path.exists():
            temp_path.unlink(missing_ok=True)
        else:
            temp_path.replace(safe_path)

        result = index_pdf(str(safe_path))
        result["document_id"] = file_hash
        return result

    except HTTPException:
        temp_path.unlink(missing_ok=True)
        raise

    except (ValueError, FileNotFoundError) as exc:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        logger.exception("Document processing failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(exc)}",
        ) from exc


@app.post("/ask")
def ask_question(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        result = answer_question(question)
        return result

    except Exception as exc:
        logger.exception("Question answering failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Unable to answer the question: {str(exc)}",
        ) from exc


@app.get("/documents")
def list_documents():
    try:
        store = get_vector_store()
        data = store.get(include=["metadatas"]) or {}
        metadatas = data.get("metadatas") or []
    except Exception as exc:
        logger.exception("Unable to load indexed documents.")
        raise HTTPException(
            status_code=500,
            detail="Unable to list documents.",
        ) from exc

    documents = {}

    for metadata in metadatas:
        if not isinstance(metadata, dict):
            continue

        doc_id = metadata.get("document_id")
        if not doc_id:
            continue

        if doc_id not in documents:
            documents[doc_id] = {
                "document_id": doc_id,
                "filename": metadata.get("source", "Unknown"),
                "pages": set(),
                "chunks": 0,
            }

        page_number = metadata.get("page")
        if page_number is not None:
            documents[doc_id]["pages"].add(page_number)
        documents[doc_id]["chunks"] += 1

    return [
        {
            **doc,
            "pages": len(doc["pages"]),
        }
        for doc in documents.values()
    ]


if __name__ == "__main__":
    import asyncio
    import sys
    import uvicorn

    if sys.platform == "win32":
        config = uvicorn.Config(app, host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)

        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            runner.run(server.serve())
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8000)

