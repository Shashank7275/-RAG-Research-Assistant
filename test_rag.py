from langchain_core.documents import Document

from backend.ingestion import clean_text, load_pdf, split_documents
from backend import rag_chain
from backend.retrieval import rerank_documents
from pypdf import PdfWriter


def test_answer_question_uses_fallback_model(monkeypatch):
    class FakeChain:
        def __init__(self, model):
            self.model = model

        def __or__(self, _parser):
            return self

        def invoke(self, _inputs):
            if self.model == "primary-model":
                raise ConnectionResetError("connection reset")
            return "Answer from fallback model."

    class FakePrompt:
        def __or__(self, model):
            return FakeChain(model)

    monkeypatch.setattr(
        rag_chain,
        "retrieve",
        lambda _question: [
            Document(
                page_content="Supporting document passage.",
                metadata={"source": "paper.pdf", "page": 1},
            )
        ],
    )
    monkeypatch.setattr(rag_chain, "PROMPT", FakePrompt())
    monkeypatch.setattr(rag_chain, "get_llm", lambda model: model)
    monkeypatch.setattr(rag_chain, "GEMINI_MODEL", "primary-model")
    monkeypatch.setattr(rag_chain, "GEMINI_FALLBACK_MODEL", "fallback-model")

    result = rag_chain.answer_question("What does the paper say?")

    assert result["answer"] == "Answer from fallback model."
    assert result["sources"][0]["citation"] == "S1"


def test_clean_text():
    text = "Hello   world\n\n\nPython"

    result = clean_text(text)

    assert result == "Hello world\n\nPython"


def test_chunking_preserves_page_metadata():
    pages = [
        Document(
            page_content="Research text. " * 300,
            metadata={
                "source": "paper.pdf",
                "page": 4,
                "document_id": "test-document",
            },
        )
    ]

    chunks = split_documents(pages)

    assert len(chunks) > 1
    assert all(chunk.metadata["page"] == 4 for chunk in chunks)
    assert all(chunk.metadata["chunk_id"] for chunk in chunks)


def test_scanned_pdf_page_uses_ocr(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scanned.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)

    with pdf_path.open("wb") as pdf_file:
        writer.write(pdf_file)

    monkeypatch.setattr(
        "backend.ingestion._ocr_pdf_page",
        lambda path, page_index: "Text recognized from scanned page.",
    )

    pages = load_pdf(pdf_path)

    assert len(pages) == 1
    assert pages[0].page_content == "Text recognized from scanned page."
    assert pages[0].metadata["page"] == 1
    assert pages[0].metadata["extraction_method"] == "ocr"


def test_reranker_returns_top_k():
    documents = [
        Document(page_content="Python is a programming language."),
        Document(page_content="The weather is sunny today."),
    ]

    result = rerank_documents(
        "What is Python?",
        documents,
    )

    assert len(result) == 2
    assert all(
        "rerank_score" in doc.metadata
        for doc in result
    )