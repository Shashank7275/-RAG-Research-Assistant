import hashlib
import re
from pathlib import Path

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text:str) -> str:

    """"Normalizes whitespace in the text by replacing multiple spaces, tabs, and newlines with a single space."""

    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def _ocr_pdf_page(pdf_path: str | Path, page_index: int) -> str:
    try:
        import pypdfium2 as pdfium
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "OCR dependencies are missing. Install the project requirements "
            "and make sure Tesseract OCR is installed."
        ) from exc

    try:
        with pdfium.PdfDocument(str(pdf_path)) as pdf:
            page = pdf[page_index]
            image = page.render(scale=3).to_pil()
            return pytesseract.image_to_string(image)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed or is not available on PATH."
        ) from exc


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """"Extracts text from a PDF file and splits it into chunks of text documents."""

    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    reader = PdfReader(str(path))

    if reader.is_encrypted:
        raise ValueError("Encrypted PDF files are not supported")

    document_id = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        extraction_method = "text"

        if not text:
            text = clean_text(_ocr_pdf_page(path, page_number - 1))
            extraction_method = "ocr"

        if not text:
            continue

        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "page": page_number,
                    "document_id": document_id,
                    "extraction_method": extraction_method,
                },
            )
        )

    if not pages:
        raise ValueError(
            "No readable text was found in this PDF, even after OCR. "
            "Check that the scan is clear and that Tesseract has the correct "
            "language data installed."
        )

    return pages

def split_documents(
        pages: list[Document],

) -> list[Document]:
    """Splits a list of Document objects into smaller chunks based on the specified chunk size and overlap."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    for index, chunk in enumerate(chunks):

        chunk.metadata["chunk_index"] = index

        stable_id = hashlib.sha256(
            (

            chunk.metadata["document_id"]
            + str(chunk.metadata["page"])
            + str(chunk.metadata.get("start_index", 0))
            + chunk.page_content
            ).encode("utf-8")
        ).hexdigest()

        chunk.metadata["chunk_id"] = stable_id

    return chunks

def process_pdf(pdf_path: str | Path) -> list[Document]:

    pages = load_pdf(pdf_path)
    chunks = split_documents(pages)


    return chunks
