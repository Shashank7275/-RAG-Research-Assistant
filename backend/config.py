import os
from pathlib import Path

from dotenv import load_dotenv

# Initialize system certificates on Windows to prevent SSL certificate verification errors
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "data/uploads")

CHROMA_DIR = BASE_DIR / os.getenv("CHROMA_DIR", "chroma_db")

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 200))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL", "gemini-2.5-flash"
)
GEMINI_FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL", "gemini-2.5-flash-lite"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 8))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 4))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

def validate_config():
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "GOOGLE_API_KEY missing. Please set it in the .env file or environment variables."
        )

    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be a positive integer.")

    if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be a non-negative integer less than CHUNK_SIZE.")

    if MAX_FILE_SIZE_MB <= 0:
        raise ValueError("MAX_FILE_SIZE_MB must be a positive integer.")


    