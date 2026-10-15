# 🚀 Enterprise RAG Research Assistant

An AI-powered Enterprise RAG (Retrieval-Augmented Generation) application that allows users to upload PDF documents and ask questions using natural language. The system retrieves relevant information from uploaded documents and generates context-aware answers using a Large Language Model (LLM).

## ✨ Features

- 📄 **PDF Document Ingestion:** Upload and process PDF documents.
- ✂️ **Intelligent Text Chunking:** Split documents into manageable text chunks for retrieval.
- 🧠 **Semantic Search:** Generate embeddings and retrieve relevant document passages.
- 🔎 **Hybrid Retrieval:** Support semantic retrieval and BM25 keyword search where configured.
- 🤖 **LLM-Powered Answers:** Generate answers grounded in retrieved document context.
- 🗄️ **Vector Database:** Store and search document embeddings using ChromaDB.
- ⚡ **FastAPI Backend:** Expose RAG functionality through REST API endpoints.
- 🎨 **Streamlit Frontend:** Provide an interactive interface for document uploads and question answering.
- 🩺 **Health Check:** Check backend availability using the `/health` endpoint.
- ☁️ **Cloud Deployment:** Deploy the backend and frontend as separate Render web services.

## 🏗️ Architecture

```text
User
  |
  v
Streamlit Frontend
  |
  | HTTP Requests
  v
FastAPI Backend
  |
  v
PDF Ingestion Pipeline
  |
  v
Text Chunking
  |
  v
Embedding Model
  |
  v
ChromaDB Vector Store
  |
  v
Retriever / Hybrid Search
  |
  v
Relevant Context + User Question
  |
  v
LLM Response Generation
  |
  v
Answer Displayed in Streamlit
```

## 🛠️ Technology Stack

| Component | Technologies |
|---|---|
| Programming Language | Python |
| Frontend | Streamlit |
| Backend | FastAPI, Uvicorn |
| LLM Framework | LangChain |
| Embeddings | Sentence Transformers / Hugging Face |
| Vector Database | ChromaDB |
| Retrieval | LangChain Retrievers, BM25 |
| Document Processing | PyPDF, pypdfium2 |
| LLM Integration | Google Gemini through LangChain, if configured |
| API Communication | Requests, REST API |
| Deployment | Render |
| Testing | Pytest, HTTPX |

## 📁 Project Structure

```text
Advance-RAG-Project/
│
├── app.py
├── requirements.txt
├── backend_requirements.txt
├── README.md
├── .gitignore
│
├── backend/
│   ├── api.py
│   ├── config.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── rag_chain.py
│   └── vector_store.py
│
├── chroma_db/
├── data/
│   └── uploads/
│
└── test_rag.py
```

*Note: The structure above reflects the current project files discussed. Keep any additional modules already present in your repository.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd "Advance RAG Project"
```

Replace `YOUR_GITHUB_REPOSITORY_URL` with your actual GitHub repository URL.

### 2. Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use your existing virtual environment or activate it using the appropriate method for your terminal.

### 3. Install Dependencies

Install the frontend dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install the backend dependencies:

```powershell
python -m pip install -r backend_requirements.txt
```

Ensure both requirements files exist in the project root.

### 4. Configure Environment Variables

Create a `.env` file in the project root and add the variables required by your implementation.

Example:

```env
GOOGLE_API_KEY=your_google_api_key
GOOGLE_MODEL=your_supported_gemini_model
```

Use the exact variable names expected by your `backend/config.py` and LangChain configuration. Do not commit real API keys or secrets to GitHub.

### 5. Start the FastAPI Backend

Your current backend entry point is `backend/main.py`.

```powershell
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Check the health endpoint:

http://127.0.0.1:8000/health

Expected response:

```json
{
  "status": "ok"
}
```

### 6. Start the Streamlit Frontend

Open a second terminal, activate the same virtual environment, and run:

```powershell
streamlit run app.py
```

Streamlit will display a local URL, normally:

http://localhost:8501

Ensure the frontend sends API requests to your local backend URL during local development.

## 🔌 API Endpoints

The following endpoints are based on the backend code discussed. Confirm the request schemas and additional routes in `backend/main.py`.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check backend health |
| POST | `/documents` | Upload a PDF document |
| POST | `/ask` | Ask a question about documents |
| GET | `/docs` | Open interactive API documentation |

The `/ask` endpoint should accept the request fields expected by your FastAPI implementation. The `/documents` endpoint should validate and process supported PDF uploads.

## 🔍 How RAG Works

1. **Document ingestion:** The user uploads a PDF through the frontend.
2. **Text extraction:** The backend extracts readable text from the PDF.
3. **Chunking:** The extracted text is divided into smaller chunks with suitable overlap.
4. **Embedding generation:** The embedding model converts chunks into numerical vectors.
5. **Vector storage:** The vectors and associated document metadata are stored in ChromaDB.
6. **Retrieval:** The system searches for relevant chunks based on the user's question.
7. **Context construction:** Retrieved passages are provided to the language model.
8. **Answer generation:** The LLM generates a response using the retrieved context.
9. **Response delivery:** FastAPI returns the result, and Streamlit displays it to the user.

## ☁️ Render Deployment

Deploy the project as two separate web services using the same GitHub repository.

### Backend Deployment — FastAPI

Create a new Render Web Service.

**Build Command**

```bash
pip install -r backend_requirements.txt
```

**Start Command**

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Add the necessary LLM API keys and other backend configuration variables in the Render Environment settings.

After deployment, your backend URL will resemble:

```text
https://your-rag-api.onrender.com
```

### Frontend Deployment — Streamlit

Create a second Render Web Service connected to the same repository.

**Build Command**

```bash
pip install -r requirements.txt
```

**Start Command**

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

Add the following environment variable to the frontend service:

```env
API_BASE_URL=https://your-rag-api.onrender.com
```

Replace the example URL with the actual backend URL.

Update `app.py` to read `API_BASE_URL` from the environment and use it for all backend requests. Do not use `localhost:8000` for requests between the two deployed services.

## 🔐 Security and Production Considerations

- Store API keys in environment variables or a secrets manager.
- Never commit `.env` files, credentials, or private documents.
- Validate uploaded file types and file sizes.
- Configure CORS only when required by your deployment architecture.
- Add request timeouts and useful error handling.
- Protect document upload and query endpoints if the application handles private information.
- Use persistent storage or an external vector database to retain indexed documents across redeployments.
- Avoid storing sensitive documents in a public repository.
- Configure OCR dependencies separately if scanned PDFs require Tesseract OCR.
- Pin and test compatible dependency versions before production deployment.

## 🧪 Testing

Run the project's test suite:

```powershell
python -m pytest -v
```

Check dependency consistency:

```powershell
python -m pip check
```

Test the API through `/docs`, then verify that document ingestion, retrieval, and answer generation work correctly from the Streamlit interface.

## 🚧 Troubleshooting

| Problem | Suggested Solution |
|---|---|
| `ModuleNotFoundError` | Install the missing package in the correct environment. |
| Uvicorn cannot import the application | Verify that `backend/main.py` defines `app = FastAPI()`. |
| `--port` requires an argument | Use `--port 8000` locally and `$PORT` on Render. |
| Backend connection failure | Verify `API_BASE_URL` and backend availability. |
| Gemini API error | Check the API key, supported model name, quotas, and provider configuration. |
| ChromaDB data disappears | Configure persistent storage or use an external vector database. |
| PDF extraction returns no text | Check whether the PDF is scanned and whether OCR is configured. |
| Dependency conflicts | Review package compatibility and pin tested versions. |

## 🎯 Future Enhancements

- Reranking for improved retrieval relevance.
- Source citations and page-level references in answers.
- Conversation history and follow-up questions.
- Authentication and document-level access control.
- Support for DOCX, TXT, and additional document formats.
- Background processing for large document collections.
- Retrieval evaluation and answer-quality metrics.
- Monitoring, structured logging, and usage analytics.

## 💼 Resume Project Description

**Enterprise RAG Research Assistant | Python, LangChain, FastAPI, Streamlit, ChromaDB**

Developed an AI-powered document question-answering system using Retrieval-Augmented Generation (RAG). Implemented PDF ingestion, text chunking, embedding generation, vector-based retrieval, and LLM-powered answer generation. Built a FastAPI backend and interactive Streamlit frontend, with support for cloud deployment and modular retrieval components.

## 👨‍💻 Author

**Shashank SIngh**

- GitHub: YOUR_GITHUB_PROFILE_URL


---

*This project is intended for educational, research, and enterprise document-assistance use cases. Replace the example configuration values and URLs with your actual deployment details.*
