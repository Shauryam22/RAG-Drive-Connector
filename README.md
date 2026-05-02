# RAG-DRIVE-CONNECTOR-PIPELINE
<!-- RESUME: ADD -->

## Overview
This repository contains a fully automated RAG pipeline designed to sync documents directly from Google Drive, process them into a local vector knowledge base, and serve highly accurate, grounded answers using Meta's Llama-3.1 model. The application is served via a RESTful API built with FastAPI and is structured for production-ready containerized deployment.

### Pipeline Flow

```
Google Drive Folder
        │
        │  (Service Account Auth)
        ▼
  List & Download Files
        │
        ▼
  Extract Text (PyPDFLoader)
        │
        ▼
  Chunk Text  ──── chunk_size=1000, overlap=200
  + Attach Metadata (doc_id, file_name, source=gdrive)
        │
        ▼
  Generate Embeddings (all-MiniLM-L6-v2)
        │
        ▼
  Store in FAISS Index
        │
       ═══════════════════════════  (On /ask)
        │
  User Query ──► Embed Query ──► FAISS Search (top-3 chunks)
        │
        ▼
  Build Context from Retrieved Chunks
        │
        ▼
  Groq LLM (llama-3.1-8b-instant)
        │
        ▼
  { "answer": "...", "sources": [...] }
```


## Project Structure

```
RAG-Drive-Connector/
│
├── api.py           # FastAPI app — exposes /sync-drive and /ask endpoints
├── main.py          # Orchestrates the full ingestion pipeline
├── drive.py         # Google Drive auth, file listing, and download
├── parser.py        # PDF text extraction and chunking with metadata
├── vector.py        # FAISS vector store — add and search embeddings
├── llm.py           # Groq LLM integration — generates RAG answers
│
├── cred.json        # Google Service Account credentials (DO NOT COMMIT)
├── .env             # Environment variables (DO NOT COMMIT)
├── .gitignore       # Excludes credentials and venv
│
└── downloads/       # Temporary folder for downloaded Drive files
```

---

## Setup & Installation

### Prerequisites

- Python 3.9+
- A Google Cloud project with the **Google Drive API** enabled
- A **Service Account** with access to the target Drive folder
- A **Groq API key** (free at [console.groq.com](https://console.groq.com))

### 1. Clone the Repository

```bash
git clone https://github.com/Shauryam22/RAG-Drive-Connector.git
cd RAG-Drive-Connector
```

### 2. Create & Activate a Virtual Environment

```bash
python -m venv myenv

# Windows
myenv\Scripts\activate

# macOS / Linux
source myenv/bin/activate
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn python-dotenv \
    google-auth google-api-python-client \
    langchain langchain-community pypdf \
    sentence-transformers faiss-cpu \
    groq
```

---

## Configuration

### 1. Google Service Account (`cred.json`)

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → **IAM & Admin** → **Service Accounts**
2. Create a service account and download the JSON key as `cred.json`
3. Place `cred.json` in the project root
4. Share your target Google Drive folder with the service account's email address (give **Viewer** access)

### 2. Environment Variables (`.env`)

Create a `.env` file in the project root:

```env
# Path to your Google Service Account credentials file
SERVICE_FILE=cred.json

# The ID of your Google Drive folder to sync from
# (Get this from the folder's URL: drive.google.com/drive/folders/<FOLDER_ID>)
FOLDER_ID=your_google_drive_folder_id_here

# Your Groq API key
GROQ_API_KEY=your_groq_api_key_here
```

---

## Running the API

```bash
uvicorn api:app --reload
```

The API will be live at: **`http://127.0.0.1:8000`**

Interactive docs available at: **`http://127.0.0.1:8000/docs`**

---

## API Reference

### `POST /sync-drive`

Connects to Google Drive, downloads documents from the configured folder, chunks and embeds them, and loads everything into the in-memory FAISS vector store.

**Request:** No body required.

**Response:**
```json
{
  "message": "Success! Synced and embedded 42 chunks from Google Drive."
}
```

---

### `POST /ask`

Takes a natural language question, retrieves the top 3 most semantically relevant chunks from the vector store, and returns a grounded answer with source citations.

**Request:**
```json
{
  "query": "What is our refund policy?"
}
```

**Response:**
```json
{
  "answer": "Customers can return products within 30 days of purchase for a full refund.",
  "sources": ["company_policy.pdf"]
}
```

---

## Sample Queries & Outputs

> **Setup:** The Google Drive folder contained multiple pdfs. After calling `/sync-drive`, the documents were split into **chunks** and embedded into FAISS.

---

**Query Test**

```json
POST /ask
{ "query": "what is the architecture required in assignment??" }
```
```json
{
  "answer": "The expected architecture for the assignment involves connecting various components to process and answer questions from documents. It includes connectors to Google Drive, processing for parsing and chunking, embedding, search, and API layers.",
  "sources": [
    "Complaints-Policy-and-Procedure.pdf",
    "AI_Platform_Engineer_RAG_Assignment.pdf"
  ]
}

