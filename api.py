from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import run_sync_pipeline 
from vector import DocStore        # vector  
from llm import generate_rag_answer   #llm

app = FastAPI(title="Highwatch AI RAG Pipeline")


store = DocStore()

# Pydantic model to match the exact JSON input requested by the assignment
class QueryRequest(BaseModel):
    query: str

@app.post("/sync-drive")
def sync_drive():
    """
    Connects to Google Drive, fetches documents, processes/chunks them, 
    and generates embeddings to store in the knowledge base.
    """
    try:
        # Run the ingestion pipeline (Auth -> Download -> Parse)
        chunks = run_sync_pipeline()
        
        if not chunks:
            return {"message": "No documents found in Google Drive."}
            
        # 2. Add the processed chunks to the FAISS vector store
        store.add_chunks(chunks)
        
        return {"message": f"Success! Synced and embedded {len(chunks)} chunks from Google Drive."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(request: QueryRequest):
    """
    Takes a user query, retrieves relevant chunks, and generates a grounded answer.
    """
    try:
        # Search the vector database for the top 3 most relevant chunks
        retrieved_chunks = store.search(request.query, top_k=3)
        
        if not retrieved_chunks:
            return {
                "answer": "The knowledge base is empty. Please call /sync-drive first.",
                "sources": []
            }
            
        # Pass the question and the context to the LLM
        response = generate_rag_answer(request.query, retrieved_chunks)
        
        # Return the exact JSON format requested in the assignment
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))