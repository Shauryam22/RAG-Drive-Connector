from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import run_sync_pipeline 
from vector import DocStore        # vector  
from llm import generate_rag_answer   #llm
from time import time
app = FastAPI(title="RAG Pipeline")


store = DocStore()
# query_log=[]
query_log = []

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
        start = time()  # ← start timer

        # Search the vector database for the top 3 most relevant chunks
        retriever = store.get_retriever(top_k=3)
        

        if store.index is None:
            return {
                "answer": "The knowledge base is empty. Please call /sync-drive first.",
                "sources": []
            }

        # Pass the question and the context to the LLM
        # AUGMENTATION PHASE
        response = generate_rag_answer(request.query, retriever)

        # Track latency
        latency_ms = round((time() - start) * 1000, 2)  # ← end timer
        query_log.append(latency_ms)
        response["latency_ms"] = latency_ms  # ← attach to response

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_metrics():
    if not query_log:
        return {"message": "No queries yet"}
    vectors_indexed = store.index.index.ntotal if store.index is not None else 0
        
    return {
        "total_queries": len(query_log),
        "avg_latency_ms": round(sum(query_log)/len(query_log), 2),
        "p95_latency_ms": round(sorted(query_log)[int(len(query_log)*0.95)], 2),
        "vectors_indexed": vectors_indexed
    }