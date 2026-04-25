import os
from groq import Groq
from dotenv import load_dotenv


load_dotenv()
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

def generate_rag_answer(query_text, retrieved_chunks):
  
    context_text = "\n\n---\n\n".join([chunk['chunk_text'] for chunk in retrieved_chunks])
    
    # Extracting the unique file names for our 'sources' list
    # We use a set() so if multiple chunks came from the same PDF, it only lists it once
    sources = list(set([chunk['metadata']['file_name'] for chunk in retrieved_chunks]))
    
   
    system_prompt = f"""You are a strict, factual assistant. 
    You must answer the user's question using ONLY the provided context below. 
    If the answer is not explicitly contained within the context, do not guess or use outside knowledge. 
    Instead, reply exactly with: "I cannot answer this based on the provided documents."
    
    CONTEXT:
    {context_text}
    """
    
    print("Sending context to LLM...")
    
    # 4. Generate the answer using the free Llama-3 model
    response = client.chat.completions.create(
        model="llama3-8b-8192", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_text}
        ],
        temperature=0.0 # Setting temperature to 0 removes creative hallucination
    )
    
    answer_text = response.choices[0].message.content
    
    return {
        "answer": answer_text,
        "sources": sources
    }


if __name__ == '__main__':
    # Simulating FAISS search() function o/p
    mock_retrieved_chunks = [
        {
            "chunk_text": "The trial assignment duration is 48-72 hours. Deliverables include a GitHub repo and a README.",
            "metadata": {"file_name": "AI_Platform_Engineer_RAG_Assignment.pdf", "doc_id": "123", "source": "gdrive"}
        }
    ]
    
    test_query = "What are the deliverables for this assignment?"
    
    final_output = generate_rag_answer(test_query, mock_retrieved_chunks)
    
    print("\n--- FINAL RAG RESPONSE ---")
    import json
    print(json.dumps(final_output, indent=2))