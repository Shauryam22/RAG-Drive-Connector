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
    
   
    system_prompt = f"""You are a factual assistant. 
    you must think about possible answer that you can derive from the chunks. 
    Any company name in the document , is considered as our,own for that document. 
    Give only 2 liner answers, dont find exact word, instead find synonymous, but should be correct and relevant.
    "
    
    CONTEXT:
    {context_text}
    """
    
    print("Sending context to LLM...")
    
    # 4. Generate the answer using the free Llama-3 model
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant", 
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