import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class DocStore:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dim = 384
        self.index = faiss.IndexFlatL2(self.dim)
        self.chunk_data = []
        
    def add_chunks(self,formatted_chunks):
        if not formatted_chunks:
            return
        print('Total chunks: ',len(formatted_chunks))
        text = [i['chunk_text'] for i in formatted_chunks]# extracting the text only.
        # to be fed into Transformer
        embeddings = self.embedder.encode(text) # returns a numpy array.
        
        self.index.add(np.array(embeddings))
        self.chunk_data.extend(formatted_chunks)
        
        print(self.index.ntotal)
        
    def search(self,query,top_k=2):
        if self.index.ntotal==0:
            print("EMPTY DB")
            return []
        query_vector = self.embedder.encode([query])
        clean_query_vector = np.array(query_vector).astype('float32')
        
        distances, indices = self.index.search(clean_query_vector, top_k)
        results = []
        for i in range(top_k):
            vector_id = indices[0][i]
            
            # FAISS returns -1 if it can't find enough matches (e.g., you asked for top 5 but only have 2 chunks)
            if vector_id != -1: 
                chunk_info = self.chunk_data[vector_id]
                results.append(chunk_info)
                
        return results
        
        
if __name__ == '__main__':
    
    sample_chunks = [
        {
            "chunk_text": "The company refund policy allows returns within 30 days of purchase.",
            "metadata": {"file_name": "policy.pdf", "doc_id": "123", "source": "gdrive"}
        },
        {
            "chunk_text": "Employees get 15 days of paid time off per year.",
            "metadata": {"file_name": "pto.pdf", "doc_id": "456", "source": "gdrive"}
        }
    ]
    
    store = DocStore()
    
    # Add the chunks
    store.add_chunks(sample_chunks)
    question = "How many vacation days do I get?"
    #question = 'Is there any refund?'
    found_chunks = store.search(question, top_k=1)
    
    if found_chunks:
        print(f"Question: {question}")
        print(f"Best Match: {found_chunks[0]['chunk_text']}")
        print(f"Source: {found_chunks[0]['metadata']['file_name']}")
    
    
        
        