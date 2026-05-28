# import faiss
# import numpy as np
# from sentence_transformers import SentenceTransformer


# class DocStore:
#     def __init__(self):
#         self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
#         self.dim = 384
#         self.index = faiss.IndexFlatL2(self.dim)
#         self.chunk_data = []
        
#     def add_chunks(self,formatted_chunks):
#         if not formatted_chunks:
#             return
#         print('Total chunks: ',len(formatted_chunks))
#         text = [i['chunk_text'] for i in formatted_chunks]# extracting the text only.
#         # to be fed into Transformer
#         embeddings = self.embedder.encode(text) # returns a numpy array.
        
#         self.index.add(np.array(embeddings))
#         self.chunk_data.extend(formatted_chunks)
        
#         print(self.index.ntotal)
        
#     def search(self,query,top_k=2):
#         if self.index.ntotal==0:
#             print("EMPTY DB")
#             return []
#         query_vector = self.embedder.encode([query])
#         clean_query_vector = np.array(query_vector).astype('float32')
        
#         distances, indices = self.index.search(clean_query_vector, top_k)
#         results = []
#         for i in range(top_k):
#             vector_id = indices[0][i]
            
#             # FAISS returns -1 if it can't find enough matches (e.g., you asked for top 5 but only have 2 chunks)
#             if vector_id != -1: 
#                 chunk_info = self.chunk_data[vector_id]
#                 results.append(chunk_info)
                
#         return results
        
        
# if __name__ == '__main__':
    
#     sample_chunks = [
#         {
#             "chunk_text": "The company refund policy allows returns within 30 days of purchase.",
#             "metadata": {"file_name": "policy.pdf", "doc_id": "123", "source": "gdrive"}
#         },
#         {
#             "chunk_text": "Employees get 15 days of paid time off per year.",
#             "metadata": {"file_name": "pto.pdf", "doc_id": "456", "source": "gdrive"}
#         }
#     ]
    
#     store = DocStore()
    
#     # Add the chunks
#     store.add_chunks(sample_chunks)
#     question = "How many vacation days do I get?"
#     #question = 'Is there any refund?'
#     found_chunks = store.search(question, top_k=1)
    
#     if found_chunks:
#         print(f"Question: {question}")
#         print(f"Best Match: {found_chunks[0]['chunk_text']}")
#         print(f"Source: {found_chunks[0]['metadata']['file_name']}")
    

# INDEXING AND BOTH RETRIEVAL(Prep) PHASE:

from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()
class DocStore:
    def __init__(self):
        self.index = None
        self.embedder = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    # Actual Indexing function
    def add_chunks(self,formatted_chunks):
        if not formatted_chunks:
            return
        #for chunk in formatted_chunks:
            #print(chunk['chunk_text'])
        
        docs = [
            Document(
                page_content=chunk['chunk_text'],
                metadata = chunk['metadata']
                
            )
            for chunk in formatted_chunks
        ]
        
        if self.index is None:
            self.index = FAISS.from_documents(docs,embedding=self.embedder) # initializing vector store
            
        else:
            # self.index = FAISS.add_documents(documents = docs)  THIS IS WRONG!!
            # THis will reinitialise the vector store and since to initialize the vectorstore we first need from_doc, hence it will show error.
            
            self.index.add_documents(docs)
        
        
        print(f"Total vectors in VectorStore: {self.index.index.ntotal}")
    # Actual Retrieval function 
    def get_retriever(self,top_k=5):
        
        if self.index is None:
            print('Vector DB is empty.')
            return
        retriever = self.index.as_retriever(search_type='similarity',search_kwargs={'k':top_k})
        
        return retriever # we have to give out the retriever that is aware of our VDB only.
        # this retriever has top_k stored in it. Dont worry about to put it on Docs.
        # similarity is encoded in this retriever
        
        
        
if __name__ =='__main__':        
    docst = DocStore()
    formt = [ {
                "chunk_text": "The company refund policy allows returns within 30 days of purchase.",
                "metadata": {"file_name": "policy.pdf", "doc_id": "123", "source": "gdrive"}
            }]
    docst.add_chunks(formt)

    print(docst.get_retriever())


        
            
        