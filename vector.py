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


        
            
        
