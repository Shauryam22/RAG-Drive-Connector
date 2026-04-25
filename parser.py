import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter




def process_doc(path_file,file_id,file_name):
    """Extracts the text from a doc, then chunk it.
    """
    print(f"Parsing :{file_name}")
    
    # loading the pdf
    
    loader = PyPDFLoader(file_path=path_file)
    doc = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
        separators=["\n"," ",""]
    )
    chunks = text_splitter.split_documents(doc)
    # metadata very imp , since where the info is located, is stored in metadata.
    # our chunk should have that, otherwise info loss
    formatted_chunks = []
    for chunk in chunks:
        # langchain will only grab source,page: we are adding file_id,file_name,hardcode the gdrive source.
        chunk.metadata.update(
            {
                'doc_id':file_id,
                'file_name':file_name,
                'source':'gdrive'

            }
        )
        formatted_chunks.append({
            'chunk_text':chunk.page_content,
            'metadata':chunk.metadata
        })
        
        
    print(f'Split into {len(formatted_chunks)} chunks')
    return formatted_chunks
    
    
    
if __name__=='__main__':
    
    file_path = "C:\\Users\\shaurya\\Downloads\\AI_Platform_Engineer_RAG_Assignment.pdf"
    if os.path.exists(file_path):
        results = process_doc(
            path_file=file_path,
            file_id= '1UCewiwsoEMmtmSzYFaotnoT_FJh45JgY',
            file_name="AI_Platform_Engineer_RAG_Assignment.pdf"
        )
        
        if results:
            print(f"{results[0]['chunk_text'][:200]}")
        
        