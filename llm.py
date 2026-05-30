from dotenv import load_dotenv

# GENERATION PHASE

load_dotenv()
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

def format_docs(retrieved_docs):
    context_text = '\n\n'.join([doc.page_content for doc in retrieved_docs])
    return context_text

def generate_rag_answer(query,retriever):
   
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system","""You are a factual assistant. 
                    you must think about possible answer that you can derive from the chunks. 
                    Any company name in the document , is considered as our,own for that document. 
                    Give only 2 liner answers, dont find exact word, instead find synonymous, but should be correct and relevant.
                    {context}"""),
        ("human","""question: {question}"""),
    ]
    )
    
    # when we invoke parallel_chain, both context and question will get the input query, context will apply retriever and formatting , and question just takes input as it is.
    # parallel_chain = RunnableParallel(
    #     {
    #         'context':retriever|RunnableLambda(format_docs),   
    #         'question': RunnablePassthrough()
    #     }
    # )
    parallel_chain = RunnableParallel(
        {
            'docs':retriever,   
            'question': RunnablePassthrough()
        }
    )
    parser = StrOutputParser()
    # main_chain = parallel_chain|prompt|llm|parser
    main_chain = parallel_chain | {
        
        # Path A: Generate the Answer
        "answer": (
            RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"]}) 
            | prompt 
            | llm 
            | parser
        ),
        
        # Path B: Extract the Sources
        "sources": RunnableLambda(lambda x: list(set([doc.metadata.get('file_name', 'Unknown') for doc in x["docs"]])))
    }
    
    return main_chain.invoke(query)
    
    
    
    
    