from ingestion.pdf_parser import extract_all_pdfs
from ingestion.chunker import chunk_documents
from retrieval.retriever import store_documents, get_retriever
from summarize.summary import get_llm_summarizer

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate



# documents = extract_all_pdfs('data')
# print("✅ Extracted all documents")

# chunks = chunk_documents(documents)
# print("✅ Completed Chunking")
# print(type(chunks))

# print(len(chunks))

# Only run when storing new documents
#store_documents(chunks)

retriever = get_retriever()

# Temporary prompt, we can change later
prompt = ChatPromptTemplate.from_template("""
You are an expert AI Engineering consultant for KPMG. Answer the following question using the provided context. Relate your findings from the context into a
business insight for KPMG. Explain concepts assuming you're speaking to a non-technical audience. 

Context: 
{context}

Question: 
{question}
"""
)

llm = get_llm_summarizer()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# # Response all at once
# response = rag_chain.invoke(input("Enter Query: "))
# print(response)

# Streamed response
for chunk in rag_chain.stream(input("Enter Query: ")):
    print(chunk, end="", flush=True)