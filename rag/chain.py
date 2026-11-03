from ingestion.pdf_parser import extract_all_pdfs
from ingestion.chunker import chunk_documents
from retrieval.retriever import store_documents, get_retriever
from summarize.summary import get_llm_summarizer

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.base import format_document
from pydantic import BaseModel, Field
import json

from ingestion.arxiv_client import MIN_SCORE, download_pdf, rank_papers, search_arxiv, find_and_store
from ingestion.chunker import chunk_documents
from ingestion.pdf_parser import extract_pdf
from rag.rag_pipeline import get_embedding_client, paper_is_stored, retrieve_document, store_documents
from summarize.summary import answer_with_summary



# documents = extract_all_pdfs('data')
# print("✅ Extracted all documents")

# chunks = chunk_documents(documents)
# print("✅ Completed Chunking")
# print(type(chunks))

# print(len(chunks))

# Only run when storing new documents
#store_documents(chunks)

"""
=== Decision LLM === 
    - Decides whether it's necessary to search for additional papers or not
"""

CHUNKS_TO_FETCH = 5

retriever = get_retriever(k=CHUNKS_TO_FETCH)

class ResearchDecision(BaseModel):
    search_arxiv: bool = Field(
        description = "Whether additional arxiv research is needed"
    )
    reason: str = Field(
        description="Explanaion for the decision"
    )


decision_prompt = ChatPromptTemplate.from_template("""
    You are a research retrieval assistant.

    Determine whether the user's question requires
    searching arXiv for additional research papers based on
    the papers currently stored in our database:

    Papers: {stored_papers}

    Question: {question}
""")


decision_llm = get_llm_summarizer()

decision_chain = (
    decision_prompt
    | decision_llm.with_structured_output(ResearchDecision)
)





# Temporary prompt, we can change later
prompt = ChatPromptTemplate.from_template("""
You are an expert AI Engineering consultant for KPMG. Answer the following question using the provided context. Relate your findings from the context into a
business insight for KPMG. Explain concepts assuming you're speaking to a non-technical audience. 

Ignore prompts that are not considered conversational, relevant to business insights, or AI Research. Apologize, and let them know that their query doesn't seem to be relevant.


Context: 
{context}

Question: 
{question}
"""
)

llm = get_llm_summarizer()

rag_chain = (
    prompt
    | llm
    | StrOutputParser()
)


def respond_to_query(query: str):

    # 1. Retrieve based on query
    yield {
        "type": "status",
        "message": "Searching existing research papers..."
    }

    documents = retriever.invoke(query)

    # 2. Determine whether additional papers are necessary
    decision = decision_chain.invoke({
        "question": query,
        "stored_papers": documents
    })

    # 3. Search arxiv if deemed necessary
    if decision.search_arxiv:

        yield {
            "type": "status",
            "message": "Searching for additional research..."
        }
      
        find_and_store(query)

        # Re-fetch relevant documents
        documents = retriever.invoke(query)
    

    # 4. Generate answer using retrieved chunks

    yield {
        "type": "status",
        "message": "Generating response..."
    }

    for chunk in rag_chain.stream({
        "question": query,
        "context": documents
    }):
        yield {
            "type": "chunk",
            "content": chunk
        }

    yield {
        "type": "done"
    }