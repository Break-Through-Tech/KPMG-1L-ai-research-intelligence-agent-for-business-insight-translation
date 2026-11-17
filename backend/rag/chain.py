from ingestion.pdf_parser import extract_all_pdfs
from ingestion.chunker import chunk_documents
from retrieval.retriever import store_documents, get_retriever, get_vector_store
from summarize.summary import get_llm_summarizer

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.base import format_document
from typing import Literal
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
    message_category: Literal["conversational", "new_question", "follow_up"] = Field(
        description = "The category of user message, between conversational, new_question, and follow_up"
    )
    search_arxiv: bool = Field(
        description = "Whether additional arxiv research is needed"
    )
    reason: str = Field(
        description="Explanation for the decision"
    )


decision_prompt = ChatPromptTemplate.from_template("""
    You are a research retrieval assistant.

    First, determine the category of the user's message. Select one of the following categories:
        1. "conversational" - the message is just conversational, no research is needed
        2. "new_question" - the user asks a new question that may need a scan of the papers
        3. "follow_up" - the user asks a follow up question, another research scan may not be necessary

    If the prompt is non-conversational, determine whether the user's question requires
    searching arXiv for additional research papers based on
    the papers currently stored in our database, otherwise decide False:

    Papers: {stored_papers}

    Conversation so far:
    {history}

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

If the user prompt is conversational, respond in a friendly, relevant way. Explain your role, and ways you can help.

Ignore prompts that are not considered conversational, relevant to business insights, or AI Research. Apologize, and let them know that their query doesn't seem to be relevant.


Conversation so far:
{history}

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


def get_stored_paper_titles() -> list[str]:
    """List the distinct titles of stored papers without running a similarity search."""
    try:
        metadatas = get_vector_store().get(include=["metadatas"])["metadatas"]
    except Exception:   # no DB or collection yet
        return []
    return sorted({m["title"] for m in metadatas if m and m.get("title")})


def format_history(history: list) -> str:
    """Render prior conversation messages (oldest first) for use in prompts.

    Each message is expected to be an object like {"role": ..., "content": ...}
    (as sent by the frontend); plain strings are also accepted.
    """
    if not history:
        return "(no previous messages)"

    lines = []
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        else:
            lines.append(str(msg))
    return "\n".join(lines)


def respond_to_query(query: str, history: list):
    history = format_history(history)
    print(history)

    # 1. Classify the message and assess whether additional research is needed
    decision = decision_chain.invoke({
        "question": query,
        "history": history,
        "stored_papers": get_stored_paper_titles()
    })
    print(f"{decision.message_category=}")

    documents = []

    # 2. Only search documents if the message isn't conversational
    if decision.message_category != "conversational":
        yield {
            "type": "status",
            "message": "Searching existing research papers..."
        }

        documents = retriever.invoke(query)

        # 3. Search arxiv if deemed necessary
        if decision.search_arxiv:
            yield {
                "type": "status",
                "message": "Searching for additional research..."
            }

            find_and_store(query)

            # Re-fetch relevant documents
            documents = retriever.invoke(query)

    # 4. Generate answer (with no context for conversational messages)
    yield {
        "type": "status",
        "message": "Generating response..."
    }

    for chunk in rag_chain.stream({
        "question": query,
        "history": history,
        "context": documents
    }):
        yield {
            "type": "chunk",
            "content": chunk
        }

    yield {
        "type": "done"
    }
