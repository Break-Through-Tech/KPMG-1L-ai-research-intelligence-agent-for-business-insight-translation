import json

from ingestion.arxiv_client import MIN_SCORE, download_pdf, rank_papers, search_arxiv, find_and_store
from ingestion.chunker import chunk_documents
from ingestion.pdf_parser import extract_pdf
from rag.rag_pipeline import get_embedding_client, paper_is_stored, retrieve_document, store_documents
from summarize.summary import answer_with_summary

query = input("Enter Query: ")

paper = find_and_store(query)

# 3. Existing retrieval, now limited to this paper's chunks, then the existing summarizer
result = retrieve_document(query=query, k=1)
print(json.dumps(result[0][0].metadata, indent=4))
print(answer_with_summary(query, result))

# from ingestion.pdf_parser import extract_all_pdfs
# from ingestion.chunker import chunk_documents
# from rag.rag_pipeline import store_documents, retrieve_document
# import json


# documents = extract_all_pdfs('data')
# print("✅ Extracted all documents")

# chunks = chunk_documents(documents)
# print("✅ Completed Chunking")
# print(type(chunks))

# print(len(chunks))

# # Only run when storing new documents
# #store_documents(chunks)
# result = retrieve_document(query=input("Enter Query: "), k=1)

# print(json.dumps(result[0][0].metadata, indent=4))
