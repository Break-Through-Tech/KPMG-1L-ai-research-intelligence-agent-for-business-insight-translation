import json

from ingestion.arxiv_client import MIN_SCORE, download_pdf, rank_papers, search_arxiv
from ingestion.chunker import chunk_documents
from ingestion.pdf_parser import extract_pdf
from rag.rag_pipeline import get_embedding_client, paper_is_stored, retrieve_document, store_documents
from summarize.summary import answer_with_summary

query = input("Enter Query: ")

# 1. Find the one best paper on arXiv for this question
candidates = search_arxiv(query)
ranked = rank_papers(query, candidates, get_embedding_client())
if not ranked or ranked[0][1] < MIN_SCORE:
    print(f"No sufficiently relevant arXiv paper found ({len(candidates)} candidates screened).")
    raise SystemExit
paper, score = ranked[0]
print(f"✅ Selected: {paper.title} (arXiv:{paper.arxiv_id}, similarity {score:.2f}, "
      f"picked from {len(candidates)} candidates)")

# 2. Download it, then reuse the existing parse -> chunk -> store steps (only the first time)
pdf_path = download_pdf(paper)
if not paper_is_stored(paper.arxiv_id):
    chunks = chunk_documents(extract_pdf(pdf_path))
    for chunk in chunks:  # extra metadata so results can be cited
        chunk.metadata.update({
            "title": paper.title,
            "authors": ", ".join(paper.authors[:6]),
            "published": paper.published,
            "abs_url": paper.abs_url,
        })
    store_documents(chunks)
    print(f"✅ Stored {len(chunks)} chunks")

# 3. Existing retrieval, now limited to this paper's chunks, then the existing summarizer
result = retrieve_document(query=query, k=1, filter={"document_id": paper.arxiv_id})
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
