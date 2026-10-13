"""Live research flow on top of the existing pipeline.

question -> Gemini query variants -> arXiv search -> re-rank abstracts -> download top papers
-> parse/chunk/store ONLY papers not already in Chroma -> retrieve ONLY from this question's
papers -> Gemini summary with sources.

Reuses existing code: extract_pdf, chunk_documents, store_documents, get_embedding_client and the
Chroma settings (PERSIST_DIR, COLLECTION_NAME).
"""
import numpy as np
from langchain_chroma import Chroma

from arxiv_search import find_papers
from ingestion.chunker import chunk_documents
from ingestion.pdf_parser import extract_pdf
from rag.rag_pipeline import (
    COLLECTION_NAME,
    PERSIST_DIR,
    get_embedding_client,
    store_documents,
)
from summarizer import rewrite_queries, summarize


def _open_store(embeddings) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
        relevance_score_fn=lambda distance: 1.0 - (distance / 2.0),  # same as retrieve_document
    )


def _is_stored(store: Chroma, doc_id: str) -> bool:
    """Ask Chroma itself, so there's no separate state file to go stale if chroma_db is deleted."""
    return bool(store.get(where={"document_id": doc_id}, limit=1)["ids"])


def _ingest_new(papers, store: Chroma) -> list[str]:
    chunks, added = [], []
    for p in papers:
        doc_id = p.pdf_path.stem  # extract_pdf stores this as metadata["document_id"]
        if _is_stored(store, doc_id):
            continue
        pages = extract_pdf(p.pdf_path)
        if not pages:
            print(f"  Warning: no extractable text in {p.arxiv_id} (scanned PDF?), skipping")
            continue
        chunks.extend(chunk_documents(pages))
        added.append(p.arxiv_id)
    if chunks:
        store_documents(chunks)
    return added


def answer(question: str, days: int = 30, top_papers: int = 3, k_chunks: int = 5) -> str:
    embeddings = get_embedding_client()  # load MiniLM once and reuse it everywhere below
    embed_fn = lambda texts: np.array(embeddings.embed_documents(texts))

    queries = rewrite_queries(question)
    print(f"Search queries: {queries}")

    papers = find_papers(question, query_variants=queries, days=days,
                         top_k=top_papers, embed_fn=embed_fn)
    if not papers:
        return "No recent arXiv papers matched. Try a wider date window or different wording."
    for p in papers:
        print(f"  {p.score:.2f}  {p.arxiv_id}  {p.title}")

    store = _open_store(embeddings)
    added = _ingest_new(papers, store)
    print(f"Newly ingested: {added or 'none (all cached)'}")

    ids = [p.pdf_path.stem for p in papers]
    titles = {p.pdf_path.stem: p.title for p in papers}
    results = store.similarity_search_with_relevance_scores(
        query=question, k=k_chunks, filter={"document_id": {"$in": ids}}
    )
    if not results:
        return "Papers were found but no readable text could be retrieved from them."

    blocks = [
        f"[{titles.get(d.metadata['document_id'], d.metadata['document_id'])}, "
        f"arXiv:{d.metadata['document_id']}, p.{d.metadata['page_label']}]\n{d.page_content}"
        for d, _score in results
    ]
    sources = "\n".join(f"- {p.title} (https://arxiv.org/abs/{p.arxiv_id})" for p in papers)
    return f"{summarize(question, blocks)}\n\nSources:\n{sources}"


if __name__ == "__main__":
    print(answer(input("Enter Query: ")))
