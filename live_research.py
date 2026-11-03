"""Live research flow on top of the existing pipeline.

question -> Gemini query variants -> arXiv search -> re-rank abstracts -> download top papers
-> parse/chunk/store ONLY papers not already in Chroma -> retrieve ONLY from this question's
papers -> Gemini summary with sources.

Reuses existing code: extract_pdf, chunk_documents, store_documents, get_embedding_client and the
Chroma settings (PERSIST_DIR, COLLECTION_NAME).

run()    -> structured dict (used by evaluate.py)
answer() -> just the text

`python live_research.py` asks questions in a loop; each answer is auto-checked (faithful,
relevant, business usefulness, real citations) and logged to logs/live_questions.jsonl.
`python evaluate.py --report-log` summarizes that log.
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from langchain_chroma import Chroma

from arxiv_search import DEFAULT_DAYS, find_papers
from ingestion.chunker import chunk_documents
from ingestion.pdf_parser import extract_pdf
from rag.rag_pipeline import (
    COLLECTION_NAME,
    PERSIST_DIR,
    get_embedding_client,
    store_documents,
)
from summarizer import is_no_answer, rewrite_queries, summarize

# Abstain when the best abstract similarity is below this. PLACEHOLDER (same starting value as
# MIN_SCORE in ingestion/arxiv_client.py); tune it from the score table evaluate.py prints.
MIN_PAPER_SCORE = 0.30
REFUSAL = ("The papers found don't directly address this question, so I can't answer it "
           "from the research without guessing.")


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


def ingest_paths(pdf_paths: list[Path], store: Chroma) -> list[str]:
    """Parse/chunk/store any PDFs not already in Chroma. Returns the document_ids added."""
    chunks, added = [], []
    for path in pdf_paths:
        doc_id = Path(path).stem  # extract_pdf stores this as metadata["document_id"]
        if _is_stored(store, doc_id):
            continue
        pages = extract_pdf(path)
        if not pages:
            print(f"  Warning: no extractable text in {doc_id} (scanned PDF?), skipping")
            continue
        chunks.extend(chunk_documents(pages))
        added.append(doc_id)
    if chunks:
        store_documents(chunks)
    return added


def retrieve_and_summarize(question: str, store: Chroma, doc_ids: list[str],
                           titles: dict[str, str], k_chunks: int) -> tuple[list, list[str], str]:
    """Retrieve chunks from ONLY the given documents, then summarize. Shared with evaluate.py."""
    results = store.similarity_search_with_relevance_scores(
        query=question, k=k_chunks, filter={"document_id": {"$in": doc_ids}}
    )
    if not results:
        return [], [], ""
    blocks = [
        f"[{titles.get(d.metadata['document_id'], d.metadata['document_id'])}, "
        f"arXiv:{d.metadata['document_id']}, p.{d.metadata['page_label']}]\n{d.page_content}"
        for d, _score in results
    ]
    return results, blocks, summarize(question, blocks)


def run(question: str, days: int = DEFAULT_DAYS, top_papers: int = 3, k_chunks: int = 5,
        min_score: float = MIN_PAPER_SCORE) -> dict:
    """Full live flow. Returns everything the evaluator needs, not just the text."""
    out = {"queries": [], "papers": [], "results": [], "blocks": [],
           "answer": "", "abstained": False, "abstain_reason": None}

    embeddings = get_embedding_client()  # load MiniLM once and reuse it everywhere below
    embed_fn = lambda texts: np.array(embeddings.embed_documents(texts))

    out["queries"] = queries = rewrite_queries(question)
    print(f"Search queries: {queries}")

    papers = find_papers(question, query_variants=queries, days=days,
                         top_k=top_papers, embed_fn=embed_fn)
    out["papers"] = papers
    if not papers:
        out.update(abstained=True, abstain_reason="no_papers", answer="No recent arXiv papers matched. "
                   "Try a wider date window or different wording.")
        return out
    for p in papers:
        print(f"  {p.score:.2f}  {p.arxiv_id}  {p.title}")
    if papers[0].score < min_score:
        out.update(abstained=True, abstain_reason="low_paper_score", answer=(
            f"No sufficiently relevant papers found (best match {papers[0].score:.2f} "
            f"< {min_score:.2f}), so I can't answer this from recent research."))
        return out

    store = _open_store(embeddings)
    added = ingest_paths([p.pdf_path for p in papers], store)
    print(f"Newly ingested: {added or 'none (all cached)'}")

    ids = [p.pdf_path.stem for p in papers]
    titles = {p.pdf_path.stem: p.title for p in papers}
    results, blocks, summary = retrieve_and_summarize(question, store, ids, titles, k_chunks)
    out.update(results=results, blocks=blocks)
    if not results:
        out.update(abstained=True, abstain_reason="no_text",
                   answer="Papers were found but no readable text could be retrieved from them.")
        return out

    if is_no_answer(summary):
        out.update(abstained=True, abstain_reason="insufficient_evidence", answer=REFUSAL)
        return out

    sources = "\n".join(f"- {p.title} (https://arxiv.org/abs/{p.arxiv_id})" for p in papers)
    out["answer"] = f"{summary}\n\nSources:\n{sources}"
    return out


def answer(question: str, **kwargs) -> str:
    return run(question, **kwargs)["answer"]


# ---------------- live grading + logging ----------------
# Real user questions have no "correct paper" label, so the question bank in evaluate.py can't
# score them. These checks need no label: faithfulness, relevance, business usefulness (LLM
# judge), made-up citations and format. Every question is appended to the log so the team (and
# KPMG reviewers) can look back at real usage; `python evaluate.py --report-log` summarizes it.
LOG_PATH = Path(__file__).resolve().parent / "logs" / "live_questions.jsonl"


def _grade(question: str, paper_ids: list[str], blocks: list[str], answer_text: str,
           abstained: bool) -> dict:
    """Label-free checks on one live answer. Reuses evaluate.py's checks so scores match."""
    from evaluate import cited_ids_not_retrieved, format_ok, judge, norm

    grades = {"hallucinated_ids": cited_ids_not_retrieved(answer_text, {norm(i) for i in paper_ids})}
    if abstained:
        return grades  # nothing to judge; the log keeps abstain_reason
    grades["format_ok"] = format_ok(answer_text)
    grades.update(judge(question, blocks, answer_text))
    return grades


def grade(question: str, out: dict) -> dict:
    return _grade(question, [p.arxiv_id for p in out["papers"]], out["blocks"],
                  out["answer"], out["abstained"])


def grade_record(r: dict) -> dict:
    """Same checks on an entry read back from the log (used by evaluate.py --report-log)."""
    return _grade(r["question"], [p["arxiv_id"] for p in r["papers"]], r["blocks"],
                  r["answer"], r["abstained"])


def log_question(question: str, out: dict, latency_s: float, grades: dict | None) -> None:
    record = {
        "asked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "search_queries": out["queries"],
        "papers": [{"arxiv_id": p.arxiv_id, "title": p.title, "score": round(p.score, 3)}
                   for p in out["papers"]],
        "abstained": out["abstained"],
        "abstain_reason": out["abstain_reason"],
        "answer": out["answer"],
        "blocks": out["blocks"],     # kept so the answer can be (re)graded later
        "latency_s": latency_s,
        "grades": grades,            # None if run with --no-grade
    }
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def print_grades(grades: dict) -> None:
    tick = lambda ok: "yes" if ok else "NO"
    parts = [f"citations real: {tick(not grades['hallucinated_ids'])}"]
    if "faithful" in grades:
        parts = [f"faithful: {tick(grades['faithful'])}",
                 f"answers question: {tick(grades['relevant'])}",
                 f"business usefulness: {grades['business_useful']}/5"] + parts
    print("\n--- Auto-check: " + " | ".join(parts))
    if grades.get("reason"):
        print(f"    judge: {grades['reason']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ask research questions; each is graded and logged.")
    ap.add_argument("--no-grade", action="store_true",
                    help="skip the auto-check (saves one Gemini call per question)")
    args = ap.parse_args()

    while True:
        q = input("\nEnter Query (blank to quit): ").strip()
        if not q:
            break
        t0 = time.time()
        out = run(q)
        latency = round(time.time() - t0, 1)
        print(f"\n{out['answer']}")
        grades = None if args.no_grade else grade(q, out)
        if grades:
            print_grades(grades)
        log_question(q, out, latency, grades)
        print("(logged to logs/live_questions.jsonl)")
