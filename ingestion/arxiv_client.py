"""Everything that talks to arXiv: search for relevant candidate papers, rank them, download one PDF.

This file has NO dependency on the rest of the project. The embedding model is passed in
by the caller (main.py), so ingestion/ never has to import from rag/.
"""
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from rag.rag_pipeline import get_embedding_client, paper_is_stored, retrieve_document, store_documents
from ingestion.chunker import chunk_documents
from ingestion.pdf_parser import extract_pdf

import numpy as np
import requests


API_URL = "https://export.arxiv.org/api/query"
HEADERS = {"User-Agent": "kpmg-research-rag/0.1 (contact: michellehongx@gmail.com)"}  # edit 
MIN_INTERVAL_S = 3.0          # arXiv asks API users to wait ~3 s between requests
PDF_DIR = Path(__file__).resolve().parent.parent / "data" / "arxiv"   # separate from data/*.pdf

# --- Scope knobs (this is where the business focus lives) -------------------------------
# arXiv subject areas that cover the use cases in the prompt library. Verify at
# https://arxiv.org/category_taxonomy and edit as a team.
CATEGORIES = ["cs.CL", "cs.IR", "cs.AI", "cs.LG", "cs.CR", "cs.CY", "cs.HC", "cs.MA", "cs.SE", "cs.DB"] # add more later
MONTHS_BACK = 24              # only consider papers submitted in the last N months
MAX_CANDIDATES = 25           # how many papers to pull from arXiv before ranking
MIN_SCORE = 0.30              # PLACEHOLDER: below this the best paper is "not relevant enough"

ATOM = "{http://www.w3.org/2005/Atom}"
STOPWORDS = {
    "what", "which", "when", "where", "how", "does", "that", "this", "with", "from", "have",
    "into", "about", "between", "research", "paper", "papers", "study", "studies", "find",
    "give", "summarize", "explain", "overview", "recent", "latest", "most", "relevant", "main",
    "approaches", "evidence", "using", "used", "current", "practical", "based", "show",
    "identify", "compare", "comparing", "summary", "review", "other", "their", "there",
    "these", "those", "should", "could", "would", "also", "across", "within", "without",
    "systems", "techniques", "methods", "work",
}


@dataclass
class Paper:
    arxiv_id: str      # e.g. "2301.07041" (no version suffix)
    title: str
    abstract: str
    authors: list[str]
    published: str     # YYYY-MM-DD
    abs_url: str
    pdf_url: str


_last_call = 0.0


def _wait_turn() -> None:
    """Sleep just long enough that requests to arXiv are >= MIN_INTERVAL_S apart."""
    global _last_call
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


def extract_keywords(question: str, limit: int = 8) -> list[str]:
    """arXiv search is keyword-based, so reduce the question to its meaningful words."""
    seen, keywords = set(), []
    for w in re.findall(r"[a-z]+", question.lower()):
        if len(w) > 3 and w not in STOPWORDS and w not in seen:
            seen.add(w)
            keywords.append(w)
    return keywords[:limit]


def build_query(keywords: list[str]) -> str:
    """(abs:"w1" OR abs:"w2" ...) AND (cat:... OR cat:...) AND submittedDate:[start TO end]
    OR across keywords is deliberately broad; rank_papers() restores precision afterwards.
    Dates are whole days so the same question always yields the same query string."""
    words = " OR ".join(f'abs:"{k}"' for k in keywords)
    cats = " OR ".join(f"cat:{c}" for c in CATEGORIES)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=30 * MONTHS_BACK)
    return f"({words}) AND ({cats}) AND submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"


def _clean(el, tag: str) -> str:
    child = el.find(tag)
    return " ".join(child.text.split()) if child is not None and child.text else ""


def parse_feed(xml_bytes: bytes) -> list[Paper]:
    papers = []
    for entry in ET.fromstring(xml_bytes).findall(ATOM + "entry"):
        m = re.search(r"abs/(.+?)(?:v\d+)?$", _clean(entry, ATOM + "id"))
        if not m:   # arXiv reports a malformed query as a fake <entry>; skip it
            continue
        arxiv_id = m.group(1)
        papers.append(Paper(
            arxiv_id=arxiv_id,
            title=_clean(entry, ATOM + "title"),
            abstract=_clean(entry, ATOM + "summary"),
            authors=[_clean(a, ATOM + "name") for a in entry.findall(ATOM + "author")],
            published=_clean(entry, ATOM + "published")[:10],
            abs_url=f"https://arxiv.org/abs/{arxiv_id}",
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        ))
    return papers


def search_arxiv(question: str) -> list[Paper]:
    """Step 1: ask arXiv for up to MAX_CANDIDATES papers that might be relevant."""
    keywords = extract_keywords(question)
    if not keywords:
        return []
    _wait_turn()
    r = requests.get(API_URL, headers=HEADERS, timeout=30, params={
        "search_query": build_query(keywords), "start": 0, "max_results": MAX_CANDIDATES,
        "sortBy": "relevance", "sortOrder": "descending",
    })
    r.raise_for_status()
    return parse_feed(r.content)


def rank_papers(question: str, papers: list[Paper], embeddings) -> list[tuple[Paper, float]]:
    """Step 2: order candidates by meaning, using the same embedding model as the vector DB.
    Vectors are normalized, so the dot product is the cosine similarity."""
    if not papers:
        return []
    q = np.asarray(embeddings.embed_query(question))
    docs = np.asarray(embeddings.embed_documents([f"{p.title}. {p.abstract}" for p in papers]))
    scores = (docs @ q).tolist()
    return sorted(zip(papers, scores), key=lambda pair: -pair[1])


def download_pdf(paper: Paper) -> Path:
    """Step 3: save the PDF to data/arxiv/<id>.pdf (skipped if we already have it)."""
    path = PDF_DIR / f"{paper.arxiv_id.replace('/', '_')}.pdf"
    if path.exists():
        return path
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    _wait_turn()
    r = requests.get(paper.pdf_url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path

def find_and_store(query: str) -> Paper:
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

    return paper
