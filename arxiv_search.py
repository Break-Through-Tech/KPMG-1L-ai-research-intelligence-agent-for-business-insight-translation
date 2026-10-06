"""Live arXiv search -> abstract re-rank -> on-demand PDF download (with cache).

Nothing is pre-processed. Per question: a few API calls, top_k PDF downloads.
"""
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests

ARXIV_API = "https://export.arxiv.org/api/query"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
PDF_CACHE = Path(os.getenv("PDF_CACHE_DIR", "pdf_cache"))
HEADERS = {"User-Agent": "arxiv-rag-project/0.1 (personal research tool)"}
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "what", "how", "why", "which", "do", "does", "with", "about", "from", "by",
    "can", "be", "that", "this", "it", "as", "at", "recent", "latest", "papers",
}

_last_call = 0.0


def _throttle(min_gap: float = 3.0) -> None:
    """arXiv asks for at most one request per 3 seconds."""
    global _last_call
    wait = min_gap - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.monotonic()


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str
    pdf_url: str
    published: str
    score: float = 0.0
    pdf_path: Optional[Path] = None


def _keywords(query: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9\-]+", query.lower())
    seen, out = set(), []
    for w in words:
        if w not in STOPWORDS and len(w) > 2 and w not in seen:
            seen.add(w)
            out.append(w)
    return out[:8]


def search_arxiv(query: str, category: str = "cs.AI", days: int = 30,
                 max_results: int = 50) -> list[Paper]:
    """Keyword search restricted to a category and a recent date window."""
    words = _keywords(query)
    if not words:
        return []
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=days)).strftime("%Y%m%d0000")
    end = now.strftime("%Y%m%d2359")
    terms = " OR ".join(f"all:{w}" for w in words)
    search = f"cat:{category} AND ({terms}) AND submittedDate:[{start} TO {end}]"

    _throttle()
    resp = requests.get(
        ARXIV_API,
        params={"search_query": search, "start": 0, "max_results": max_results,
                "sortBy": "relevance", "sortOrder": "descending"},
        headers=HEADERS, timeout=30,
    )
    resp.raise_for_status()

    papers = []
    for entry in ET.fromstring(resp.text).findall("a:entry", ATOM):
        abs_url = entry.findtext("a:id", default="", namespaces=ATOM)
        arxiv_id = abs_url.split("/abs/")[-1]
        papers.append(Paper(
            arxiv_id=arxiv_id,
            title=" ".join(entry.findtext("a:title", "", ATOM).split()),
            abstract=" ".join(entry.findtext("a:summary", "", ATOM).split()),
            pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            published=entry.findtext("a:published", "", ATOM),
        ))
    return papers


_model = None


def default_embed(texts: list[str]) -> np.ndarray:
    """Lazy-loaded MiniLM. Swap for your existing embedding function if you have one."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return np.asarray(_model.encode(texts, normalize_embeddings=True))


def rerank(question: str, papers: list[Paper], top_k: int = 3,
           embed_fn: Callable[[list[str]], np.ndarray] = default_embed) -> list[Paper]:
    """Rank candidates by cosine similarity of question vs. title+abstract."""
    if not papers:
        return []
    vecs = embed_fn([question] + [f"{p.title}. {p.abstract}" for p in papers])
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = vecs[1:] @ vecs[0]
    for p, s in zip(papers, sims):
        p.score = float(s)
    return sorted(papers, key=lambda p: p.score, reverse=True)[:top_k]


def download_pdf(paper: Paper) -> Path:
    """Download one PDF, skipping the network if it's already cached."""
    PDF_CACHE.mkdir(parents=True, exist_ok=True)
    path = PDF_CACHE / f"{paper.arxiv_id.replace('/', '_')}.pdf"
    if path.exists() and path.stat().st_size > 0:
        paper.pdf_path = path
        return path
    _throttle()
    resp = requests.get(paper.pdf_url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    tmp = path.with_suffix(".part")
    tmp.write_bytes(resp.content)
    tmp.rename(path)  # atomic, so a half-written file is never mistaken for a cache hit
    paper.pdf_path = path
    return path


def find_papers(question: str, query_variants: Optional[list[str]] = None,
                category: str = "cs.AI", days: int = 30, top_k: int = 3,
                embed_fn: Callable[[list[str]], np.ndarray] = default_embed) -> list[Paper]:
    """Full flow: search (one or more query phrasings) -> dedupe -> re-rank -> download top_k."""
    candidates: dict[str, Paper] = {}
    for q in (query_variants or [question]):
        for p in search_arxiv(q, category=category, days=days):
            candidates.setdefault(p.arxiv_id, p)
    top = rerank(question, list(candidates.values()), top_k=top_k, embed_fn=embed_fn)
    for p in top:
        download_pdf(p)
    return top
