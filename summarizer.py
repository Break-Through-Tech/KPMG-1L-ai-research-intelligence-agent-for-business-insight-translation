"""Gemini helpers: rewrite a question into arXiv search queries, and summarize retrieved chunks.

Needs GEMINI_API_KEY (or API_KEY) in .env / environment. Model name is configurable via GEMINI_MODEL.
"""
import json
import os
import re
import time

from google import genai
from google.genai import errors

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")  # same model as summarize/summary.py on main
_client = None


def _gemini():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.environ["API_KEY"])  # API_KEY = name used in summarize/summary.py on main
    return _client


def _generate(prompt: str, retries: int = 5) -> str:
    """Call Gemini, waiting out rate limits (429) and overload (503). The free tier allows
    ~15 requests/minute, so a full eval run will hit this; the error says how long to wait."""
    for attempt in range(retries + 1):
        try:
            return _gemini().models.generate_content(model=MODEL, contents=prompt).text.strip()
        except errors.APIError as e:
            if e.code not in (429, 500, 503) or attempt == retries:
                raise
            m = re.search(r"retry in ([\d.]+)s", str(e))
            wait = float(m.group(1)) + 1 if m else min(60, 5 * 2 ** attempt)
            print(f"  Gemini {e.code}, retrying in {wait:.0f}s ({attempt + 1}/{retries})")
            time.sleep(wait)


def rewrite_queries(question: str, n: int = 3) -> list[str]:
    """Turn a business question into n short keyword queries for arXiv. Falls back to the question."""
    prompt = (
        f"Turn this question into {n} different short keyword queries (3-6 words each) "
        "that would find relevant AI research papers on arXiv. Use technical vocabulary "
        "that authors would use, and vary the phrasing. "
        'Return ONLY a JSON list of strings, no markdown.\n\n'
        f"Question: {question}"
    )
    try:
        text = _generate(prompt).replace("```json", "").replace("```", "").strip()
        queries = [q for q in json.loads(text) if isinstance(q, str) and q.strip()]
        return queries[:n] or [question]
    except Exception:
        return [question]


# Returned (alone) when the excerpts don't answer the question, so callers can abstain reliably
# instead of pattern-matching free-text refusals.
NO_ANSWER = "INSUFFICIENT_EVIDENCE"


def is_no_answer(summary: str) -> bool:
    return summary.strip().upper().startswith(NO_ANSWER)


def summarize(question: str, context_blocks: list[str]) -> str:
    """Answer the question for a business reader, grounded only in the retrieved text."""
    context = "\n\n---\n\n".join(context_blocks)
    prompt = (
        "You translate AI research into business insight for non-technical readers.\n"
        "Answer the question using ONLY the excerpts below.\n"
        "If the excerpts address the question even partially, answer with what they support "
        "and note what's missing in the caveat; plain-language or business framing of a "
        "technical topic still counts as addressing it.\n"
        f"Only if the excerpts are about a different subject than the question (sharing a few "
        f"keywords is not enough), reply with exactly {NO_ANSWER} and nothing else.\n"
        "Structure: a 2-3 sentence plain-language answer, then 3 bullet points on business "
        "implications, then a one-line caveat about limits of the evidence. Avoid jargon.\n\n"
        f"Question: {question}\n\nExcerpts:\n{context}"
    )
    return _generate(prompt)
