"""Gemini helpers: rewrite a question into arXiv search queries, and summarize retrieved chunks.

Needs GEMINI_API_KEY in your .env / environment. Model name is configurable via GEMINI_MODEL.
"""
import json
import os

from google import genai

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # check this is a current model name
_client = None


def _gemini():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def _generate(prompt: str) -> str:
    return _gemini().models.generate_content(model=MODEL, contents=prompt).text.strip()


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


def summarize(question: str, context_blocks: list[str]) -> str:
    """Answer the question for a business reader, grounded only in the retrieved text."""
    context = "\n\n---\n\n".join(context_blocks)
    prompt = (
        "You translate AI research into business insight for non-technical readers.\n"
        "Answer the question using ONLY the excerpts below. If they don't contain the answer, say so.\n"
        "Structure: a 2-3 sentence plain-language answer, then 3 bullet points on business "
        "implications, then a one-line caveat about limits of the evidence. Avoid jargon.\n\n"
        f"Question: {question}\n\nExcerpts:\n{context}"
    )
    return _generate(prompt)
