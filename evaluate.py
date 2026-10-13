"""Evaluator for the live research pipeline.

Usage:  python evaluate.py [eval_set.json]

eval_set.json format (you hand-label relevant_ids by reading abstracts):
[
  {"id": "q1",
   "question": "How are companies using LLM agents for customer service?",
   "relevant_ids": ["2509.12345", "2509.23456", "2510.00123"]}
]
"""
import json
import re
import sys
import time
from pathlib import Path

from live_research import run
from summarizer import _generate

# ---- Pass/fail thresholds: run once, look at the baseline, then tune these ----
K_PAPERS = 3                  # papers kept after re-ranking
K_CHUNKS = 5                  # chunks sent to Gemini
MIN_PAPER_HITS = 1            # gate 1: at least this many relevant papers in the top K_PAPERS
MIN_CHUNK_PRECISION = 0.6     # gate 2: >= 60% of retrieved chunks come from relevant papers
REQUIRE_JUDGE = True          # gates 4-5: LLM judge must say faithful AND relevant
MIN_SUITE_PASS_RATE = 0.7     # whole-suite verdict (exit code)


def norm(arxiv_id: str) -> str:
    """'2509.12345v2' -> '2509.12345' so gold labels needn't include the version."""
    return re.sub(r"v\d+$", "", arxiv_id.strip())


# ---------------- retrieval metrics ----------------
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return sum(r in relevant for r in top) / k if k else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


# ---------------- answer checks ----------------
def judge(question: str, blocks: list[str], answer: str) -> dict:
    prompt = (
        "You are a strict evaluator. Given a question, source excerpts, and an answer, decide:\n"
        '- "faithful": every factual claim in the answer is supported by the excerpts\n'
        '- "relevant": the answer actually addresses the question\n'
        'Return ONLY JSON: {"faithful": true/false, "relevant": true/false, "reason": "<one sentence>"}\n\n'
        f"Question: {question}\n\nExcerpts:\n" + "\n---\n".join(blocks) + f"\n\nAnswer:\n{answer}"
    )
    try:
        text = _generate(prompt).replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        return {"faithful": False, "relevant": False, "reason": f"judge failed: {e}"}


def format_ok(answer: str) -> bool:
    body = answer.split("Sources:")[0]
    bullets = [l for l in body.splitlines() if l.strip().startswith(("-", "*", "•"))]
    return len(bullets) >= 3 and "Sources:" in answer


def hallucinated_ids(answer: str, allowed: set[str]) -> list[str]:
    body = answer.split("Sources:")[0]
    return [i for i in set(re.findall(r"\b\d{4}\.\d{4,5}\b", body)) if i not in allowed]


# ---------------- per-question evaluation ----------------
def evaluate_one(item: dict) -> dict:
    relevant = {norm(i) for i in item["relevant_ids"]}
    t0 = time.time()
    out = run(item["question"], top_papers=K_PAPERS, k_chunks=K_CHUNKS)
    latency = time.time() - t0

    paper_ids = [norm(p.arxiv_id) for p in out["papers"]]
    chunk_ids = [norm(d.metadata["document_id"]) for d, _ in out["results"]]

    m = {
        "paper_precision@k": precision_at_k(paper_ids, relevant, K_PAPERS),
        "paper_recall@k": recall_at_k(paper_ids, relevant, K_PAPERS),
        "chunk_precision@k": precision_at_k(chunk_ids, relevant, K_CHUNKS),
        "paper_coverage_in_chunks": recall_at_k(chunk_ids, relevant, K_CHUNKS),
    }
    verdict = judge(item["question"], out["blocks"], out["answer"]) if out["blocks"] else \
        {"faithful": False, "relevant": False, "reason": "no chunks retrieved"}
    bad_ids = hallucinated_ids(out["answer"], set(paper_ids))

    gates = {
        "G1_relevant_paper_found": sum(i in relevant for i in paper_ids) >= MIN_PAPER_HITS,
        "G2_chunk_precision": m["chunk_precision@k"] >= MIN_CHUNK_PRECISION,
        "G3_no_hallucinated_sources": not bad_ids,
        "G4_faithful": verdict["faithful"] or not REQUIRE_JUDGE,
        "G5_relevant_answer": verdict["relevant"] or not REQUIRE_JUDGE,
        "G6_format": format_ok(out["answer"]),
    }
    return {"id": item["id"], "question": item["question"], "metrics": m, "gates": gates,
            "passed": all(gates.values()), "judge_reason": verdict.get("reason", ""),
            "hallucinated_ids": bad_ids, "latency_s": round(latency, 1),
            "retrieved_papers": paper_ids, "answer": out["answer"]}


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "eval_set.json")
    items = json.loads(path.read_text())
    results = []
    for item in items:
        print(f"\n=== {item['id']}: {item['question']}")
        r = evaluate_one(item)
        results.append(r)
        failed = [g for g, ok in r["gates"].items() if not ok]
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['metrics']}  failed={failed or 'none'}")

    n = len(results)
    avg = lambda key: sum(r["metrics"][key] for r in results) / n
    pass_rate = sum(r["passed"] for r in results) / n
    print("\n===== SUMMARY =====")
    for key in results[0]["metrics"]:
        print(f"  mean {key}: {avg(key):.2f}")
    print(f"  pass rate: {pass_rate:.0%}  ({sum(r['passed'] for r in results)}/{n})")
    Path("eval_results.json").write_text(json.dumps(results, indent=2))
    sys.exit(0 if pass_rate >= MIN_SUITE_PASS_RATE else 1)


if __name__ == "__main__":
    main()
