"""Evaluator for the live research pipeline (end-to-end + abstention).

Usage:  python evaluate.py [eval_set.json] [--no-judge]

Eval set: a list of labeled queries. Accepts both the original format and the
threshold-tuning format, so one file can serve both scripts:
[
  {"id": 1, "category": "agents", "relevant": true,
   "question": "How are companies using LLM agents for customer service?",
   "relevant_ids": ["2509.12345", "2509.23456"]},
  {"id": 2, "category": "offtopic", "relevant": false,
   "question": "What's a good recipe for banana bread?", "relevant_ids": []}
]
Also accepted: "query" instead of "question", "expected_paper" instead of "relevant_ids".

Two kinds of cases, two sets of gates:
  relevant=true   -> must retrieve the right paper, ground the answer, cite correctly
  relevant=false  -> must ABSTAIN (no answer built on irrelevant context, no invented sources)
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean

from live_research import run
from summarizer import _generate

# ---- Pass/fail thresholds: run once, look at the baseline, then tune these ----
K_PAPERS = 3                   # papers kept after re-ranking
K_CHUNKS = 5                   # chunks sent to Gemini
MIN_PAPER_HITS = 1             # G1: at least this many relevant papers in the top K_PAPERS
MIN_CHUNK_PRECISION = 0.6      # G2: >= 60% of retrieved chunks come from relevant papers
MIN_POS_PASS_RATE = 0.7        # suite: pass rate on answerable questions
MIN_ABSTAIN_RATE = 0.8         # suite: share of unanswerable questions correctly refused

# If run() exposes an explicit flag (out["abstained"]), that wins. Otherwise we infer
# abstention from "no chunks retrieved" or a refusal phrase in the answer. Adjust to
# whatever wording your pipeline uses when it declines.
ABSTAIN_PATTERN = re.compile(
    r"(not enough|insufficient|no relevant|couldn'?t find|could not find|unable to find|"
    r"cannot answer|can'?t answer|no (papers|sources|evidence))", re.I)


def norm(arxiv_id: str) -> str:
    """'2509.12345v2' -> '2509.12345' so gold labels needn't include the version."""
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def load_items(path: Path) -> list[dict]:
    items = []
    for i, q in enumerate(json.loads(path.read_text(encoding="utf-8"))):
        relevant = q.get("relevant", True)
        ids = q.get("relevant_ids") or ([q["expected_paper"]] if q.get("expected_paper") else [])
        if relevant and not ids:
            raise ValueError(f"item {q.get('id', i)} is relevant but has no relevant_ids/expected_paper")
        items.append({
            "id": str(q.get("id", i)),
            "question": q.get("question") or q["query"],
            "category": q.get("category", "general"),
            "relevant": relevant,
            "relevant_ids": {norm(x) for x in ids},
        })
    return items


# ---------------- retrieval metrics ----------------
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(r in relevant for r in retrieved[:k]) / k if k else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


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


def cited_ids_not_retrieved(answer: str, allowed: set[str]) -> list[str]:
    """arXiv-style ids anywhere in the answer (body or Sources) that weren't retrieved."""
    return sorted(i for i in set(re.findall(r"\b\d{4}\.\d{4,5}\b", answer)) if i not in allowed)


def abstained(out: dict) -> bool:
    if "abstained" in out:
        return bool(out["abstained"])
    return not out.get("blocks") or bool(ABSTAIN_PATTERN.search(out.get("answer", "")))


# ---------------- per-question evaluation ----------------
def evaluate_one(item: dict, use_judge: bool) -> dict:
    t0 = time.time()
    out = run(item["question"], top_papers=K_PAPERS, k_chunks=K_CHUNKS)
    latency = round(time.time() - t0, 1)

    paper_ids = [norm(p.arxiv_id) for p in out["papers"]]
    chunk_ids = [norm(d.metadata["document_id"]) for d, _ in out["results"]]
    top_score = max((s for _, s in out["results"]), default=0.0)  # same score scale as the threshold script
    bad_ids = cited_ids_not_retrieved(out["answer"], set(paper_ids))
    did_abstain = abstained(out)

    result = {"id": item["id"], "category": item["category"], "relevant": item["relevant"],
              "question": item["question"], "top_score": round(float(top_score), 3),
              "latency_s": latency, "retrieved_papers": paper_ids,
              "hallucinated_ids": bad_ids, "abstained": did_abstain, "answer": out["answer"]}

    if not item["relevant"]:
        # Unanswerable: success = refuse cleanly, without inventing sources.
        gates = {"N1_abstained": did_abstain, "N2_no_hallucinated_sources": not bad_ids}
        result.update(metrics={}, gates=gates, passed=all(gates.values()), judge_reason="")
        return result

    rel = item["relevant_ids"]
    metrics = {
        "paper_precision@k": precision_at_k(paper_ids, rel, K_PAPERS),
        "paper_recall@k": recall_at_k(paper_ids, rel, K_PAPERS),
        "chunk_precision@k": precision_at_k(chunk_ids, rel, K_CHUNKS),
        "paper_coverage_in_chunks": recall_at_k(chunk_ids, rel, K_CHUNKS),
    }
    if use_judge and out["blocks"] and not did_abstain:
        verdict = judge(item["question"], out["blocks"], out["answer"])
    else:
        verdict = {"faithful": not use_judge, "relevant": not use_judge,
                   "reason": "abstained or no chunks" if use_judge else "judge skipped"}

    gates = {
        "G0_did_not_abstain": not did_abstain,   # refusing an answerable question is a miss
        "G1_relevant_paper_found": sum(i in rel for i in paper_ids) >= MIN_PAPER_HITS,
        "G2_chunk_precision": metrics["chunk_precision@k"] >= MIN_CHUNK_PRECISION,
        "G3_no_hallucinated_sources": not bad_ids,
        "G4_faithful": bool(verdict.get("faithful")),
        "G5_relevant_answer": bool(verdict.get("relevant")),
        "G6_format": format_ok(out["answer"]),
    }
    result.update(metrics=metrics, gates=gates, passed=all(gates.values()),
                  judge_reason=verdict.get("reason", ""))
    return result


# ---------------- reporting ----------------
def category_report(results: list[dict]) -> None:
    """Pass rate and top-chunk-score spread per category and answerability.
    Compare answerable vs. unanswerable score ranges: if they overlap, no retrieval
    threshold can separate them, and abstention has to rely on something else."""
    print(f"\n{'category':<14}{'kind':<7}{'n':>3}{'pass':>7}{'score min':>11}{'mean':>7}{'max':>7}")
    for cat in sorted({r["category"] for r in results}):
        for rel in (True, False):
            rs = [r for r in results if r["category"] == cat and r["relevant"] == rel]
            if not rs:
                continue
            scores = [r["top_score"] for r in rs]
            print(f"{cat:<14}{'pos' if rel else 'neg':<7}{len(rs):>3}"
                  f"{sum(r['passed'] for r in rs) / len(rs):>7.0%}"
                  f"{min(scores):>11.3f}{mean(scores):>7.3f}{max(scores):>7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_set", nargs="?", default="eval_set.json")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM judge (gates 4-5 pass trivially)")
    args = ap.parse_args()

    items = load_items(Path(args.eval_set))
    results = []
    for item in items:
        print(f"\n=== {item['id']} [{item['category']}{'' if item['relevant'] else ', unanswerable'}]: {item['question']}")
        r = evaluate_one(item, use_judge=not args.no_judge)
        results.append(r)
        failed = [g for g, ok in r["gates"].items() if not ok]
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  top_score={r['top_score']}  failed={failed or 'none'}")

    pos = [r for r in results if r["relevant"]]
    neg = [r for r in results if not r["relevant"]]
    print("\n===== SUMMARY =====")
    if pos:
        for key in pos[0]["metrics"]:
            print(f"  mean {key}: {mean(r['metrics'][key] for r in pos):.2f}")
    pos_rate = sum(r["passed"] for r in pos) / len(pos) if pos else 1.0
    neg_rate = sum(r["passed"] for r in neg) / len(neg) if neg else 1.0
    print(f"  answerable pass rate:   {pos_rate:.0%} ({sum(r['passed'] for r in pos)}/{len(pos)})")
    if neg:
        print(f"  abstention rate:        {neg_rate:.0%} ({sum(r['passed'] for r in neg)}/{len(neg)})")
    else:
        print("  abstention rate:        n/a (no relevant=false cases in the eval set)")
    category_report(results)

    Path("eval_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ok = pos_rate >= MIN_POS_PASS_RATE and neg_rate >= MIN_ABSTAIN_RATE
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()