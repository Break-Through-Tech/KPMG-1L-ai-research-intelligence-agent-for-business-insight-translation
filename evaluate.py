"""End-to-end evaluator for the research pipeline: retrieval, grounding, citations, abstention.

Usage:
  python evaluate.py                                   # corpus mode, evaluation/threshold_queries.json
  python evaluate.py --mode live                       # live arXiv pipeline (live_research.run), last 180 days
  python evaluate.py --no-judge --limit 10             # quick, cheap smoke run
  python evaluate.py --category business --category near_miss
  python evaluate.py --report-log                      # summarize real questions asked via live_research.py

Modes
  corpus  Retrieve from the fixed PDFs in data/ (ingested into Chroma on first run). Reproducible:
          the gold labels in threshold_queries.json refer to these papers. Use this to compare
          prompt/retrieval changes against each other.
  live    The real product flow: Gemini query rewrite -> arXiv search -> re-rank -> download ->
          chunk retrieval -> summary. Results drift as arXiv changes, and gold papers are only
          findable if they fall inside --days, so treat it as a smoke test, not a benchmark.

Eval set: a list of labeled queries. Both formats are accepted:
  {"id": 1, "category": "agents", "relevant": true,
   "question": "...", "relevant_ids": ["2509.12345", "2509.23456"]}
  {"id": 1, "category": "direct", "relevant": true,
   "query": "...", "expected_paper": "2608.20316v1_Pandora's AI Model Routing Box ..."}
Gold ids may be bare arXiv ids, versioned ids, or corpus file stems; all are normalized to the
bare id ("2608.20316").

Two kinds of cases, two sets of gates:
  relevant=true   -> must retrieve the right paper, ground the answer, cite correctly
  relevant=false  -> must ABSTAIN (no answer built on irrelevant context, no invented sources)

Outputs eval_results.json (per question) and exits non-zero if the suite thresholds aren't met.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from summarizer import _generate

ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_SET = ROOT / "evaluation" / "threshold_queries.json"
CORPUS_DIR = ROOT / "data"

# ---- Pass/fail thresholds: run once, look at the baseline, then tune these ----
K_PAPERS = 3                   # papers kept after re-ranking
K_CHUNKS = 5                   # chunks sent to Gemini
MIN_PAPER_HITS = 1             # G1: at least this many relevant papers in the top K_PAPERS
MIN_CHUNK_PRECISION = 0.6      # G2: >= 60% of retrieved chunks come from relevant papers
MIN_POS_PASS_RATE = 0.7        # suite: pass rate on answerable questions
MIN_ABSTAIN_RATE = 0.8         # suite: share of unanswerable questions correctly refused
# Corpus mode abstains when the best chunk score is below this. 0.43 sits in the gap between the
# lowest answerable (0.465) and the off-topic/unrelated queries (<= 0.389) in threshold_queries.json;
# near-misses above it are left to the summarizer's INSUFFICIENT_EVIDENCE check. Re-tune if the corpus changes.
CORPUS_MIN_CHUNK_SCORE = 0.43

# Fallback only: both runners below set out["abstained"] explicitly (see abstained()). This
# phrase list is for evaluating some other pipeline that just returns text.
ABSTAIN_PATTERN = re.compile(
    r"(not enough|insufficient|no relevant|couldn'?t find|could not find|unable to find|"
    r"cannot answer|can'?t answer|do(?:es)? not contain|don'?t contain|no (papers|sources|evidence))",
    re.I)
ARXIV_ID = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])")


def norm(arxiv_id: str) -> str:
    """'2509.12345v2' / '2608.20316v1_Some Title' -> bare id; anything else is returned stripped."""
    m = ARXIV_ID.match(arxiv_id.strip())
    return m.group(1) if m else arxiv_id.strip()


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


# ---------------- pipelines under test ----------------
class CorpusRunner:
    """Same retrieval + summarization as live_research, but over the fixed PDFs in data/."""

    def __init__(self, min_chunk_score: float):
        from live_research import _open_store, ingest_paths
        from rag.rag_pipeline import get_embedding_client

        pdfs = sorted(CORPUS_DIR.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"no PDFs in {CORPUS_DIR}")
        self.store = _open_store(get_embedding_client())
        added = ingest_paths(pdfs, self.store)
        print(f"Corpus: {len(pdfs)} PDFs, newly ingested: {len(added)}")
        self.doc_ids = [p.stem for p in pdfs]
        # "2608.20316v1_Title" -> "Title"
        self.titles = {s: (s.split("_", 1)[1] if "_" in s else s) for s in self.doc_ids}
        self.min_chunk_score = min_chunk_score

    def __call__(self, question: str, top_papers: int, k_chunks: int) -> dict:
        from live_research import REFUSAL, retrieve_and_summarize
        from summarizer import is_no_answer

        out = {"papers": [], "results": [], "blocks": [], "answer": "", "abstained": False}
        # Rank papers by their best chunk over a wide pull (summary still gets only k_chunks).
        wide = self.store.similarity_search_with_relevance_scores(
            query=question, k=max(k_chunks * 4, 20), filter={"document_id": {"$in": self.doc_ids}})
        ranked = list(dict.fromkeys(d.metadata["document_id"] for d, _ in wide))[:top_papers]
        out["papers"] = ranked
        top = max((s for _, s in wide), default=0.0)
        if not wide or top < self.min_chunk_score:
            out.update(results=wide[:k_chunks], abstained=True,
                       answer=f"No relevant papers in the corpus (best chunk {top:.2f}).")
            return out
        results, blocks, summary = retrieve_and_summarize(
            question, self.store, ranked, self.titles, k_chunks)
        if is_no_answer(summary):
            out.update(results=results, blocks=blocks, abstained=True, answer=REFUSAL)
            return out
        sources = "\n".join(f"- {self.titles[d]} (https://arxiv.org/abs/{norm(d)})" for d in ranked)
        out.update(results=results, blocks=blocks, answer=f"{summary}\n\nSources:\n{sources}")
        return out


class LiveRunner:
    def __init__(self, days: int):
        self.days = days

    def __call__(self, question: str, top_papers: int, k_chunks: int) -> dict:
        from live_research import run

        out = run(question, days=self.days, top_papers=top_papers, k_chunks=k_chunks)
        out["papers"] = [p.arxiv_id for p in out["papers"]]
        return out


# ---------------- retrieval metrics ----------------
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(r in relevant for r in retrieved[:k]) / k if k else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    return next((1 / (i + 1) for i, r in enumerate(retrieved) if r in relevant), 0.0)


# ---------------- answer checks ----------------
def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0) if m else text)


def judge(question: str, blocks: list[str], answer: str) -> dict:
    body = answer.split("Sources:")[0].strip()  # the appended source list isn't a claim
    prompt = (
        "You are a strict evaluator of a research assistant that explains AI research to "
        "business readers. Given a question, source excerpts, and an answer, decide:\n"
        '- "faithful": every factual claim in the answer is supported by the excerpts\n'
        '- "relevant": the answer actually addresses the question\n'
        '- "business_useful": 1-5, how actionable and clearly explained the business '
        "implications are for a non-technical executive (1 = generic or jargon-heavy, "
        "5 = specific, actionable, plain language)\n"
        'Return ONLY JSON: {"faithful": true/false, "relevant": true/false, '
        '"business_useful": 1-5, "reason": "<one sentence>"}\n\n'
        f"Question: {question}\n\nExcerpts:\n" + "\n---\n".join(blocks) + f"\n\nAnswer:\n{body}"
    )
    try:
        v = _parse_json(_generate(prompt))
        v["business_useful"] = int(v.get("business_useful", 0))
        return v
    except Exception as e:
        return {"faithful": False, "relevant": False, "business_useful": 0,
                "reason": f"judge failed: {e}"}


def format_ok(answer: str) -> bool:
    body = answer.split("Sources:")[0]
    bullets = [l for l in body.splitlines() if l.strip().startswith(("-", "*", "•"))]
    return len(bullets) >= 3 and "Sources:" in answer


def cited_ids_not_retrieved(answer: str, allowed: set[str]) -> list[str]:
    """arXiv-style ids (versioned or not) anywhere in the answer that weren't retrieved."""
    return sorted({m.group(1) for m in ARXIV_ID.finditer(answer)} - allowed)


def abstained(out: dict) -> bool:
    """Trust the pipeline's explicit flag. Only fall back to phrase-matching for a pipeline that
    doesn't report one: answers now carry caveats like "the excerpts do not contain X", which the
    pattern would misread as a refusal."""
    if not out.get("blocks"):
        return True
    if "abstained" in out:
        return bool(out["abstained"])
    return bool(ABSTAIN_PATTERN.search(out.get("answer", "").split("Sources:")[0]))


# ---------------- per-question evaluation ----------------
def evaluate_one(item: dict, runner, use_judge: bool) -> dict:
    t0 = time.time()
    out = runner(item["question"], top_papers=K_PAPERS, k_chunks=K_CHUNKS)
    latency = round(time.time() - t0, 1)

    paper_ids = [norm(p) for p in out["papers"]]
    chunk_ids = [norm(d.metadata["document_id"]) for d, _ in out["results"]]
    top_score = max((s for _, s in out["results"]), default=0.0)
    bad_ids = cited_ids_not_retrieved(out["answer"], set(paper_ids) | set(chunk_ids))
    did_abstain = abstained(out)

    result = {"id": item["id"], "category": item["category"], "relevant": item["relevant"],
              "question": item["question"], "top_score": round(float(top_score), 3),
              "latency_s": latency, "retrieved_papers": paper_ids,
              "gold_papers": sorted(item["relevant_ids"]),
              "hallucinated_ids": bad_ids, "abstained": did_abstain, "answer": out["answer"]}

    if not item["relevant"]:
        # Unanswerable: success = refuse cleanly, without inventing sources.
        gates = {"N1_abstained": did_abstain, "N2_no_hallucinated_sources": not bad_ids}
        result.update(metrics={}, gates=gates, passed=all(gates.values()), judge={})
        return result

    rel = item["relevant_ids"]
    metrics = {
        "paper_hit@k": float(any(i in rel for i in paper_ids[:K_PAPERS])),
        "paper_mrr": reciprocal_rank(paper_ids, rel),
        "paper_precision@k": precision_at_k(paper_ids, rel, K_PAPERS),
        "paper_recall@k": recall_at_k(paper_ids, rel, K_PAPERS),
        "chunk_precision@k": precision_at_k(chunk_ids, rel, K_CHUNKS),
        "paper_coverage_in_chunks": recall_at_k(chunk_ids, rel, K_CHUNKS),
    }
    if use_judge and out["blocks"] and not did_abstain:
        verdict = judge(item["question"], out["blocks"], out["answer"])
        metrics["business_useful"] = verdict["business_useful"]
    elif use_judge:
        verdict = {"faithful": False, "relevant": False, "reason": "abstained or no chunks"}
    else:
        verdict = {"faithful": True, "relevant": True, "reason": "judge skipped"}

    gates = {
        "G0_did_not_abstain": not did_abstain,   # refusing an answerable question is a miss
        "G1_relevant_paper_found": sum(i in rel for i in paper_ids[:K_PAPERS]) >= MIN_PAPER_HITS,
        "G2_chunk_precision": metrics["chunk_precision@k"] >= MIN_CHUNK_PRECISION,
        "G3_no_hallucinated_sources": not bad_ids,
        "G4_faithful": bool(verdict.get("faithful")),
        "G5_relevant_answer": bool(verdict.get("relevant")),
        "G6_format": format_ok(out["answer"]),
    }
    result.update(metrics=metrics, gates=gates, passed=all(gates.values()), judge=verdict)
    return result


def error_result(item: dict, err: Exception) -> dict:
    """A crash (network, quota, ...) counts as a failure but doesn't stop the suite."""
    return {"id": item["id"], "category": item["category"], "relevant": item["relevant"],
            "question": item["question"], "top_score": 0.0, "latency_s": 0.0,
            "retrieved_papers": [], "gold_papers": sorted(item["relevant_ids"]),
            "hallucinated_ids": [], "abstained": False, "answer": "",
            "metrics": {}, "gates": {"E_no_error": False}, "passed": False, "judge": {},
            "error": f"{type(err).__name__}: {err}"}


def warn_outside_window(items: list[dict], days: int) -> None:
    """arXiv ids start with YYMM; gold papers older than the search window can't be found."""
    now = datetime.now(timezone.utc)
    old = set()
    for it in items:
        for i in it["relevant_ids"]:
            if re.match(r"\d{4}\.", i):
                yy, mm = int(i[:2]), int(i[2:4])
                if (now.year - 2000 - yy) * 12 + (now.month - mm) > days / 30:
                    old.add(i)
    if old:
        print(f"WARNING: {len(old)} gold paper(s) are older than --days {days} and can't be "
              f"retrieved in live mode (e.g. {sorted(old)[0]}). Increase --days.")


# ---------------- reporting ----------------
def category_report(results: list[dict]) -> None:
    """Pass rate and top-chunk-score spread per category and answerability.
    Compare answerable vs. unanswerable score ranges: if they overlap, no retrieval
    threshold can separate them, and abstention has to rely on something else."""
    print(f"\n{'category':<14}{'kind':<6}{'n':>4}{'pass':>7}{'score min':>11}{'mean':>7}{'max':>7}")
    for cat in sorted({r["category"] for r in results}):
        for rel in (True, False):
            rs = [r for r in results if r["category"] == cat and r["relevant"] == rel]
            if not rs:
                continue
            scores = [r["top_score"] for r in rs]
            print(f"{cat:<14}{'pos' if rel else 'neg':<6}{len(rs):>4}"
                  f"{sum(r['passed'] for r in rs) / len(rs):>7.0%}"
                  f"{min(scores):>11.3f}{mean(scores):>7.3f}{max(scores):>7.3f}")


def gate_report(results: list[dict]) -> None:
    """Which gates fail most often: tells you what to fix first."""
    counts: dict[str, list[int]] = {}
    for r in results:
        for g, ok in r["gates"].items():
            c = counts.setdefault(g, [0, 0])
            c[0] += ok
            c[1] += 1
    print(f"\n{'gate':<30}{'pass':>10}")
    for g, (ok, n) in sorted(counts.items()):
        print(f"{g:<30}{ok:>5}/{n:<4}")


def summarize_results(results: list[dict]) -> tuple[float, float]:
    pos = [r for r in results if r["relevant"]]
    neg = [r for r in results if not r["relevant"]]
    print("\n===== SUMMARY =====")
    keys = sorted({k for r in pos for k in r["metrics"]})
    for key in keys:
        vals = [r["metrics"][key] for r in pos if key in r["metrics"]]
        print(f"  mean {key:<26} {mean(vals):.2f}  (n={len(vals)})")
    pos_rate = sum(r["passed"] for r in pos) / len(pos) if pos else 1.0
    neg_rate = sum(r["passed"] for r in neg) / len(neg) if neg else 1.0
    print(f"  answerable pass rate:   {pos_rate:.0%} ({sum(r['passed'] for r in pos)}/{len(pos)})"
          f"   target >= {MIN_POS_PASS_RATE:.0%}")
    if neg:
        print(f"  abstention rate:        {neg_rate:.0%} ({sum(r['passed'] for r in neg)}/{len(neg)})"
              f"   target >= {MIN_ABSTAIN_RATE:.0%}")
    else:
        print("  abstention rate:        n/a (no relevant=false cases in the eval set)")
    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"  errors:                 {len(errors)} (see 'error' in the results file)")
    print(f"  mean latency:           {mean(r['latency_s'] for r in results):.1f}s")
    category_report(results)
    gate_report(results)
    return pos_rate, neg_rate


def report_log(path: Path) -> None:
    """Summarize real questions logged by live_research.py. These have no correct-paper label, so
    only label-free checks apply. Entries logged with --no-grade are graded now and saved back."""
    from live_research import grade_record

    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not records:
        sys.exit(f"{path} is empty")
    ungraded = [r for r in records if r.get("grades") is None]
    for r in ungraded:
        print(f"grading: {r['question'][:70]}")
        r["grades"] = grade_record(r)
    if ungraded:
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    answered = [r for r in records if not r["abstained"]]
    refused = [r for r in records if r["abstained"]]
    judged = [r for r in answered if "faithful" in r["grades"]]
    pct = lambda k, rs: f"{sum(bool(r['grades'].get(k)) for r in rs) / len(rs):.0%}" if rs else "n/a"

    print(f"\n===== LIVE QUESTIONS ({path}) =====")
    print(f"  questions logged:       {len(records)}")
    print(f"  answered:               {len(answered)}")
    print(f"  refused:                {len(refused)}")
    for reason in sorted({r['abstain_reason'] for r in refused}, key=str):
        print(f"    - {reason}: {sum(r['abstain_reason'] == reason for r in refused)}")
    if judged:
        print(f"  faithful:               {pct('faithful', judged)}   (claims backed by the papers)")
        print(f"  answers the question:   {pct('relevant', judged)}")
        print(f"  mean business useful:   {mean(r['grades']['business_useful'] for r in judged):.1f} / 5")
        print(f"  correct format:         {pct('format_ok', judged)}")
    bad = [r for r in records if r["grades"].get("hallucinated_ids")]
    print(f"  made-up citations:      {len(bad)} answer(s)")
    print(f"  mean latency:           {mean(r['latency_s'] for r in records):.1f}s")

    flagged = [r for r in judged if not (r["grades"]["faithful"] and r["grades"]["relevant"])] + bad
    if flagged:
        print("\nWorth a human look (judge flagged, or invented a citation):")
        for r in flagged:
            print(f"  - {r['question'][:80]}\n      {r['grades'].get('reason', '')}")
    if refused:
        print("\nRefused (check these were right to refuse):")
        for r in refused:
            top = r["papers"][0]["score"] if r["papers"] else None
            print(f"  - [{r['abstain_reason']}, best paper {top}] {r['question'][:80]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_set", nargs="?", default=str(DEFAULT_EVAL_SET))
    ap.add_argument("--mode", choices=["corpus", "live"], default="corpus")
    ap.add_argument("--days", type=int, default=180, help="live mode: arXiv search window (matches live_research)")
    ap.add_argument("--min-chunk-score", type=float, default=CORPUS_MIN_CHUNK_SCORE,
                    help="corpus mode: abstain below this top chunk score")
    ap.add_argument("--no-judge", action="store_true", help="skip LLM judge (gates G4-G5 pass trivially)")
    ap.add_argument("--category", action="append", help="only run these categories (repeatable)")
    ap.add_argument("--limit", type=int, help="only run the first N items (after filtering)")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--report-log", nargs="?", const=str(ROOT / "logs" / "live_questions.jsonl"),
                    metavar="PATH", help="summarize questions logged by live_research.py, then exit")
    args = ap.parse_args()

    import os
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")):
        sys.exit("No Gemini key: add API_KEY=... (or GEMINI_API_KEY=...) to .env in the project root "
                 "(needed for summaries even with --no-judge).")

    if args.report_log:
        report_log(Path(args.report_log))
        return

    items = load_items(Path(args.eval_set))
    if args.category:
        items = [it for it in items if it["category"] in args.category]
    if args.limit:
        items = items[:args.limit]
    if not items:
        sys.exit("no items to evaluate")

    if args.mode == "live":
        warn_outside_window(items, args.days)
        runner = LiveRunner(args.days)
    else:
        runner = CorpusRunner(args.min_chunk_score)

    results = []
    for item in items:
        print(f"\n=== {item['id']} [{item['category']}{'' if item['relevant'] else ', unanswerable'}]: "
              f"{item['question']}")
        try:
            r = evaluate_one(item, runner, use_judge=not args.no_judge)
        except Exception as e:
            r = error_result(item, e)
        results.append(r)
        failed = [g for g, ok in r["gates"].items() if not ok]
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  top_score={r['top_score']}  "
              f"failed={failed or 'none'}{'  error=' + r['error'][:150] if r.get('error') else ''}")

    pos_rate, neg_rate = summarize_results(results)
    Path(args.out).write_text(json.dumps(
        {"mode": args.mode, "eval_set": args.eval_set, "judge": not args.no_judge,
         "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "results": results}, indent=2), encoding="utf-8")
    print(f"\nWrote {args.out}")
    sys.exit(0 if pos_rate >= MIN_POS_PASS_RATE and neg_rate >= MIN_ABSTAIN_RATE else 1)


if __name__ == "__main__":
    main()
