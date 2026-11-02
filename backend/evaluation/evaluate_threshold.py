import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from rag.rag_pipeline import retrieve_document

QUERIES_PATH = Path(__file__).with_name("threshold_queries.json")
MIN_RECALL = 0.80


@dataclass
class QueryResult:
    id: int
    category: str
    relevant: bool
    expected_paper: str | None
    retrieved_paper: str
    score: float

    @property
    def paper_match(self) -> bool:
        return self.retrieved_paper == self.expected_paper


def collect_results(k: int = 1) -> list[QueryResult]:
    """Run each query once. Scores don't depend on the threshold, so this is the only slow step."""
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    results = []
    for q in queries:
        doc, score = retrieve_document(query=q["query"], k=k)[0]
        results.append(QueryResult(
            id=q["id"], category=q["category"], relevant=q["relevant"],
            expected_paper=q["expected_paper"],
            retrieved_paper=doc.metadata["document_id"], score=score,
        ))
    return results


def retrieval_accuracy(results: list[QueryResult]) -> float:
    """Top-1 accuracy on relevant queries only. Independent of the threshold."""
    relevant = [r for r in results if r.relevant]
    return sum(r.paper_match for r in relevant) / len(relevant)


def score_threshold(results: list[QueryResult], threshold: float) -> dict:
    tp = fn = fp = tn = wrong_paper = 0
    for r in results:
        accepted = r.score >= threshold
        if r.relevant:
            if accepted and r.paper_match:
                tp += 1
            elif accepted:
                wrong_paper += 1  # passed the gate, but with the wrong context
            else:
                fn += 1
        elif accepted:
            fp += 1
        else:
            tn += 1

    accepted_total = tp + fp + wrong_paper
    precision = tp / accepted_total if accepted_total else 0.0
    recall = tp / (tp + fn + wrong_paper)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "wrong_paper": wrong_paper}


def category_report(results: list[QueryResult], threshold: float) -> None:
    """Score distribution and acceptance rate per category, which shows where the threshold breaks."""
    print(f"\n{'category':<12}{'n':>4}{'min':>8}{'mean':>8}{'max':>8}{'accepted':>10}")
    for cat in sorted({r.category for r in results}):
        scores = [r.score for r in results if r.category == cat]
        accepted = sum(s >= threshold for s in scores) / len(scores)
        print(f"{cat:<12}{len(scores):>4}{min(scores):>8.3f}{mean(scores):>8.3f}"
              f"{max(scores):>8.3f}{accepted:>10.0%}")


def sweep(results: list[QueryResult], start=0.0, stop=1.0, step=0.05) -> list[dict]:
    thresholds = [round(start + i * step, 3) for i in range(int(round((stop - start) / step)) + 1)]
    return [score_threshold(results, t) for t in thresholds]


def select_threshold(rows: list[dict], min_recall: float = MIN_RECALL) -> dict | None:
    """Highest precision among thresholds meeting the recall floor; ties go to higher recall."""
    eligible = [m for m in rows if m["recall"] >= min_recall]
    if not eligible:
        return None
    return max(eligible, key=lambda m: (m["precision"], m["recall"]))


if __name__ == "__main__":
    results = collect_results()

    print(f"Top-1 retrieval accuracy (relevant queries): {retrieval_accuracy(results):.0%}")

    rows = sweep(results)
    print(f"\n{'thresh':>7}{'prec':>7}{'rec':>7}{'f1':>7}{'fp':>5}{'fn':>5}{'wrong':>7}")
    for m in rows:
        print(f"{m['threshold']:>7.2f}{m['precision']:>7.2f}{m['recall']:>7.2f}{m['f1']:>7.2f}"
              f"{m['fp']:>5}{m['fn']:>5}{m['wrong_paper']:>7}")

    best = select_threshold(rows)
    if best is None:
        print(f"\nNo threshold reaches recall >= {MIN_RECALL:.0%}")
    else:
        print(f"\nSelected threshold: {best['threshold']} "
              f"(precision {best['precision']:.2f}, recall {best['recall']:.2f}, "
              f"recall floor {MIN_RECALL:.0%})")
        category_report(results, best["threshold"])
