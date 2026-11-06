"""
evaluation.py - Ranking-quality metrics against human relevance labels.

Labels file: CSV with header `file,relevance` and graded labels 0-3 (0 = not relevant, 1 = weak,
2 = partial fit, 3 = strong fit). Metrics computed for every ranking method:
  NDCG@5   - graded gain (2^rel - 1) with logarithmic position discount
  P@3/P@5  - share of the top-k CVs with relevance >= 2
  Spearman - rank correlation between method scores and labels over all labelled CVs
"""
from __future__ import annotations

import csv
import math


def load_labels(path: str) -> dict:
    """Read `file,relevance` rows into a dict {file name: relevance}."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["file"].strip(): float(row["relevance"]) for row in csv.DictReader(fh)}


def rank_by_score(scores: dict) -> list:
    """Order file names by descending score (ties broken alphabetically for determinism)."""
    return sorted(scores, key=lambda f: (-scores[f], f))


def ndcg_at_k(ranking: list, labels: dict, k: int) -> float:
    """Normalised discounted cumulative gain of the top-k positions."""
    dcg = sum((2 ** labels.get(f, 0) - 1) / math.log2(i + 2) for i, f in enumerate(ranking[:k]))
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def precision_at_k(ranking: list, labels: dict, k: int, threshold: float = 2) -> float:
    """Fraction of the top-k CVs whose relevance label is at least `threshold`."""
    return sum(labels.get(f, 0) >= threshold for f in ranking[:k]) / k


def _average_ranks(values: list) -> list:
    """Ranks starting at 1 (ascending), where tied values share their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(xs: list, ys: list) -> float:
    """Spearman rank correlation, i.e. the Pearson correlation of tie-averaged ranks."""
    rx, ry = _average_ranks(xs), _average_ranks(ys)
    mean_x, mean_y = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    spread = math.sqrt(sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry))
    return cov / spread if spread else 0.0


def evaluate_methods(method_scores: dict, labels: dict):
    """Compute all metrics for every method; returns (metric rows, {method: {file: rank}})."""
    rows, ranks = [], {}
    for method, scores in method_scores.items():
        scored = {f: s for f, s in scores.items() if f in labels}
        gold = {f: labels[f] for f in scored}
        ranking = rank_by_score(scored)
        ranks[method] = {f: i + 1 for i, f in enumerate(ranking)}
        rows.append({"method": method,
                     "NDCG@5": round(ndcg_at_k(ranking, gold, 5), 3),
                     "P@3": round(precision_at_k(ranking, gold, 3), 3),
                     "P@5": round(precision_at_k(ranking, gold, 5), 3),
                     "Spearman": round(spearman([scored[f] for f in scored], [gold[f] for f in scored]), 3)})
    return rows, ranks


def format_metrics(rows: list) -> str:
    """Render the metric rows as a terminal table."""
    lines = [f"{'Method':<44}{'NDCG@5':>8}{'P@3':>7}{'P@5':>7}{'Spearman':>10}", "-" * 76]
    lines += [f"{r['method']:<44}{r['NDCG@5']:>8.3f}{r['P@3']:>7.3f}{r['P@5']:>7.3f}{r['Spearman']:>10.3f}"
              for r in rows]
    return "\n".join(lines)


def save_evaluation(metrics_path: str, ranks_path: str, rows: list, ranks: dict, labels: dict) -> None:
    """Write the metric table and the per-CV rank table (rank of each CV under each method) to CSV."""
    with open(metrics_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    methods = list(ranks)
    with open(ranks_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file", "label"] + methods)
        for f in sorted(labels, key=lambda name: (-labels[name], name)):
            writer.writerow([f, labels[f]] + [ranks[m].get(f, "") for m in methods])
