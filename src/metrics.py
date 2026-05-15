"""
metrics.py

Metrics for IR evaluation with run format:
  run[qid] = {docid: score, ...}

qrels format:
  qrels[qid] = {docid: relevance_int, ...}

Notes:
- nDCG uses graded relevance (standard for TREC-DL).
- MRR / MAP are computed using binary relevance:
    relevant if rel >= rel_threshold (default 1)
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import math


Run = Dict[str, Dict[str, float]]
Qrels = Dict[str, Dict[str, int]]


def sorted_docids(run_for_q: Dict[str, float], k: int) -> List[str]:
    """Return top-k docids sorted by score desc, tie-broken by docid for determinism."""
    items = list(run_for_q.items())
    items.sort(key=lambda x: (-x[1], x[0]))
    return [d for d, _ in items[:k]]


def dcg_at_k(rels: List[int], k: int) -> float:
    """Discounted cumulative gain with log2 discount, rels already aligned to ranking."""
    dcg = 0.0
    for i, rel in enumerate(rels[:k], start=1):
        if rel > 0:
            dcg += (2.0 ** rel - 1.0) / math.log2(i + 1.0)
    return dcg


def ndcg_at_k(run_for_q: Dict[str, float], qrels_for_q: Dict[str, int], k: int) -> float:
    """
    Graded nDCG@k.
    If no relevant documents exist in qrels, return 0.0.
    """
    ranked = sorted_docids(run_for_q, k)
    rels = [int(qrels_for_q.get(d, 0)) for d in ranked]
    dcg = dcg_at_k(rels, k)

    # Ideal ranking: sort qrels by relevance desc
    ideal_rels = sorted((int(r) for r in qrels_for_q.values()), reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def mrr_at_k(run_for_q: Dict[str, float], qrels_for_q: Dict[str, int], k: int, rel_threshold: int = 1) -> float:
    """
    Binary MRR@k: reciprocal rank of first relevant doc within top-k, else 0.
    Relevant if rel >= rel_threshold.
    """
    ranked = sorted_docids(run_for_q, k)
    for idx, docid in enumerate(ranked, start=1):
        if int(qrels_for_q.get(docid, 0)) >= rel_threshold:
            return 1.0 / float(idx)
    return 0.0


def ap_at_k_binary(ranked_docids: List[str], qrels_for_q: Dict[str, int], k: int, rel_threshold: int = 1) -> float:
    """
    Binary Average Precision at k.
    AP@k = average of precision@i over ranks i where doc is relevant, within top-k.
    If no relevant in top-k, return 0.
    """
    hits = 0
    sum_prec = 0.0
    for i, docid in enumerate(ranked_docids[:k], start=1):
        if int(qrels_for_q.get(docid, 0)) >= rel_threshold:
            hits += 1
            sum_prec += hits / float(i)
    if hits == 0:
        return 0.0
    return sum_prec / float(hits)


def map_at_k(run_for_q: Dict[str, float], qrels_for_q: Dict[str, int], k: int, rel_threshold: int = 1) -> float:
    ranked = sorted_docids(run_for_q, k)
    return ap_at_k_binary(ranked, qrels_for_q, k, rel_threshold=rel_threshold)


def evaluate_all(
    run: Run,
    qrels: Qrels,
    primary_k: int = 10,
    map_k: int = 1000,
    rel_threshold: int = 1,
) -> Dict[str, float]:
    """
    Compute:
      - nDCG@primary_k (graded)
      - MRR@primary_k  (binary)
      - MAP@map_k      (binary)
    """
    qids = sorted(set(run.keys()) & set(qrels.keys()))
    if not qids:
        return {"nDCG@{}".format(primary_k): 0.0, "MRR@{}".format(primary_k): 0.0, "MAP@{}".format(map_k): 0.0}

    ndcgs, mrrs, maps = [], [], []
    for qid in qids:
        rfq = run[qid]
        qfq = qrels[qid]
        ndcgs.append(ndcg_at_k(rfq, qfq, primary_k))
        mrrs.append(mrr_at_k(rfq, qfq, primary_k, rel_threshold=rel_threshold))
        maps.append(map_at_k(rfq, qfq, map_k, rel_threshold=rel_threshold))

    def mean(xs: List[float]) -> float:
        return sum(xs) / float(len(xs)) if xs else 0.0

    return {
        f"nDCG@{primary_k}": mean(ndcgs),
        f"MRR@{primary_k}": mean(mrrs),
        f"MAP@{map_k}": mean(maps),
    }


def per_query_metrics(
    run: Run,
    qrels: Qrels,
    primary_k: int = 10,
    map_k: int = 1000,
    rel_threshold: int = 1,
) -> Dict[str, Dict[str, float]]:
    """
    Return per-qid metrics for diagnostics / no-harm analysis.
    """
    out: Dict[str, Dict[str, float]] = {}
    qids = sorted(set(run.keys()) & set(qrels.keys()))
    for qid in qids:
        rfq = run[qid]
        qfq = qrels[qid]
        out[qid] = {
            f"nDCG@{primary_k}": ndcg_at_k(rfq, qfq, primary_k),
            f"MRR@{primary_k}": mrr_at_k(rfq, qfq, primary_k, rel_threshold=rel_threshold),
            f"AP@{map_k}": map_at_k(rfq, qfq, map_k, rel_threshold=rel_threshold),  # per-query AP@map_k
        }
    return out
