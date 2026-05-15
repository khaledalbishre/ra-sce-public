# src/eval/trec_eval.py
from __future__ import annotations
from typing import Dict, Any, Optional
import pytrec_eval


def _to_float_qrels(qrels: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    # pytrec_eval expects {qid: {docid: relevance_float}}
    out: Dict[str, Dict[str, float]] = {}
    for qid, docs in qrels.items():
        out[qid] = {str(docid): float(rel) for docid, rel in docs.items()}
    return out


def _ensure_run_sorted(run: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    # pytrec_eval doesn't require sorting, but stable + reproducible ordering is good.
    out: Dict[str, Dict[str, float]] = {}
    for qid, docs in run.items():
        # Keep only numeric scores and stringify docids
        items = [(str(did), float(score)) for did, score in docs.items()]
        items.sort(key=lambda x: x[1], reverse=True)
        out[qid] = {did: score for did, score in items}
    return out


def eval_run_metrics(
    run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, Any]],
    ndcg_k: int = 10,
    map_k: int = 1000,
    mrr_k: int = 10,
    relevance_level: int = 1,
) -> Dict[str, float]:
    """
    Returns macro-averaged metrics consistent with trec_eval:
      - ndcg_cut_{k}
      - map_cut_{k}
      - recip_rank (MRR, but cutoff handled below)
    """

    qrels_f = _to_float_qrels(qrels)
    run_f = _ensure_run_sorted(run)

    # IMPORTANT:
    # - ndcg_cut.10 is graded (uses relevance values)
    # - map_cut.1000 is binary-ish if you threshold qrels; trec_eval style uses rel>0 as relevant.
    # - recip_rank is computed from first relevant hit (rel>0).
    measures = {
        f"ndcg_cut.{ndcg_k}",
        f"map_cut.{map_k}",
        "recip_rank",
    }

    evaluator = pytrec_eval.RelevanceEvaluator(qrels_f, measures)
    per_query = evaluator.evaluate(run_f)

    # If you want strict binary relevance (rel >= relevance_level),
    # you should preprocess qrels before calling this function.
    # Most TREC-DL usage treats rel>0 as relevant for MAP/MRR.

    n = len(per_query) if per_query else 1
    ndcg = sum(v.get(f"ndcg_cut_{ndcg_k}", 0.0) for v in per_query.values()) / n
    mp   = sum(v.get(f"map_cut_{map_k}", 0.0) for v in per_query.values()) / n

    # MRR@k is NOT directly given by pytrec_eval.
    # recip_rank is uncut; to enforce cutoff, compute RR@k manually.
    mrr = mrr_at_k(run_f, qrels_f, k=mrr_k, rel_threshold=relevance_level)

    return {
        f"ndcg@{ndcg_k}": ndcg,
        f"map@{map_k}": mp,
        f"mrr@{mrr_k}": mrr,
    }


def mrr_at_k(
    run: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, float]],
    k: int = 10,
    rel_threshold: int = 1,
) -> float:
    """
    Standard MRR@k: for each query, find rank of first doc with rel>=threshold within top-k.
    Macro-average across queries.
    """
    rr_sum = 0.0
    qcount = 0
    for qid, doc_scores in run.items():
        qcount += 1
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        rr = 0.0
        for rank, (docid, _) in enumerate(ranked, start=1):
            rel = qrels.get(qid, {}).get(docid, 0.0)
            if rel >= rel_threshold:
                rr = 1.0 / rank
                break
        rr_sum += rr
    return rr_sum / (qcount if qcount else 1)
