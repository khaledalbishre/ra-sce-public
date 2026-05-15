#!/usr/bin/env python
"""
benchmark_ra_sce_bm25.py — Cross-architecture validation for RA-SCE.

Validates that RA-SCE policy is retriever-agnostic by replacing the
Contriever (FAISS dense) baseline with BM25 (Lucene sparse) while keeping
the same RA-SCE policy.

Pipeline:
    BM25 retrieval (k0=100) → RA-SCE policy → optional CE rerank → final ranking

Compatible with: src/policy/ra_sce_v4.py (no modifications to policy code).

Usage:
    python benchmark_ra_sce_bm25.py \
        --config configs/ra_sce/dl19_bm25.yaml \
        --output artifacts/results/bm25_dl19/

Author: Khaled Albishre
Date: 2026-05
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder
import pytrec_eval

# Import RA-SCE policy (no modifications)
sys.path.insert(0, str(Path(__file__).parent / "src"))
from policy.ra_sce_v4 import (
    PolicyCfg,
    AdaptiveBlendCfg,
    PolicyDecision,
    apply_policy_for_query,
    compute_thresholds_from_query_signals,
    compute_query_signals,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)


# ============================================================================
# Data Loading
# ============================================================================

def load_queries(queries_file: Path) -> Dict[str, str]:
    """Load TSV: qid<TAB>query_text (no header)."""
    queries = {}
    with open(queries_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    log.info(f"Loaded {len(queries)} queries")
    return queries


def load_qrels(qrels_file: Path) -> Dict[str, Dict[str, int]]:
    qrels = {}
    with open(qrels_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                qrels.setdefault(qid, {})[docid] = rel
    log.info(f"Loaded qrels for {len(qrels)} queries")
    return qrels


# ============================================================================
# Retrieval & Reranking
# ============================================================================

def bm25_retrieve(searcher, queries, k=100):
    """Returns dict[qid -> dict[docid -> score]]"""
    log.info(f"BM25 retrieval (k={k}) on {len(queries)} queries...")
    results = {}
    t0 = time.time()
    for qid, query in queries.items():
        hits = searcher.search(query, k=k)
        results[qid] = {hit.docid: float(hit.score) for hit in hits}
    elapsed = time.time() - t0
    log.info(f"BM25 done: {elapsed:.2f}s ({elapsed/len(queries)*1000:.1f}ms/q)")
    return results


def fetch_doc_text(searcher, docid):
    doc = searcher.doc(docid)
    if doc is None:
        return ""
    if doc.contents():
        return doc.contents()
    raw = doc.raw()
    if raw:
        try:
            return json.loads(raw).get('contents', '')
        except Exception:
            return raw
    return ""


def cross_encoder_score(ce_model, searcher, query, docids, batch_size=64):
    """Returns dict[docid -> CE score]"""
    if not docids:
        return {}
    pairs, valid = [], []
    for d in docids:
        t = fetch_doc_text(searcher, d)
        if t:
            pairs.append([query, t])
            valid.append(d)
    if not pairs:
        return {}
    scores = ce_model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
    return {d: float(s) for d, s in zip(valid, scores)}


# ============================================================================
# Run construction (CE-All, Random-30%)
# ============================================================================

def build_reranked_run(base_scores, ce_scores):
    """
    Combine CE-reranked top with rest of baseline, preserving order.
    
    Strategy: offset CE scores so that the LOWEST CE score is HIGHER than
    the HIGHEST score in 'rest'. This guarantees:
    - CE-reranked documents always rank ABOVE the un-reranked rest
    - Internal CE ordering is preserved
    - Internal BM25 ordering of rest is preserved
    """
    rest = {d: s for d, s in base_scores.items() if d not in ce_scores}
    if ce_scores and rest:
        max_rest = max(rest.values())     # highest baseline score in rest
        min_ce = min(ce_scores.values())  # lowest CE score
        # Offset so min_ce + offset > max_rest
        offset = max_rest - min_ce + 1.0
        ce_scores = {d: s + offset for d, s in ce_scores.items()}
    return {**ce_scores, **rest}


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_run(qrels, run, metrics=('ndcg_cut.10', 'recip_rank', 'map')):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(metrics))
    per_query = evaluator.evaluate(run)
    if not per_query:
        return {}, {}
    metric_names = list(next(iter(per_query.values())).keys())
    aggregate = {m: float(np.mean([q[m] for q in per_query.values()]))
                 for m in metric_names}
    return per_query, aggregate


def harm_profile(base_pq, method_pq, metric='ndcg_cut_10'):
    h, n, hr = 0, 0, 0
    for qid in base_pq:
        if qid not in method_pq:
            continue
        delta = method_pq[qid].get(metric, 0) - base_pq[qid].get(metric, 0)
        if delta > 1e-6:
            h += 1
        elif delta < -1e-6:
            hr += 1
        else:
            n += 1
    return h, n, hr


def decision_to_run(decision):
    """Convert PolicyDecision.ranked_docids to run format (rank-based scores)."""
    n = len(decision.ranked_docids)
    return {d: float(n - i) for i, d in enumerate(decision.ranked_docids)}


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(cfg):
    log.info("=" * 70)
    log.info(f"Experiment: {cfg['experiment_name']}")
    log.info(f"BM25 + {cfg['cross_encoder']}")
    log.info("=" * 70)
    
    queries = load_queries(Path(cfg['queries_file']))
    qrels = load_qrels(Path(cfg['qrels_file']))
    
    log.info(f"Loading BM25 index: {cfg['bm25_index']}")
    searcher = LuceneSearcher.from_prebuilt_index(cfg['bm25_index'])
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info(f"Loading CE on {device}: {cfg['cross_encoder']}")
    ce_model = CrossEncoder(cfg['cross_encoder'], device=device, max_length=512)
    
    blend_cfg = AdaptiveBlendCfg(
        base_alpha=cfg.get('alpha_base', 0.60),
        alpha_high_conf=cfg.get('alpha_high', 0.75),
        alpha_low_conf=cfg.get('alpha_low', 0.45),
    )
    policy_cfg = PolicyCfg(
        rho=cfg.get('rho', 0.30),
        kg=cfg.get('kg', 50),
        w_gap=cfg.get('w_gap', 1.0),
        w_ent=cfg.get('w_ent', 0.25),
        w_disp=cfg.get('w_disp', 0.05),
        confidence_lock=cfg.get('confidence_lock', True),
        agreement_gate=cfg.get('agreement_gate', True),
        topn_lock=cfg.get('topn_lock', True),
        adaptive_blend=cfg.get('adaptive_blend', True),
        blend=blend_cfg,
    )
    
    ce_rerank_k = cfg.get('ce_rerank_k', 50)
    ce_bs = cfg.get('ce_batch_size', 64)
    
    # ===== Stage 1: BM25 baseline =====
    log.info("\n[1/4] BM25 baseline")
    t0 = time.time()
    base_scores = bm25_retrieve(searcher, queries, k=cfg.get('k0', 100))
    bm25_time = time.time() - t0
    base_pq, base_agg = evaluate_run(qrels, base_scores)
    log.info(f"  Base nDCG@10: {base_agg.get('ndcg_cut_10', 0):.4f}")
    
    # ===== Stage 2: CE-All =====
    log.info("\n[2/4] CE-All (full reranking)")
    
    # Warmup: GPU + model + index caches
    log.info("  Warming up GPU/model caches...")
    sample_qid = next(iter(queries.keys()))
    sample_top = sorted(base_scores[sample_qid].keys(),
                        key=lambda d: -base_scores[sample_qid][d])[:5]
    _ = cross_encoder_score(ce_model, searcher, queries[sample_qid], sample_top, ce_bs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Per-query latency tracking
    per_query_latency_ms = {}
    
    t0 = time.time()
    ce_all_run = {}
    for qid, sc in base_scores.items():
        q_start = time.time()
        top = sorted(sc.keys(), key=lambda d: -sc[d])[:ce_rerank_k]
        ce_sc = cross_encoder_score(ce_model, searcher, queries[qid], top, ce_bs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        per_query_latency_ms[qid] = (time.time() - q_start) * 1000
        ce_all_run[qid] = build_reranked_run(sc, ce_sc)
    ce_all_time = time.time() - t0
    ce_all_pq, ce_all_agg = evaluate_run(qrels, ce_all_run)
    h, n, hr = harm_profile(base_pq, ce_all_pq)
    log.info(f"  CE-All nDCG@10: {ce_all_agg.get('ndcg_cut_10', 0):.4f}")
    log.info(f"  Time: {ce_all_time:.1f}s ({ce_all_time/len(queries):.2f}s/q)")
    
    # Latency statistics (post-warmup)
    lat_values = list(per_query_latency_ms.values())
    lat_mean = float(np.mean(lat_values))
    lat_median = float(np.median(lat_values))
    lat_p95 = float(np.percentile(lat_values, 95))
    log.info(f"  Latency: mean={lat_mean:.1f}ms, median={lat_median:.1f}ms, P95={lat_p95:.1f}ms")
    log.info(f"  H/N/Harmed: {h}/{n}/{hr} ({hr/len(queries)*100:.1f}%)")
    
    # ===== Stage 3: RA-SCE =====
    log.info("\n[3/4] RA-SCE selective refinement")
    t0 = time.time()
    
    # Pass 1: signals & thresholds
    log.info("  Pass 1: computing query signals")
    qid_order = list(queries.keys())
    all_signals = [compute_query_signals(base_scores[qid], policy_cfg) for qid in qid_order]
    thresholds = compute_thresholds_from_query_signals(all_signals, policy_cfg)
    
    # Risk threshold from Thresholds dataclass
    tau_risk = thresholds.tau_risk
    log.info(f"  Risk threshold τ: {tau_risk:.4f}")
    log.info(f"  Confidence lock: ent<={thresholds.ent_lock_max:.4f}, "
             f"mnorm>={thresholds.mnorm_lock_min:.4f}")
    
    # Pass 2: apply policy
    log.info("  Pass 2: applying policy per query")
    ra_sce_run = {}
    refined_qids = []
    decisions = {}
    for qid, sig in zip(qid_order, all_signals):
        # Risk gating: only refine if risk >= tau_risk
        needs_ce = sig.risk >= tau_risk
        if needs_ce:
            top = sorted(base_scores[qid].keys(),
                         key=lambda d: -base_scores[qid][d])[:ce_rerank_k]
            ce_sc = cross_encoder_score(ce_model, searcher, queries[qid], top, ce_bs)
            refined_qids.append(qid)
        else:
            ce_sc = None
        
        decision = apply_policy_for_query(
            qid=qid,
            base_scores_k1=base_scores[qid],
            ce_scores=ce_sc,
            thresholds=thresholds,
            cfg=policy_cfg,
        )
        decisions[qid] = decision
        ra_sce_run[qid] = decision_to_run(decision)
    
    ra_sce_time = time.time() - t0
    ra_sce_pq, ra_sce_agg = evaluate_run(qrels, ra_sce_run)
    h, n, hr = harm_profile(base_pq, ra_sce_pq)
    log.info(f"  RA-SCE nDCG@10: {ra_sce_agg.get('ndcg_cut_10', 0):.4f}")
    log.info(f"  Gating: {len(refined_qids)/len(queries)*100:.1f}% "
             f"({len(refined_qids)}/{len(queries)})")
    log.info(f"  H/N/Harmed: {h}/{n}/{hr} ({hr/len(queries)*100:.1f}%)")
    log.info(f"  Time: {ra_sce_time:.1f}s")
    
    # Action counts (from policy decisions)
    action_counts = {}
    for d in decisions.values():
        action_counts[d.action_taken] = action_counts.get(d.action_taken, 0) + 1
    log.info(f"  Actions: {action_counts}")
    
    # ===== Stage 4: Random-30% =====
    log.info("\n[4/4] Random-30% (5 seeds)")
    t0 = time.time()
    n_refine = max(1, int(len(queries) * cfg.get('rho', 0.30)))
    seeds_results = []
    seeds_harmed = []
    
    for seed in range(5):
        rng = np.random.RandomState(seed)
        rand_qids = set(rng.choice(list(queries.keys()), size=n_refine, replace=False).tolist())
        
        rand_run = {}
        for qid, sc in base_scores.items():
            if qid in rand_qids:
                top = sorted(sc.keys(), key=lambda d: -sc[d])[:ce_rerank_k]
                ce_sc = cross_encoder_score(ce_model, searcher, queries[qid], top, ce_bs)
                rand_run[qid] = build_reranked_run(sc, ce_sc)
            else:
                rand_run[qid] = sc
        
        rand_pq, rand_agg = evaluate_run(qrels, rand_run)
        _, _, hr = harm_profile(base_pq, rand_pq)
        seeds_results.append(rand_agg)
        seeds_harmed.append(hr)
    
    random_time = time.time() - t0
    random_avg = {m: float(np.mean([s[m] for s in seeds_results]))
                  for m in seeds_results[0].keys()}
    random_std = {m: float(np.std([s[m] for s in seeds_results]))
                  for m in seeds_results[0].keys()}
    log.info(f"  Random-30% nDCG@10: {random_avg.get('ndcg_cut_10', 0):.4f} "
             f"± {random_std.get('ndcg_cut_10', 0):.4f}")
    log.info(f"  Mean harmed: {np.mean(seeds_harmed):.1f}")
    log.info(f"  Time: {random_time:.1f}s")
    
    # ===== Compile output =====
    h_ra, n_ra, hr_ra = harm_profile(base_pq, ra_sce_pq)
    h_ce, n_ce, hr_ce = harm_profile(base_pq, ce_all_pq)
    
    output = {
        'experiment': cfg['experiment_name'],
        'retriever': 'BM25 (Lucene)',
        'cross_encoder': cfg['cross_encoder'],
        'dataset': cfg['dataset'],
        'n_queries': len(queries),
        'config': {
            'kg': policy_cfg.kg, 'rho': policy_cfg.rho,
            'w_gap': policy_cfg.w_gap, 'w_ent': policy_cfg.w_ent,
            'w_disp': policy_cfg.w_disp, 'k0': cfg.get('k0', 100),
            'ce_rerank_k': ce_rerank_k,
        },
        'aggregate_metrics': {
            'base': base_agg, 'ce_all': ce_all_agg, 'ra_sce': ra_sce_agg,
            'random_30_mean': random_avg, 'random_30_std': random_std,
        },
        'no_harm_profile': {
            'ce_all': {'helped': h_ce, 'neutral': n_ce, 'harmed': hr_ce,
                       'harmed_pct': hr_ce/len(queries)*100},
            'ra_sce': {'helped': h_ra, 'neutral': n_ra, 'harmed': hr_ra,
                       'harmed_pct': hr_ra/len(queries)*100},
            'random_30_mean_harmed': float(np.mean(seeds_harmed)),
            'random_30_mean_harmed_pct': float(np.mean(seeds_harmed))/len(queries)*100,
        },
        'gating': {
            'rate': len(refined_qids)/len(queries),
            'tau_risk': float(tau_risk),
            'ent_lock_max': float(thresholds.ent_lock_max),
            'mnorm_lock_min': float(thresholds.mnorm_lock_min),
            'refined_qids': refined_qids,
            'action_counts': action_counts,
        },
        'timing_seconds': {
            'bm25_total': bm25_time,
            'ce_all_total': ce_all_time,
            'ce_all_per_query_mean': ce_all_time/len(queries),
            'ra_sce_total': ra_sce_time,
            'random_30_total': random_time,
        },
        'latency_ms': {
            'ce_all_per_query_mean': lat_mean,
            'ce_all_per_query_median': lat_median,
            'ce_all_per_query_p95': lat_p95,
            'ce_all_per_query_min': float(np.min(lat_values)),
            'ce_all_per_query_max': float(np.max(lat_values)),
            'note': 'Measured after GPU warmup with torch.cuda.synchronize()',
        },
        'per_query_metrics': {
            'base': base_pq, 'ce_all': ce_all_pq, 'ra_sce': ra_sce_pq,
        },
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    
    results = run_experiment(cfg)
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cfg['experiment_name']}_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    log.info(f"\n✅ Saved: {out_file}")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {cfg['experiment_name']}")
    print("=" * 70)
    a = results['aggregate_metrics']
    h = results['no_harm_profile']
    print(f"  Base@k0:    nDCG@10 = {a['base'].get('ndcg_cut_10', 0):.4f}")
    print(f"  CE-All:     nDCG@10 = {a['ce_all'].get('ndcg_cut_10', 0):.4f}  "
          f"(harmed: {h['ce_all']['harmed_pct']:.1f}%)")
    print(f"  Random-30%: nDCG@10 = {a['random_30_mean'].get('ndcg_cut_10', 0):.4f} "
          f"± {a['random_30_std'].get('ndcg_cut_10', 0):.4f}")
    print(f"  RA-SCE:     nDCG@10 = {a['ra_sce'].get('ndcg_cut_10', 0):.4f}  "
          f"(harmed: {h['ra_sce']['harmed_pct']:.1f}%)")
    print(f"  Gating:     {results['gating']['rate']*100:.1f}%")
    print("=" * 70)


if __name__ == '__main__':
    main()
