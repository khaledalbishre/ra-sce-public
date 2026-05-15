#!/usr/bin/env python
"""
sensitivity_analysis.py — Sensitivity analysis for RA-SCE risk coefficients.

Addresses Reviewer 2 (comment 1): coefficient justification.

Tests 9 coefficient combinations to demonstrate that RA-SCE results are
robust across a wide range of (w_ent, w_disp) values, not over-fit to
the default (0.25, 0.05).

The risk formula is:
    r(q) = w_gap * g(q) + w_ent * H(q) - w_disp * sigma(q)

We sweep w_ent and w_disp while keeping w_gap=1.0 fixed.

This script can run on CPU (no cross-encoder needed for risk computation),
but uses GPU if available for the CE rerank stage.

Usage:
    python sensitivity_analysis.py \
        --config configs/ra_sce/dl19_bm25.yaml \
        --output artifacts/results/sensitivity/dl19/

Author: Khaled Albishre
Date: 2026-05
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder
import pytrec_eval

sys.path.insert(0, str(Path(__file__).parent / "src"))
from policy.ra_sce_v4 import (
    PolicyCfg,
    AdaptiveBlendCfg,
    apply_policy_for_query,
    compute_thresholds_from_query_signals,
    compute_query_signals,
)

# Reuse helpers from main benchmark
sys.path.insert(0, str(Path(__file__).parent))
from benchmark_ra_sce_bm25 import (
    load_queries,
    load_qrels,
    bm25_retrieve,
    cross_encoder_score,
    build_reranked_run,
    evaluate_run,
    harm_profile,
    decision_to_run,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


# 9 coefficient combinations: (w_ent, w_disp)
COEFFICIENT_SWEEP = [
    # (w_ent, w_disp, label)
    (0.10, 0.05, "low_ent"),
    (0.25, 0.05, "default"),
    (0.50, 0.05, "high_ent"),
    (0.25, 0.01, "low_disp"),
    (0.25, 0.10, "high_disp"),
    (0.10, 0.01, "both_low"),
    (0.50, 0.10, "both_high"),
    (0.0,  0.0,  "gap_only"),
    (1.0,  0.20, "extreme"),
]


def run_sensitivity(cfg: dict) -> dict:
    """Run sensitivity analysis over coefficient combinations."""
    
    log.info("=" * 70)
    log.info(f"Sensitivity Analysis: {cfg['experiment_name']}")
    log.info(f"Sweeping {len(COEFFICIENT_SWEEP)} coefficient combinations")
    log.info("=" * 70)
    
    # Load data once
    queries = load_queries(Path(cfg['queries_file']))
    qrels = load_qrels(Path(cfg['qrels_file']))
    
    # BM25 retrieval (once)
    log.info(f"Loading BM25 index: {cfg['bm25_index']}")
    searcher = LuceneSearcher.from_prebuilt_index(cfg['bm25_index'])
    
    log.info("Running BM25 retrieval (once)...")
    base_scores = bm25_retrieve(searcher, queries, k=cfg.get('k0', 100))
    base_pq, base_agg = evaluate_run(qrels, base_scores)
    log.info(f"Base nDCG@10: {base_agg.get('ndcg_cut_10', 0):.4f}")
    
    # CE scores (once — cache for all coefficient combos)
    log.info("Pre-computing CE scores for all queries (cached for sweep)...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ce_model = CrossEncoder(cfg['cross_encoder'], device=device, max_length=512)
    ce_rerank_k = cfg.get('ce_rerank_k', 50)
    ce_bs = cfg.get('ce_batch_size', 64)
    
    # Warmup
    sample_qid = next(iter(queries.keys()))
    sample_top = sorted(base_scores[sample_qid].keys(),
                       key=lambda d: -base_scores[sample_qid][d])[:5]
    _ = cross_encoder_score(ce_model, searcher, queries[sample_qid], sample_top, ce_bs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Compute CE scores for ALL queries (cache)
    ce_cache = {}
    t0 = time.time()
    for qid in queries:
        top = sorted(base_scores[qid].keys(),
                    key=lambda d: -base_scores[qid][d])[:ce_rerank_k]
        ce_cache[qid] = cross_encoder_score(ce_model, searcher, queries[qid], top, ce_bs)
    log.info(f"CE cache built in {time.time()-t0:.1f}s for {len(queries)} queries")
    
    # ===== Sweep =====
    sweep_results = []
    
    for i, (w_ent, w_disp, label) in enumerate(COEFFICIENT_SWEEP):
        log.info(f"\n[{i+1}/{len(COEFFICIENT_SWEEP)}] Testing ({w_ent}, {w_disp}) — {label}")
        
        # Build policy cfg with these coefficients
        blend_cfg = AdaptiveBlendCfg(
            base_alpha=cfg.get('alpha_base', 0.60),
            alpha_high_conf=cfg.get('alpha_high', 0.75),
            alpha_low_conf=cfg.get('alpha_low', 0.45),
        )
        policy_cfg = PolicyCfg(
            rho=cfg.get('rho', 0.30),
            kg=cfg.get('kg', 50),
            w_gap=cfg.get('w_gap', 1.0),
            w_ent=w_ent,
            w_disp=w_disp,
            confidence_lock=True,
            agreement_gate=True,
            topn_lock=True,
            adaptive_blend=True,
            blend=blend_cfg,
        )
        
        # Pass 1: signals + thresholds
        qid_order = list(queries.keys())
        all_signals = [compute_query_signals(base_scores[qid], policy_cfg) for qid in qid_order]
        thresholds = compute_thresholds_from_query_signals(all_signals, policy_cfg)
        tau_risk = thresholds.tau_risk
        
        # Pass 2: apply policy
        ra_sce_run = {}
        refined_count = 0
        for qid, sig in zip(qid_order, all_signals):
            needs_ce = sig.risk >= tau_risk
            ce_sc = ce_cache[qid] if needs_ce else None
            if needs_ce:
                refined_count += 1
            
            decision = apply_policy_for_query(
                qid=qid,
                base_scores_k1=base_scores[qid],
                ce_scores=ce_sc,
                thresholds=thresholds,
                cfg=policy_cfg,
            )
            ra_sce_run[qid] = decision_to_run(decision)
        
        # Evaluate
        ra_sce_pq, ra_sce_agg = evaluate_run(qrels, ra_sce_run)
        h, n, hr = harm_profile(base_pq, ra_sce_pq)
        
        result = {
            'w_ent': w_ent,
            'w_disp': w_disp,
            'label': label,
            'tau_risk': float(tau_risk),
            'ndcg_10': float(ra_sce_agg.get('ndcg_cut_10', 0)),
            'mrr': float(ra_sce_agg.get('recip_rank', 0)),
            'map': float(ra_sce_agg.get('map', 0)),
            'gating_rate': refined_count / len(queries),
            'gating_count': refined_count,
            'helped': h,
            'neutral': n,
            'harmed': hr,
            'harmed_pct': hr / len(queries) * 100,
            'delta_ndcg_vs_base': float(ra_sce_agg.get('ndcg_cut_10', 0) - base_agg.get('ndcg_cut_10', 0)),
        }
        sweep_results.append(result)
        
        log.info(f"  τ={tau_risk:.4f}, gating={result['gating_rate']*100:.1f}%, "
                 f"nDCG@10={result['ndcg_10']:.4f}, harmed={result['harmed_pct']:.1f}%")
    
    # Summary statistics
    ndcg_values = [r['ndcg_10'] for r in sweep_results]
    harmed_values = [r['harmed_pct'] for r in sweep_results]
    gating_values = [r['gating_rate']*100 for r in sweep_results]
    
    output = {
        'experiment': cfg['experiment_name'] + '_sensitivity',
        'dataset': cfg['dataset'],
        'n_queries': len(queries),
        'base_ndcg': float(base_agg.get('ndcg_cut_10', 0)),
        'base_mrr': float(base_agg.get('recip_rank', 0)),
        'sweep_results': sweep_results,
        'summary_statistics': {
            'ndcg_10': {
                'min': float(min(ndcg_values)),
                'max': float(max(ndcg_values)),
                'mean': float(np.mean(ndcg_values)),
                'std': float(np.std(ndcg_values)),
                'range': float(max(ndcg_values) - min(ndcg_values)),
            },
            'harmed_pct': {
                'min': float(min(harmed_values)),
                'max': float(max(harmed_values)),
                'mean': float(np.mean(harmed_values)),
            },
            'gating_pct': {
                'min': float(min(gating_values)),
                'max': float(max(gating_values)),
                'mean': float(np.mean(gating_values)),
            },
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
    
    results = run_sensitivity(cfg)
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cfg['experiment_name']}_sensitivity.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    log.info(f"\n✅ Saved: {out_file}")
    
    # Print summary table
    print("\n" + "=" * 80)
    print(f"SENSITIVITY ANALYSIS: {cfg['experiment_name']}")
    print("=" * 80)
    print(f"{'Label':<14} {'w_ent':<7} {'w_disp':<7} {'nDCG@10':<10} {'ΔnDCG':<10} {'Gate%':<8} {'Harmed%':<10}")
    print("-" * 80)
    for r in results['sweep_results']:
        print(f"{r['label']:<14} {r['w_ent']:<7.2f} {r['w_disp']:<7.2f} "
              f"{r['ndcg_10']:<10.4f} {r['delta_ndcg_vs_base']:+<10.4f} "
              f"{r['gating_rate']*100:<8.1f} {r['harmed_pct']:<10.1f}")
    
    s = results['summary_statistics']
    print("-" * 80)
    print(f"nDCG@10 range: [{s['ndcg_10']['min']:.4f}, {s['ndcg_10']['max']:.4f}] "
          f"(spread: {s['ndcg_10']['range']:.4f}, std: {s['ndcg_10']['std']:.4f})")
    print(f"Harmed%: [{s['harmed_pct']['min']:.1f}, {s['harmed_pct']['max']:.1f}]%")
    print(f"Gating%: [{s['gating_pct']['min']:.1f}, {s['gating_pct']['max']:.1f}]%")
    print("=" * 80)


if __name__ == '__main__':
    main()
