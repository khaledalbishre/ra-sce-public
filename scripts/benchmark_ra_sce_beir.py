"""
RA-SCE BEIR Benchmark
=====================
Evaluates RA-SCE on any BEIR dataset using the beir library for
corpus/index management and Contriever as the dense retriever.

Usage:
    python benchmark_ra_sce_beir.py --dataset scifact --rho 0.3
    python benchmark_ra_sce_beir.py --dataset nfcorpus --rho 0.3 --max_queries 50
    python benchmark_ra_sce_beir.py --dataset trec-covid --rho 0.3

The beir library handles dataset download, corpus loading, and FAISS
index construction automatically.  After retrieval, the same RA-SCE
policy from benchmark_ra_sce_v4.py is applied (risk gating, confidence
lock, agreement gate, adaptive blending, top-N lock).

Outputs (in --results_dir):
    diagnostics_<dataset>.csv   per-query diagnostics
    ra_sce_beir_summary.csv     aggregate results
    run_*.trec                  TREC-format run files
"""

import os
os.environ["PYTHONUTF8"] = "1"

import argparse
import math
import gc
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# ── BEIR imports ─────────────────────────────────────────────
from beir import util as beir_util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# ── Models ───────────────────────────────────────────────────
from transformers import AutoTokenizer, AutoModel

try:
    from sentence_transformers import CrossEncoder
    HAS_CE = True
except ImportError:
    HAS_CE = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Available BEIR datasets ─────────────────────────────────
BEIR_DATASETS = [
    "scifact", "nfcorpus", "fiqa", "arguana", "scidocs",
    "trec-covid", "webis-touche2020", "quora", "dbpedia-entity",
    "fever", "climate-fever", "hotpotqa", "nq",
    "msmarco",  # uses dev split
]

# Known Contriever-msmarco nDCG@10 baselines (from official paper / BEIR leaderboard)
KNOWN_BASELINES = {
    "trec-covid": 0.596,
    "nfcorpus":   0.328,
    "scifact":    0.677,
    "fiqa":       0.329,
}


# =============================================================
# Config
# =============================================================
@dataclass
class Config:
    # Dataset
    dataset: str = "scifact"
    beir_data_dir: str = "artifacts/beir_datasets"
    results_dir: str = "artifacts/results/beir"
    split: str = ""  # auto-detect: "dev" for msmarco, "test" for others

    # Models
    model_name: str = "facebook/contriever-msmarco"
    ce_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Retrieval depths
    k0: int = 100        # initial retrieval depth
    k1: int = 1000       # deep retrieval depth
    kg: int = 50         # features computed over top-kg

    # RA-SCE policy
    rho: float = 0.30
    use_cross_encoder: bool = True
    ce_rerank_k: int = 50

    # Confidence lock
    confidence_lock: bool = True
    ent_lock_quantile: float = 0.35
    mnorm_lock_quantile: float = 0.75

    # Agreement gate
    agreement_gate: bool = True
    agree_k: int = 20
    agree_min_overlap: int = 6

    # Adaptive blending
    base_alpha: float = 0.60
    alpha_high_conf: float = 0.75
    alpha_low_conf: float = 0.45

    # Top-N lock
    topn_lock: bool = True
    topn_lock_quantile: float = 0.85
    topn: int = 2

    # Eval
    eval_k: int = 10
    max_queries: Optional[int] = None

    # Encoding
    batch_size: int = 64      # for corpus encoding
    query_batch_size: int = 32


# =============================================================
# Feature / signal functions (identical to benchmark_ra_sce_v4)
# =============================================================
def geo_signal_from_dists(dists: np.ndarray) -> np.ndarray:
    diffs = np.diff(dists, append=dists[-1])
    return np.log1p(np.abs(diffs)).astype(np.float32)


def weighted_gap_feature(dists: np.ndarray, kg: int) -> float:
    geo = geo_signal_from_dists(dists[:kg])
    w = 1.0 / np.log2(np.arange(kg) + 2.0)
    return float(np.sum(geo * w) / (np.sum(w) + 1e-9))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)


def entropy_topk(scores: np.ndarray) -> float:
    p = _softmax(scores.astype(np.float32))
    return float(-(p * np.log(p + 1e-12)).sum())


def compute_features(base_scores: np.ndarray, dists: np.ndarray, kg: int) -> dict:
    head = base_scores[:kg].astype(np.float32)
    disp = float(np.std(head))
    gap = weighted_gap_feature(dists, kg)
    ent = entropy_topk(head)
    m12 = float(head[0] - head[1]) if len(head) > 1 else 0.0
    mnorm = float(m12 / (np.std(head) + 1e-6))
    risk = gap + 0.25 * ent - 0.05 * disp
    return {"disp": disp, "gap": gap, "ent": ent, "m12": m12, "mnorm": mnorm, "risk": risk}


def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mu) / (sd + 1e-9)


def overlap_count(a: list, b: list) -> int:
    return len(set(a).intersection(set(b)))


# =============================================================
# nDCG helpers (internal, for per-query diagnostics)
# =============================================================
def dcg_at_k(rels, k):
    s = 0.0
    for i, rel in enumerate(rels[:k]):
        s += (2**rel - 1) / math.log2(i + 2)
    return s


def ndcg_at_k_from_run(qid, run, qrels, k=10):
    if qid not in qrels or qid not in run:
        return 0.0
    ranked = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)
    rels = [qrels[qid].get(d, 0) for d, _ in ranked]
    ideal = sorted(qrels[qid].values(), reverse=True)
    dcg = dcg_at_k(rels, k)
    idcg = dcg_at_k(ideal, k)
    return (dcg / idcg) if idcg > 0 else 0.0


# =============================================================
# Contriever encoder
# =============================================================
class ContrieverEncoder:
    """Encodes queries and documents using Contriever mean-pooling."""

    def __init__(self, model_name: str, device: torch.device):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModel.from_pretrained(
            model_name, use_safetensors=False
        ).to(device).eval()
        self.device = device

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        inputs = self.tok(
            texts, return_tensors="pt", truncation=True,
            padding=True, max_length=512,
        ).to(self.device)
        with torch.no_grad():
            out = self.mdl(**inputs)
            mask = inputs["attention_mask"].unsqueeze(-1)
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    def encode_queries(self, queries: Dict[str, str], batch_size: int = 32) -> Dict[str, np.ndarray]:
        qids = list(queries.keys())
        texts = [queries[qid] for qid in qids]
        all_embs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            all_embs.append(self._encode_batch(batch))
        embs = np.vstack(all_embs)
        return {qid: embs[i] for i, qid in enumerate(qids)}

    def encode_corpus(self, corpus: Dict[str, dict], batch_size: int = 64) -> Tuple[np.ndarray, List[str]]:
        """Encode all corpus documents. Returns (embeddings_matrix, doc_id_list)."""
        doc_ids = list(corpus.keys())
        texts = []
        for did in doc_ids:
            doc = corpus[did]
            title = doc.get("title", "")
            body = doc.get("text", "")
            texts.append(f"{title} {body}".strip() if title else body)

        all_embs = []
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding corpus"):
            batch = texts[start:start + batch_size]
            all_embs.append(self._encode_batch(batch))
        return np.vstack(all_embs), doc_ids


# =============================================================
# FAISS index builder
# =============================================================
def build_faiss_index(embeddings: np.ndarray) -> "faiss.IndexFlatIP":
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# =============================================================
# TREC run writer
# =============================================================
def write_trec_run(run: Dict[str, Dict[str, float]], path: str, name: str, depth: int = 1000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for qid, doc_scores in run.items():
            ranked = sorted(doc_scores.items(), key=lambda x: float(x[1]), reverse=True)[:depth]
            for rank, (docid, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {float(score):.6f} {name}\n")


# =============================================================
# Diagnostics summary
# =============================================================
def summarize_diagnostics(df: pd.DataFrame, name: str = ""):
    print(f"\n[DIAG] DIAGNOSTICS SUMMARY {name}".strip())
    if df.empty:
        print("  No rows.")
        return
    print(f"\n  Net delta (Ours - Base@k0): {df['delta_net'].mean():+.4f}")
    harmed = (df["delta_net"] < -1e-6).mean() * 100
    helped = (df["delta_net"] >  1e-6).mean() * 100
    neutral = 100 - harmed - helped
    print(f"  Helped: {helped:.1f}% | Neutral: {neutral:.1f}% | Harmed: {harmed:.1f}%")


# =============================================================
# Main
# =============================================================
def main(cfg: Config):
    t_start = time.time()

    # ── Resolve split ────────────────────────────────────────
    if not cfg.split:
        cfg.split = "dev" if cfg.dataset == "msmarco" else "test"

    print(f"[START] RA-SCE BEIR | dataset={cfg.dataset} | split={cfg.split} | rho={cfg.rho}")
    print(f"  results_dir={cfg.results_dir}")

    # ── Download + load dataset ──────────────────────────────
    data_path = os.path.join(cfg.beir_data_dir, cfg.dataset)
    if not os.path.exists(data_path):
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{cfg.dataset}.zip"
        print(f"  Downloading {cfg.dataset} from BEIR ...")
        beir_util.download_and_unzip(url, cfg.beir_data_dir)

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=cfg.split)
    print(f"  Corpus: {len(corpus)} docs | Queries: {len(queries)} | Qrels: {len(qrels)}")

    # ── Filter to judged queries only ────────────────────────
    judged_qids = sorted(qrels.keys())
    if cfg.max_queries and cfg.max_queries > 0:
        judged_qids = judged_qids[:cfg.max_queries]
        print(f"  Limited to {len(judged_qids)} queries (--max_queries)")
    queries_filtered = {qid: queries[qid] for qid in judged_qids if qid in queries}
    qrels_filtered = {qid: qrels[qid] for qid in judged_qids if qid in qrels}

    n_queries = len(queries_filtered)
    if n_queries == 0:
        print("[ERROR] No judged queries found.")
        return

    # ── Load encoder ─────────────────────────────────────────
    print(f"  Loading Contriever: {cfg.model_name}")
    encoder = ContrieverEncoder(cfg.model_name, DEVICE)

    # ── Encode corpus + build FAISS index ────────────────────
    corpus_embs, doc_ids = encoder.encode_corpus(corpus, batch_size=cfg.batch_size)
    print(f"  Corpus encoded: {corpus_embs.shape}")
    index = build_faiss_index(corpus_embs)
    print(f"  FAISS index built ({index.ntotal} vectors, dim={corpus_embs.shape[1]})")

    # Free corpus embeddings (index holds a copy)
    del corpus_embs
    gc.collect()

    # ── Load cross-encoder ───────────────────────────────────
    ce_model = None
    if cfg.use_cross_encoder and HAS_CE:
        ce_model = CrossEncoder(cfg.ce_model_name, device=DEVICE)
        print(f"  CE enabled: {cfg.ce_model_name}")
    elif cfg.use_cross_encoder and not HAS_CE:
        print("  [WARN] sentence-transformers not installed; CE disabled")

    # ── Encode queries ───────────────────────────────────────
    query_embs = encoder.encode_queries(queries_filtered, batch_size=cfg.query_batch_size)

    # ── Pass 1: baseline retrieval + feature computation ─────
    cache = []
    risks, ents, mnorms = [], [], []
    run_base_k0: Dict[str, Dict[str, float]] = {}
    run_base_k1: Dict[str, Dict[str, float]] = {}

    for qid in tqdm(judged_qids, desc="Pass1 (retrieval+features)"):
        if qid not in query_embs:
            continue
        qvec = query_embs[qid].reshape(1, -1)

        # k1 retrieval (superset of k0)
        D, I = index.search(qvec, cfg.k1)
        scores_k1 = D[0].astype(np.float32)
        indices_k1 = I[0]

        # Map to doc IDs
        pairs_k1 = []
        for j in range(len(indices_k1)):
            idx = int(indices_k1[j])
            if 0 <= idx < len(doc_ids):
                pairs_k1.append((doc_ids[idx], float(scores_k1[j])))
        pairs_k1.sort(key=lambda x: x[1], reverse=True)

        # Base@k0
        run_base_k0[qid] = {d: s for d, s in pairs_k1[:cfg.k0]}

        # Base@k1
        run_base_k1[qid] = {d: s for d, s in pairs_k1}

        # Features (computed on k0 scores)
        base_scores_k0 = np.array([s for _, s in pairs_k1[:cfg.k0]], dtype=np.float32)
        # For Contriever inner-product, distance = 1 - score
        dists_k0 = np.maximum(1.0 - np.clip(base_scores_k0, -1.0, 1.0), 1e-9).astype(np.float32)
        kg0 = min(cfg.kg, len(base_scores_k0))
        feats = compute_features(base_scores_k0, dists_k0, kg0)

        cache.append({
            "qid": qid,
            "qtext": queries_filtered[qid],
            "pairs1": pairs_k1,
            **feats,
        })
        risks.append(feats["risk"])
        ents.append(feats["ent"])
        mnorms.append(feats["mnorm"])

    if not cache:
        print("[ERROR] No queries retrieved.")
        return

    # ── Compute thresholds ───────────────────────────────────
    risks_arr = np.array(risks, dtype=np.float32)
    ents_arr = np.array(ents, dtype=np.float32)
    mnorms_arr = np.array(mnorms, dtype=np.float32)

    tau = float(np.quantile(risks_arr, 1.0 - cfg.rho))
    ent_tau = float(np.quantile(ents_arr, cfg.ent_lock_quantile))
    mnorm_tau = float(np.quantile(mnorms_arr, cfg.mnorm_lock_quantile))
    topn_tau = float(np.quantile(mnorms_arr, cfg.topn_lock_quantile))
    mnorm_q30 = float(np.quantile(mnorms_arr, 0.30))
    ent_q70 = float(np.quantile(ents_arr, 0.70))

    print(f"\n  Gate tau(risk)={tau:.6f} | gated~{cfg.rho*100:.1f}%")
    print(f"  Confidence lock: ent<={ent_tau:.6f} AND mnorm>={mnorm_tau:.6f}")
    print(f"  Top-{cfg.topn} lock when mnorm>={topn_tau:.6f}")

    # ── Pass 2: RA-SCE policy ────────────────────────────────
    # Build doc_id -> text lookup (only for CE-scored docs)
    doc_text_lookup: Dict[str, str] = {}
    if ce_model is not None:
        needed_ids = set()
        for item in cache:
            if float(item["risk"]) >= tau:
                for d, _ in item["pairs1"][:cfg.ce_rerank_k]:
                    needed_ids.add(d)
        for did in needed_ids:
            if did in corpus:
                doc = corpus[did]
                title = doc.get("title", "")
                body = doc.get("text", "")
                doc_text_lookup[did] = f"{title} {body}".strip() if title else body
        print(f"  Loaded {len(doc_text_lookup)} doc texts for CE scoring")

    run_ours: Dict[str, Dict[str, float]] = {}
    diag_rows = []
    gated_cnt = 0
    action_cnt = 0

    for item in tqdm(cache, desc="Pass2 (RA-SCE policy)"):
        qid = item["qid"]
        qtext = item["qtext"]
        pairs1 = item["pairs1"]

        # Initialize from Base@k1
        run_ours[qid] = dict(run_base_k1[qid])

        risk = float(item["risk"])
        gated = int(risk >= tau)
        if gated:
            gated_cnt += 1

        action_taken = 0
        skip_reason = "none"

        if gated and ce_model is not None:

            # Confidence lock
            if cfg.confidence_lock and (float(item["ent"]) <= ent_tau) and (float(item["mnorm"]) >= mnorm_tau):
                skip_reason = "confidence_lock"
            else:
                cand_docids = [d for d, _ in pairs1]
                topk = min(cfg.ce_rerank_k, len(cand_docids))
                head_docids = cand_docids[:topk]
                tail_docids = cand_docids[topk:]

                base_head_scores = np.array(
                    [run_base_k1[qid][d] for d in head_docids], dtype=np.float32
                )

                # CE scoring
                ce_pairs = [[qtext, doc_text_lookup.get(did, "")] for did in head_docids]
                ce_scores = (
                    ce_model.predict(ce_pairs).astype(np.float32)
                    if ce_pairs else np.zeros((0,), dtype=np.float32)
                )

                if cfg.agreement_gate:
                    ak = min(cfg.agree_k, topk)
                    base_top = head_docids[:ak]
                    ce_rank = sorted(
                        zip(head_docids, ce_scores),
                        key=lambda x: float(x[1]), reverse=True,
                    )
                    ce_top = [d for d, _ in ce_rank[:ak]]
                    ov = overlap_count(base_top, ce_top)

                    if ov < cfg.agree_min_overlap:
                        skip_reason = f"agree_gate<{cfg.agree_min_overlap} (ov={ov})"
                    else:
                        # Adaptive alpha
                        high_conf = (
                            float(item["ent"]) <= ent_tau
                            and float(item["mnorm"]) >= mnorm_tau
                        )
                        low_conf = (
                            float(item["mnorm"]) <= mnorm_q30
                            and float(item["ent"]) >= ent_q70
                        )
                        if high_conf:
                            alpha = cfg.alpha_high_conf
                        elif low_conf:
                            alpha = cfg.alpha_low_conf
                        else:
                            alpha = cfg.base_alpha

                        zb = zscore(base_head_scores)
                        zc = zscore(ce_scores)
                        blended = alpha * zb + (1.0 - alpha) * zc

                        head = list(zip(head_docids, blended, base_head_scores))
                        head.sort(key=lambda x: (float(x[1]), float(x[2])), reverse=True)
                        new_head_ids = [d for d, _, _ in head]

                        # Top-N lock
                        if cfg.topn_lock and float(item["mnorm"]) >= topn_tau:
                            locked = head_docids[:cfg.topn]
                            new_head_ids = locked + [d for d in new_head_ids if d not in set(locked)]
                            skip_reason = f"ce_blend_top{cfg.topn}lock(alpha={alpha:.2f})"
                        else:
                            skip_reason = f"ce_blend(alpha={alpha:.2f})"

                        action_taken = 1
                        action_cnt += 1

                        merged = new_head_ids + tail_docids
                        run_ours[qid] = {did: float(1000.0 - i) for i, did in enumerate(merged)}
                else:
                    # No agreement gate: just blend
                    alpha = cfg.base_alpha
                    zb = zscore(base_head_scores)
                    zc = zscore(ce_scores)
                    blended = alpha * zb + (1.0 - alpha) * zc
                    head = list(zip(head_docids, blended, base_head_scores))
                    head.sort(key=lambda x: (float(x[1]), float(x[2])), reverse=True)
                    new_head_ids = [d for d, _, _ in head]
                    merged = new_head_ids + tail_docids
                    run_ours[qid] = {did: float(1000.0 - i) for i, did in enumerate(merged)}
                    action_taken = 1
                    action_cnt += 1
                    skip_reason = f"ce_blend(alpha={alpha:.2f})"

        # Per-query diagnostics
        nd0 = ndcg_at_k_from_run(qid, run_base_k0, qrels_filtered, k=cfg.eval_k)
        ndo = ndcg_at_k_from_run(qid, run_ours, qrels_filtered, k=cfg.eval_k)

        diag_rows.append({
            "qid": qid,
            "risk": risk,
            "gated": gated,
            "action_taken": action_taken,
            "skip_reason": skip_reason,
            "ent": float(item["ent"]),
            "mnorm": float(item["mnorm"]),
            "ndcg_base_k0": nd0,
            "ndcg_ours": ndo,
            "delta_net": ndo - nd0,
        })

    # ── Pass 3: CE-All baseline ──────────────────────────────
    run_ce_all: Dict[str, Dict[str, float]] = {}
    if ce_model is not None:
        # Pre-load all doc texts needed for CE-All
        all_ce_needed = set()
        for item in cache:
            for d, _ in item["pairs1"][:cfg.ce_rerank_k]:
                all_ce_needed.add(d)
        for did in all_ce_needed:
            if did not in doc_text_lookup and did in corpus:
                doc = corpus[did]
                title = doc.get("title", "")
                body = doc.get("text", "")
                doc_text_lookup[did] = f"{title} {body}".strip() if title else body

        ce_all_alpha = cfg.base_alpha
        for item in tqdm(cache, desc="Pass3 (CE-All)"):
            qid = item["qid"]
            qtext = item["qtext"]

            cand_docids = sorted(
                run_base_k1[qid].keys(),
                key=lambda d: run_base_k1[qid][d], reverse=True,
            )
            topk = min(cfg.ce_rerank_k, len(cand_docids))
            head_docids = cand_docids[:topk]
            tail_docids = cand_docids[topk:]

            base_head_scores = np.array(
                [run_base_k1[qid][d] for d in head_docids], dtype=np.float32
            )

            ce_pairs = [[qtext, doc_text_lookup.get(did, "")] for did in head_docids]
            ce_scores = (
                ce_model.predict(ce_pairs).astype(np.float32)
                if ce_pairs else np.zeros((0,), dtype=np.float32)
            )

            zb = zscore(base_head_scores)
            zc = zscore(ce_scores)
            blended = ce_all_alpha * zb + (1.0 - ce_all_alpha) * zc

            head = list(zip(head_docids, blended, base_head_scores))
            head.sort(key=lambda x: (float(x[1]), float(x[2])), reverse=True)
            new_head_ids = [d for d, _, _ in head]

            merged = new_head_ids + tail_docids
            run_ce_all[qid] = {did: float(1000.0 - i) for i, did in enumerate(merged)}
    else:
        run_ce_all = {qid: dict(scores) for qid, scores in run_base_k1.items()}

    # ── Evaluate with BEIR ───────────────────────────────────
    evaluator = EvaluateRetrieval()
    k_values = [1, 3, 5, 10, 100]

    print(f"\n[RESULTS] {cfg.dataset} (split={cfg.split})")
    print(f"  Gated: {gated_cnt}/{n_queries} ({100*gated_cnt/n_queries:.1f}%) | CE actions: {action_cnt}")

    results = {}
    for method_name, run in [("Base@k0", run_base_k0), ("CE-All", run_ce_all), ("RA-SCE", run_ours)]:
        ndcg, _map, recall, precision = evaluator.evaluate(qrels_filtered, run, k_values)
        results[method_name] = {
            "nDCG@10": ndcg.get("NDCG@10", 0.0),
            "nDCG@100": ndcg.get("NDCG@100", 0.0),
            "Recall@100": recall.get("Recall@100", 0.0),
        }
        print(f"  {method_name:10s}: nDCG@10={ndcg.get('NDCG@10',0):.4f} | "
              f"nDCG@100={ndcg.get('NDCG@100',0):.4f} | "
              f"Recall@100={recall.get('Recall@100',0):.4f}")

    # ── Compare against known Contriever baselines ───────────
    if cfg.dataset in KNOWN_BASELINES:
        known = KNOWN_BASELINES[cfg.dataset]
        our_base = results["Base@k0"]["nDCG@10"]
        delta = our_base - known
        status = "OK" if abs(delta) < 0.03 else "MISMATCH"
        print(f"\n  [VERIFY] Known Contriever nDCG@10 for {cfg.dataset}: {known:.3f}")
        print(f"           Our Base@k0 nDCG@10: {our_base:.4f} (delta={delta:+.4f}) [{status}]")
        if "RA-SCE" in results:
            ra_delta = results["RA-SCE"]["nDCG@10"] - known
            print(f"           RA-SCE nDCG@10:      {results['RA-SCE']['nDCG@10']:.4f} (delta vs known={ra_delta:+.4f})")

    # ── Save outputs ─────────────────────────────────────────
    os.makedirs(cfg.results_dir, exist_ok=True)
    ds_tag = cfg.dataset.replace("-", "_")

    # Diagnostics CSV
    df = pd.DataFrame(diag_rows)
    diag_path = os.path.join(cfg.results_dir, f"diagnostics_{ds_tag}.csv")
    df.to_csv(diag_path, index=False)
    summarize_diagnostics(df, name=f"({cfg.dataset})")
    print(f"\n[SAVE] {diag_path}")

    # TREC run files
    write_trec_run(run_base_k0, os.path.join(cfg.results_dir, f"run_base_k0_{ds_tag}.trec"), "base_k0")
    write_trec_run(run_ce_all,  os.path.join(cfg.results_dir, f"run_ce_all_{ds_tag}.trec"),  "ce_all")
    write_trec_run(run_ours,    os.path.join(cfg.results_dir, f"run_ra_sce_{ds_tag}.trec"),  "ra_sce")
    print(f"[SAVE] TREC run files -> {cfg.results_dir}")

    # Summary CSV
    summary_rows = []
    for method_name, metrics in results.items():
        summary_rows.append({
            "Dataset": cfg.dataset,
            "Split": cfg.split,
            "Method": method_name,
            "Queries": n_queries,
            "GatedPct": 100.0 * gated_cnt / n_queries if method_name == "RA-SCE" else
                        (100.0 if method_name == "CE-All" else 0.0),
            "CE_Actions": action_cnt if method_name == "RA-SCE" else
                          (n_queries if method_name == "CE-All" else 0),
            **metrics,
        })
    summary_path = os.path.join(cfg.results_dir, "ra_sce_beir_summary.csv")

    # Append if file exists (accumulate across datasets)
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        combined = pd.concat([existing, pd.DataFrame(summary_rows)], ignore_index=True)
        combined.to_csv(summary_path, index=False)
    else:
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[SAVE] {summary_path}")

    elapsed = time.time() - t_start
    print(f"\n[DONE] {cfg.dataset} in {elapsed/60:.1f} min ({elapsed/n_queries:.1f} s/query)")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================
# CLI
# =============================================================
def parse_args():
    ap = argparse.ArgumentParser("RA-SCE BEIR Benchmark")

    ap.add_argument("--dataset", type=str, default="scifact",
                    choices=BEIR_DATASETS,
                    help="BEIR dataset name")
    ap.add_argument("--beir_data_dir", type=str, default="artifacts/beir_datasets")
    ap.add_argument("--results_dir", type=str, default="artifacts/results/beir")
    ap.add_argument("--split", type=str, default="",
                    help="Dataset split (auto: 'dev' for msmarco, 'test' for others)")

    ap.add_argument("--model_name", type=str, default=None)
    ap.add_argument("--ce_model_name", type=str, default=None)

    ap.add_argument("--k0", type=int, default=None)
    ap.add_argument("--k1", type=int, default=None)
    ap.add_argument("--kg", type=int, default=None)
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--ce_rerank_k", type=int, default=None)

    ap.add_argument("--agreement_gate", type=str, default=None)
    ap.add_argument("--confidence_lock", type=str, default=None)
    ap.add_argument("--topn_lock", type=str, default=None)

    ap.add_argument("--base_alpha", type=float, default=None)
    ap.add_argument("--alpha_high_conf", type=float, default=None)
    ap.add_argument("--alpha_low_conf", type=float, default=None)

    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_queries", type=int, default=None)

    return ap.parse_args()


def _str2bool(v):
    if v is None:
        return None
    return str(v).lower() in ("1", "true", "yes", "y")


if __name__ == "__main__":
    args = parse_args()
    cfg = Config()

    # Apply CLI overrides
    for k, v in vars(args).items():
        if v is None:
            continue
        if k in ("agreement_gate", "confidence_lock", "topn_lock"):
            setattr(cfg, k, _str2bool(v))
        elif hasattr(cfg, k):
            setattr(cfg, k, v)

    main(cfg)
