import os
os.environ['PYTHONUTF8'] = '1'

"""
RA-SCE v4 - Agreement-gated + Adaptive Blend + Top-N Lock
=========================================================
Fixes catastrophic harms on already-good queries by:
1) Agreement gate: require CE top-k to overlap base top-k sufficiently.
2) Adaptive alpha: trust base more when confidence is high.
3) Top-N lock: lock top-N for high-confidence queries.

Still:
- risk budget gating (rho quantile per split)
- confidence lock
- CE head rerank within k1 pool

Outputs:
- ra_sce_v4_summary.csv
- diagnostics_<dataset>.csv
- run_*.trec files (for trec_eval / pytrec_eval evaluation)
"""

import os
import math
import gc
import argparse
import json
try:
    import yaml
except Exception:
    yaml = None
from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import faiss
import ir_datasets
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

try:
    from sentence_transformers import CrossEncoder
    HAS_CE = True
except Exception:
    HAS_CE = False


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    index_path: str = "./artifacts/indexes/msmarco_contriever.index"
    id_path: str = "./artifacts/indexes/msmarco_ids_fixed.npy"
    results_dir: str = "artifacts/results"

    model_name: str = "facebook/contriever-msmarco"
    ce_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    k0: int = 100
    k1: int = 1000
    kg: int = 50

    rho: float = 0.30
    use_cross_encoder: bool = True
    ce_rerank_k: int = 50

    # Confidence lock (skip CE when already confident)
    confidence_lock: bool = True
    ent_lock_quantile: float = 0.35   # more protective than v3
    mnorm_lock_quantile: float = 0.75 # more protective than v3

    # Agreement gate (skip CE if CE disagrees strongly with base)
    agreement_gate: bool = True
    agree_k: int = 20                 # compare top-k sets (default: top-20)
    agree_min_overlap: int = 6        # require at least this many overlaps in top-k

    # Adaptive blending
    base_alpha: float = 0.60          # default alpha
    alpha_high_conf: float = 0.75     # if confident, trust base more
    alpha_low_conf: float = 0.45      # if very uncertain, trust CE more

    # Top-N lock when confidence high
    topn_lock: bool = True
    topn_lock_quantile: float = 0.85
    topn: int = 2

    eval_k: int = 10
    map_k: int = 1000
    mrr_k: int = 10

    # Optional override via CLI to run only one dataset key
    dataset_key: Optional[str] = None
    max_queries: Optional[int] = None

    datasets: List[Tuple[str, str]] = None
    def __post_init__(self):
        if self.datasets is None:
            self.datasets = [
                ("TREC_DL_19", "msmarco-passage/trec-dl-2019/judged"),
                ("TREC_DL_20", "msmarco-passage/trec-dl-2020/judged"),
                ("MSMARCO_Dev", "msmarco-passage/dev/small"),
            ]


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Local data paths (bypass ir_datasets for queries/qrels on Windows)
LOCAL_DATA = {
    "TREC_DL_19": {
        "queries": "artifacts/data/dl19_queries.tsv",
        "qrels": "artifacts/data/qrels_dl19.txt",
    },
    "TREC_DL_20": {
        "queries": "artifacts/data/dl20_queries.tsv",
        "qrels": "artifacts/data/qrels_dl20.txt",
    },
}

Query = namedtuple("Query", ["query_id", "text"])


def load_local_queries(path: str) -> List:
    """Load TSV queries file: qid<tab>text"""
    queries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                queries.append(Query(query_id=parts[0], text=parts[1]))
    return queries


def load_local_qrels(path: str) -> Dict[str, Dict[str, int]]:
    """Load TREC qrels file: qid 0 docid rel"""
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[0], parts[1], parts[2], int(parts[3])
                qrels.setdefault(qid, {})[docid] = rel
    return qrels


# -----------------------------
# CLI override helpers
# -----------------------------
def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("RA-SCE v4 benchmark (with CLI overrides)")

    ap.add_argument("--override_yaml", type=str, default=None, help="Path to YAML/JSON file with Config field overrides")

    # Paths / models
    ap.add_argument("--index_path", type=str, default=None)
    ap.add_argument("--id_path", type=str, default=None)
    ap.add_argument("--results_dir", type=str, default=None)
    ap.add_argument("--model_name", type=str, default=None)
    ap.add_argument("--ce_model_name", type=str, default=None)

    # Retrieval depths
    ap.add_argument("--k0", type=int, default=None)
    ap.add_argument("--k1", type=int, default=None)
    ap.add_argument("--kg", type=int, default=None)

    # Policy knobs
    ap.add_argument("--rho", type=float, default=None)
    ap.add_argument("--use_cross_encoder", type=str2bool, default=None)
    ap.add_argument("--ce_rerank_k", type=int, default=None)

    # Agreement gate
    ap.add_argument("--agreement_gate", type=str2bool, default=None)
    ap.add_argument("--agree_k", type=int, default=None)
    ap.add_argument("--agree_min_overlap", type=int, default=None)

    # Confidence lock
    ap.add_argument("--confidence_lock", type=str2bool, default=None)
    ap.add_argument("--ent_lock_quantile", type=float, default=None)
    ap.add_argument("--mnorm_lock_quantile", type=float, default=None)

    # Blending
    ap.add_argument("--base_alpha", type=float, default=None)
    ap.add_argument("--alpha_high_conf", type=float, default=None)
    ap.add_argument("--alpha_low_conf", type=float, default=None)

    # Top-N lock
    ap.add_argument("--topn_lock", type=str2bool, default=None)
    ap.add_argument("--topn_lock_quantile", type=float, default=None)
    ap.add_argument("--topn", type=int, default=None)

    # Metrics
    ap.add_argument("--eval_k", type=int, default=None)
    ap.add_argument("--map_k", type=int, default=None)
    ap.add_argument("--mrr_k", type=int, default=None)

    # Optional: run only one dataset key (TREC_DL_19, TREC_DL_20, MSMARCO_Dev)
    ap.add_argument("--dataset_key", type=str, default=None)

    # Limit number of queries (useful for quick tests)
    ap.add_argument("--max_queries", type=int, default=None,
                    help="Limit to first N judged queries per dataset (for testing)")

    return ap.parse_args()


def apply_dict_overrides(cfg, d: dict):
    """Apply key/value overrides to cfg if the attribute exists."""
    for k, v in (d or {}).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def load_override_file(path: str) -> dict:
    """Load YAML/JSON override file."""
    if path is None:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    low = path.lower()
    if low.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if low.endswith((".yml", ".yaml")):
        if yaml is None:
            raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    raise ValueError("override file must be .yaml/.yml or .json")


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    for k, v in vars(args).items():
        if v is None:
            continue
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


# -----------------------------
# IO
# -----------------------------
def write_trec_run(run: Dict[str, Dict[str, float]], out_path: str, run_name: str, depth: int = 1000) -> None:
    """Write run dict to standard TREC run format: qid Q0 docid rank score runname."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for qid, doc_scores in run.items():
            ranked = sorted(doc_scores.items(), key=lambda x: float(x[1]), reverse=True)[:depth]
            for rank, (docid, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {float(score):.6f} {run_name}\n")


# -----------------------------
# Metrics (your existing internal eval)
# Note: You are now validating with pytrec_eval externally via tools/eval_pytrec.py
# -----------------------------
def dcg_at_k(rels, k):
    s = 0.0
    for i, rel in enumerate(rels[:k]):
        gain = (2**rel - 1)
        s += gain / math.log2(i + 2)
    return s

def ndcg_at_k(rels, k, ideal_rels):
    dcg = dcg_at_k(rels, k)
    idcg = dcg_at_k(ideal_rels, k)
    return (dcg / idcg) if idcg > 0 else 0.0

def ndcg_at_k_from_run(qid, run, qrels, k=10):
    if qid not in qrels or qid not in run:
        return 0.0
    ranked = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)
    rels = [qrels[qid].get(d, 0) for d, _ in ranked]
    ideal = sorted(qrels[qid].values(), reverse=True)
    return ndcg_at_k(rels, k, ideal)

def _binary_rels_for_qid(qid: str, run: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]):
    if qid not in qrels or qid not in run:
        return []
    ranked = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)
    return [1 if qrels[qid].get(d, 0) > 0 else 0 for d, _ in ranked]

def mrr_at_k_from_run(qid: str, run: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]], k: int = 10) -> float:
    rels = _binary_rels_for_qid(qid, run, qrels)[:k]
    for i, r in enumerate(rels, start=1):
        if r:
            return 1.0 / float(i)
    return 0.0

def ap_at_k_from_run(qid: str, run: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]], k: int = 1000) -> float:
    rels = _binary_rels_for_qid(qid, run, qrels)[:k]
    if not rels:
        return 0.0
    num_rel = sum(rels)
    if num_rel == 0:
        return 0.0
    hits = 0
    s = 0.0
    for i, r in enumerate(rels, start=1):
        if r:
            hits += 1
            s += hits / float(i)
    return s / float(num_rel)

def eval_run_metrics(run: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]], ndcg_k: int, map_k: int, mrr_k: int) -> Dict[str, float]:
    ndcg_vals = []
    ap_vals = []
    mrr_vals = []
    for qid in run.keys():
        if qid not in qrels:
            continue
        ranked = sorted(run[qid].items(), key=lambda x: x[1], reverse=True)
        rels = [qrels[qid].get(d, 0) for d, _ in ranked]
        ideal = sorted(qrels[qid].values(), reverse=True)
        ndcg_vals.append(ndcg_at_k(rels, ndcg_k, ideal))
        ap_vals.append(ap_at_k_from_run(qid, run, qrels, k=map_k))
        mrr_vals.append(mrr_at_k_from_run(qid, run, qrels, k=mrr_k))
    return {
        f"nDCG@{ndcg_k}": float(np.mean(ndcg_vals)) if ndcg_vals else 0.0,
        f"MAP@{map_k}": float(np.mean(ap_vals)) if ap_vals else 0.0,
        f"MRR@{mrr_k}": float(np.mean(mrr_vals)) if mrr_vals else 0.0,
    }


# -----------------------------
# Features / signals
# -----------------------------
def geo_signal_from_dists(dists: np.ndarray) -> np.ndarray:
    diffs = np.diff(dists, append=dists[-1])
    return np.log1p(np.abs(diffs)).astype(np.float32)

def weighted_gap_feature(dists: np.ndarray, kg: int) -> float:
    geo = geo_signal_from_dists(dists[:kg])
    w = 1.0 / np.log2(np.arange(kg) + 2.0)
    return float(np.sum(geo * w) / (np.sum(w) + 1e-9))

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (np.sum(e) + 1e-12)

def entropy_topk(scores: np.ndarray) -> float:
    p = softmax(scores.astype(np.float32))
    return float(-(p * np.log(p + 1e-12)).sum())

def compute_features(base_scores: np.ndarray, dists: np.ndarray, kg: int) -> Dict[str, float]:
    head = base_scores[:kg].astype(np.float32)
    disp = float(np.std(head))
    gap = weighted_gap_feature(dists, kg)
    ent = entropy_topk(head)
    m12 = float(head[0] - head[1]) if len(head) > 1 else 0.0
    mnorm = float(m12 / (np.std(head) + 1e-6))
    risk = gap + 0.25 * ent - 0.05 * disp
    return {"disp": disp, "gap": gap, "ent": ent, "m12": m12, "mnorm": mnorm, "risk": risk}


# -----------------------------
# Models
# -----------------------------
def load_contriever(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name, use_safetensors=False).to(DEVICE).eval()
    return tok, mdl

def embed_query(tok, mdl, text: str) -> np.ndarray:
    inputs = tok(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        out = mdl(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1)
        qvec = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
        qvec = F.normalize(qvec, p=2, dim=1)
    return qvec.detach().cpu().numpy().astype(np.float32)

def scores_and_dists(D0: np.ndarray, is_l2: bool):
    if is_l2:
        dists = D0.astype(np.float32)
        base_scores = (-dists).astype(np.float32)
        return base_scores, dists
    raw = D0.astype(np.float32)
    clip = np.clip(raw, -1.0, 1.0).astype(np.float32)
    dists = np.maximum(1.0 - clip, 1e-9).astype(np.float32)
    return raw, dists

def zscore(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x))
    if sd < 1e-9:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mu) / (sd + 1e-9)

def overlap_count(a: List[str], b: List[str]) -> int:
    return len(set(a).intersection(set(b)))


# -----------------------------
# Diagnostics
# -----------------------------
def summarize_diagnostics(df: pd.DataFrame, name: str = ""):
    print(f"\n[DIAG] DIAGNOSTICS SUMMARY {name}".strip())
    if df.empty:
        print("No rows.")
        return
    print("\nOverall deltas (mean):")
    print(f"  Net (Ours - Base@k0): {df['delta_net'].mean():+.4f}")
    harmed = (df["delta_net"] < -1e-6).mean() * 100
    helped = (df["delta_net"] >  1e-6).mean() * 100
    neutral = 100 - harmed - helped
    print("\nNo-harm profile (% queries):")
    print(f"  Helped:  {helped:.1f}% | Neutral: {neutral:.1f}% | Harmed: {harmed:.1f}%")
    print("\nWorst 8 queries (net):")
    print(df.sort_values("delta_net").head(8)[["qid", "delta_net", "ndcg_base_k0", "risk", "gated", "action_taken", "skip_reason"]].to_string(index=False))


# -----------------------------
# Main
# -----------------------------
def main(cfg: Config):
    os.makedirs(cfg.results_dir, exist_ok=True)
    print(f"[START] RA-SCE v4 | rho={cfg.rho} | CE@{cfg.ce_rerank_k} | base_alpha={cfg.base_alpha}")
    print(f"   results_dir={cfg.results_dir}")
    print(f"   agreement_gate={cfg.agreement_gate} (k={cfg.agree_k}, min_overlap={cfg.agree_min_overlap}) | "
          f"confidence_lock={cfg.confidence_lock} (ent_q={cfg.ent_lock_quantile}, mnorm_q={cfg.mnorm_lock_quantile}) | "
          f"topn_lock={cfg.topn_lock} (topn={cfg.topn}, q={cfg.topn_lock_quantile})")

    index = faiss.read_index(cfg.index_path)
    is_l2 = (index.metric_type == faiss.METRIC_L2)
    print(f"   FAISS metric: {'L2' if is_l2 else 'InnerProduct'}")

    id_path = cfg.id_path
    if not os.path.exists(id_path):
        # Fallback: try without "_fixed" suffix
        alt = id_path.replace("msmarco_ids_fixed.npy", "msmarco_ids.npy")
        if os.path.exists(alt):
            print(f"   ID file not found at {id_path}, using fallback: {alt}")
            id_path = alt
    fixed_ids = np.load(id_path)
    fixed_ids_list = [str(x) for x in fixed_ids]

    tok, mdl = load_contriever(cfg.model_name)

    if cfg.use_cross_encoder:
        if not HAS_CE:
            raise RuntimeError("sentence-transformers not available but use_cross_encoder=True.")
        ce_model = CrossEncoder(cfg.ce_model_name, device=DEVICE)
        print(f"   CE enabled: {cfg.ce_model_name}")
    else:
        ce_model = None
        print("   CE disabled")

    # Load doc texts from local collection.tsv (replaces ir_datasets docstore)
    doc_texts: Dict[str, str] = {}
    collection_path = None
    for candidate in [
        os.path.expanduser("~/.ir_datasets/msmarco-passage/collection.tsv"),
        os.path.expanduser("~/.cache/ir_datasets/msmarco-passage/collection.tsv"),
        "./artifacts/data/collection.tsv",
    ]:
        if os.path.exists(candidate):
            collection_path = candidate
            break

    if ce_model is not None and not collection_path:
        print("   [WARN] No collection.tsv found; CE scoring will use empty texts")

    summary_rows = []

    # Optional dataset filter
    datasets = cfg.datasets
    if cfg.dataset_key is not None:
        datasets = [d for d in datasets if d[0] == cfg.dataset_key]
        if not datasets:
            raise ValueError(f"--dataset_key={cfg.dataset_key} not found in cfg.datasets")

    for ds_key, ds_path in datasets:
        print(f"\n[DS] Processing {ds_key} ...")

        # Load queries and qrels from local files (avoids ir_datasets Windows bug)
        if ds_key in LOCAL_DATA and os.path.exists(LOCAL_DATA[ds_key]["queries"]):
            print(f"   Loading queries/qrels from local files")
            queries = load_local_queries(LOCAL_DATA[ds_key]["queries"])
            qrels = load_local_qrels(LOCAL_DATA[ds_key]["qrels"])
        else:
            dataset = ir_datasets.load(ds_path)
            qrels: Dict[str, Dict[str, int]] = {}
            for qr in dataset.qrels_iter():
                qrels.setdefault(str(qr.query_id), {})[str(qr.doc_id)] = int(qr.relevance)
            queries = list(dataset.queries_iter())

        # Pre-filter: only keep judged queries (saves embedding + search for unjudged ones)
        queries = [q for q in queries if str(q.query_id) in qrels]
        if cfg.max_queries is not None and cfg.max_queries > 0:
            queries = queries[:cfg.max_queries]
            print(f"   {len(queries)} queries (limited by --max_queries={cfg.max_queries})")
        else:
            print(f"   {len(queries)} judged queries (filtered from full query set)")

        cache = []
        risks, ents, mnorms = [], [], []

        for q in tqdm(queries, desc=f"{ds_key} pass1(k0)"):
            qid = str(q.query_id)
            if qid not in qrels:
                continue

            qvec = embed_query(tok, mdl, q.text)
            D0, I0 = index.search(qvec, cfg.k0)
            base_scores0, dists0 = scores_and_dists(D0[0], is_l2)

            kg0 = min(cfg.kg, cfg.k0)
            feats = compute_features(base_scores0, dists0, kg0)

            cache.append({
                "qid": qid,
                "qtext": q.text,
                "qvec": qvec,
                "I0": I0[0].copy(),
                "base_scores0": base_scores0.copy(),
                **feats,
            })
            risks.append(feats["risk"])
            ents.append(feats["ent"])
            mnorms.append(feats["mnorm"])

        if len(cache) == 0:
            print(f"[WARN] No evaluable queries for {ds_key} (no qrels overlap). Skipping.")
            continue

        risks_arr = np.array(risks, dtype=np.float32)
        ents_arr = np.array(ents, dtype=np.float32)
        mnorms_arr = np.array(mnorms, dtype=np.float32)

        tau = float(np.quantile(risks_arr, 1.0 - cfg.rho))
        ent_tau = float(np.quantile(ents_arr, cfg.ent_lock_quantile))
        mnorm_tau = float(np.quantile(mnorms_arr, cfg.mnorm_lock_quantile))
        topn_tau = float(np.quantile(mnorms_arr, cfg.topn_lock_quantile))

        # dataset-level confidence quantiles for adaptive alpha (computed once)
        mnorm_q30 = float(np.quantile(mnorms_arr, 0.30))
        ent_q70 = float(np.quantile(ents_arr, 0.70))

        print(f"   Gate tau(risk)={tau:.6f} | gated~{cfg.rho*100:.1f}% by design")
        print(f"   Confidence lock: ent<= {ent_tau:.6f} AND mnorm>= {mnorm_tau:.6f}")
        print(f"   Top-{cfg.topn} lock when mnorm >= {topn_tau:.6f}")
        if cfg.agreement_gate:
            print(f"   Agreement gate: overlap(top{cfg.agree_k}) >= {cfg.agree_min_overlap}")

        # Pre-retrieve k1 for all queries and collect needed doc IDs for CE
        run_base_k0: Dict[str, Dict[str, float]] = {}
        run_base_k1: Dict[str, Dict[str, float]] = {}
        run_ours: Dict[str, Dict[str, float]] = {}
        needed_ids: set = set()

        for item in tqdm(cache, desc=f"{ds_key} k1-retrieval"):
            qid = item["qid"]
            qvec = item["qvec"]
            I0 = item["I0"]
            s0 = item["base_scores0"]

            # Base@k0
            docids0 = [fixed_ids_list[idx] if idx < len(fixed_ids_list) else None for idx in I0]
            pairs0 = [(docids0[i], float(s0[i])) for i in range(len(docids0)) if docids0[i] is not None]
            pairs0.sort(key=lambda x: x[1], reverse=True)
            run_base_k0[qid] = {d: sc for d, sc in pairs0}

            # Base@k1
            D1, I1 = index.search(qvec, cfg.k1)
            s1, _d1 = scores_and_dists(D1[0], is_l2)
            docids1 = [fixed_ids_list[idx] if idx < len(fixed_ids_list) else None for idx in I1[0]]
            pairs1 = [(docids1[i], float(s1[i])) for i in range(len(docids1)) if docids1[i] is not None]
            pairs1.sort(key=lambda x: x[1], reverse=True)
            run_base_k1[qid] = {d: sc for d, sc in pairs1}

            # Collect top ce_rerank_k doc IDs for CE scoring
            for d, _ in pairs1[:cfg.ce_rerank_k]:
                needed_ids.add(d)

            # Store k1 doc ordering for pass2
            item["pairs1"] = pairs1

        # Load only needed doc texts from collection.tsv
        if collection_path and ce_model is not None and needed_ids:
            print(f"   Loading {len(needed_ids)} doc texts from collection...")
            with open(collection_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.split("\t", 1)
                    if len(parts) == 2 and parts[0] in needed_ids:
                        doc_texts[parts[0]] = parts[1].rstrip("\n")
                        if len(doc_texts) >= len(needed_ids):
                            break
            print(f"   Loaded {len(doc_texts)}/{len(needed_ids)} doc texts")

        diag_rows = []
        gated_cnt = 0
        action_cnt = 0

        for item in tqdm(cache, desc=f"{ds_key} pass2(policy)"):
            qid = item["qid"]
            qtext = item["qtext"]
            pairs1 = item["pairs1"]  # pre-computed in k1-retrieval step

            # initialize ours from Base@k1 (no-op unless policy acts)
            run_ours[qid] = dict(run_base_k1[qid])

            risk = float(item["risk"])
            gated = int(risk >= tau)
            if gated:
                gated_cnt += 1

            action_taken = 0
            skip_reason = "none"

            if gated and ce_model is not None:

                # confidence lock
                if cfg.confidence_lock and (float(item["ent"]) <= ent_tau) and (float(item["mnorm"]) >= mnorm_tau):
                    skip_reason = "confidence_lock"
                else:
                    cand_docids = [d for d, _ in pairs1]  # already sorted by base@k1
                    topk = min(cfg.ce_rerank_k, len(cand_docids))
                    head_docids = cand_docids[:topk]
                    tail_docids = cand_docids[topk:]

                    base_head_scores = np.array([run_base_k1[qid][d] for d in head_docids], dtype=np.float32)

                    ce_pairs = []
                    for did in head_docids:
                        dtext = doc_texts.get(did, "")
                        ce_pairs.append([qtext, dtext])
                    ce_scores = ce_model.predict(ce_pairs).astype(np.float32) if ce_pairs else np.zeros((0,), dtype=np.float32)

                    if cfg.agreement_gate:
                        ak = min(cfg.agree_k, topk)
                        base_top = head_docids[:ak]
                        ce_rank = sorted(list(zip(head_docids, ce_scores)), key=lambda x: float(x[1]), reverse=True)
                        ce_top = [d for d, _ in ce_rank[:ak]]
                        ov = overlap_count(base_top, ce_top)
                        if ov < cfg.agree_min_overlap:
                            skip_reason = f"agree_gate<{cfg.agree_min_overlap} (ov={ov})"
                        else:
                            # adaptive alpha
                            high_conf = (float(item["ent"]) <= ent_tau) and (float(item["mnorm"]) >= mnorm_tau)
                            low_conf = (float(item["mnorm"]) <= mnorm_q30) and (float(item["ent"]) >= ent_q70)
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

                            # top-N lock
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
                        # no agreement gate: just blend
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

            # Per-query diagnostics (internal nDCG@eval_k)
            nd0 = ndcg_at_k_from_run(qid, run_base_k0, qrels, k=cfg.eval_k)
            nd1 = ndcg_at_k_from_run(qid, run_base_k1, qrels, k=cfg.eval_k)
            ndo = ndcg_at_k_from_run(qid, run_ours, qrels, k=cfg.eval_k)

            diag_rows.append({
                "qid": qid,
                "risk": risk,
                "gated": gated,
                "action_taken": action_taken,
                "skip_reason": skip_reason,
                "ent": float(item["ent"]),
                "mnorm": float(item["mnorm"]),
                "ndcg_base_k0": nd0,
                "ndcg_base_k1": nd1,
                "ndcg_ours": ndo,
                "delta_net": ndo - nd0,
            })

        # ----- CE-All baseline: apply CE to every query, no safety gates -----
        run_ce_all: Dict[str, Dict[str, float]] = {}
        if ce_model is not None:
            ce_all_alpha = cfg.base_alpha
            for item in tqdm(cache, desc=f"{ds_key} pass3(CE-All)"):
                qid = item["qid"]
                qtext = item["qtext"]
                qvec = item["qvec"]

                # Reuse the k1 retrieval from run_base_k1
                cand_docids = sorted(run_base_k1[qid].keys(),
                                     key=lambda d: run_base_k1[qid][d], reverse=True)
                topk = min(cfg.ce_rerank_k, len(cand_docids))
                head_docids = cand_docids[:topk]
                tail_docids = cand_docids[topk:]

                base_head_scores = np.array([run_base_k1[qid][d] for d in head_docids], dtype=np.float32)

                ce_pairs = []
                for did in head_docids:
                    dtext = doc_texts.get(did, "")
                    ce_pairs.append([qtext, dtext])
                ce_scores_arr = ce_model.predict(ce_pairs).astype(np.float32) if ce_pairs else np.zeros((0,), dtype=np.float32)

                zb = zscore(base_head_scores)
                zc = zscore(ce_scores_arr)
                blended = ce_all_alpha * zb + (1.0 - ce_all_alpha) * zc

                head_sorted = list(zip(head_docids, blended, base_head_scores))
                head_sorted.sort(key=lambda x: (float(x[1]), float(x[2])), reverse=True)
                new_head_ids = [d for d, _, _ in head_sorted]

                merged = new_head_ids + tail_docids
                run_ce_all[qid] = {did: float(1000.0 - i) for i, did in enumerate(merged)}
        else:
            # CE disabled: CE-All falls back to Base@k1
            run_ce_all = {qid: dict(scores) for qid, scores in run_base_k1.items()}

        # Aggregate internal metrics (for quick sanity; use pytrec_eval for paper tables)
        m0 = eval_run_metrics(run_base_k0, qrels, ndcg_k=cfg.eval_k, map_k=cfg.map_k, mrr_k=cfg.mrr_k)
        m1 = eval_run_metrics(run_base_k1, qrels, ndcg_k=cfg.eval_k, map_k=cfg.map_k, mrr_k=cfg.mrr_k)
        mo = eval_run_metrics(run_ours,    qrels, ndcg_k=cfg.eval_k, map_k=cfg.map_k, mrr_k=cfg.mrr_k)
        mc = eval_run_metrics(run_ce_all,  qrels, ndcg_k=cfg.eval_k, map_k=cfg.map_k, mrr_k=cfg.mrr_k)

        nd0, nd1, ndo = m0[f"nDCG@{cfg.eval_k}"], m1[f"nDCG@{cfg.eval_k}"], mo[f"nDCG@{cfg.eval_k}"]
        ap0, ap1, apo = m0[f"MAP@{cfg.map_k}"],  m1[f"MAP@{cfg.map_k}"],  mo[f"MAP@{cfg.map_k}"]
        rr0, rr1, rro = m0[f"MRR@{cfg.mrr_k}"],  m1[f"MRR@{cfg.mrr_k}"],  mo[f"MRR@{cfg.mrr_k}"]
        ndc, apc, rrc = mc[f"nDCG@{cfg.eval_k}"], mc[f"MAP@{cfg.map_k}"],  mc[f"MRR@{cfg.mrr_k}"]

        print(f"\n[RESULTS] {ds_key}")
        if ds_key == "MSMARCO_Dev":
            print(f"   [NOTE] MS MARCO Dev uses binary relevance; MRR@{cfg.mrr_k} is the primary metric")
        print(f"   Gated: {gated_cnt}/{len(cache)} ({100*gated_cnt/len(cache):.1f}%) | tau={tau:.6f}")
        print(f"   CE actions: {action_cnt}")
        print(f"   Base@k0: nDCG@{cfg.eval_k}={nd0:.4f} | MAP@{cfg.map_k}={ap0:.4f} | MRR@{cfg.mrr_k}={rr0:.4f}")
        print(f"   Base@k1: nDCG@{cfg.eval_k}={nd1:.4f} | MAP@{cfg.map_k}={ap1:.4f} | MRR@{cfg.mrr_k}={rr1:.4f}")
        print(f"   CE-All:  nDCG@{cfg.eval_k}={ndc:.4f} | MAP@{cfg.map_k}={apc:.4f} | MRR@{cfg.mrr_k}={rrc:.4f} "
              f"(DnDCG={ndc-nd0:+.4f}, DMAP={apc-ap0:+.4f}, DMRR={rrc-rr0:+.4f})")
        print(f"   Ours:    nDCG@{cfg.eval_k}={ndo:.4f} | MAP@{cfg.map_k}={apo:.4f} | MRR@{cfg.mrr_k}={rro:.4f} "
              f"(DnDCG={ndo-nd0:+.4f}, DMAP={apo-ap0:+.4f}, DMRR={rro-rr0:+.4f})")

        df = pd.DataFrame(diag_rows)
        diag_path = os.path.join(cfg.results_dir, f"diagnostics_{ds_key}.csv")
        df.to_csv(diag_path, index=False)
        summarize_diagnostics(df, name=f"({ds_key})")
        print(f"[SAVE] Saved diagnostics: {diag_path}")

        # Export trec run files (evaluate with tools/eval_pytrec.py)
        write_trec_run(run_base_k0, os.path.join(cfg.results_dir, f"run_base_k0_{ds_key}.trec"), run_name="base_k0", depth=cfg.k1)
        write_trec_run(run_base_k1, os.path.join(cfg.results_dir, f"run_base_k1_{ds_key}.trec"), run_name="base_k1", depth=cfg.k1)
        write_trec_run(run_ce_all,  os.path.join(cfg.results_dir, f"run_ce_all_{ds_key}.trec"),  run_name="ce_all",  depth=cfg.k1)
        write_trec_run(run_ours,    os.path.join(cfg.results_dir, f"run_ra_sce_{ds_key}.trec"),  run_name="ra_sce",  depth=cfg.k1)
        print(f"[SAVE] Wrote TREC run files for {ds_key} to: {cfg.results_dir}")

        # Summary rows
        summary_rows.append({
            "Dataset": ds_key,
            "Method": "Base@k0",
            f"nDCG@{cfg.eval_k}": nd0,
            f"MAP@{cfg.map_k}": ap0,
            f"MRR@{cfg.mrr_k}": rr0,
        })
        summary_rows.append({
            "Dataset": ds_key,
            "Method": "Base@k1",
            f"nDCG@{cfg.eval_k}": nd1,
            f"MAP@{cfg.map_k}": ap1,
            f"MRR@{cfg.mrr_k}": rr1,
        })
        summary_rows.append({
            "Dataset": ds_key,
            "Method": "CE-All",
            f"nDCG@{cfg.eval_k}": ndc,
            f"MAP@{cfg.map_k}": apc,
            f"MRR@{cfg.mrr_k}": rrc,
            f"Delta_nDCG@{cfg.eval_k}": ndc - nd0,
            f"Delta_MAP@{cfg.map_k}": apc - ap0,
            f"Delta_MRR@{cfg.mrr_k}": rrc - rr0,
        })
        summary_rows.append({
            "Dataset": ds_key,
            "Method": "Ours(RA-SCEv4)",
            f"nDCG@{cfg.eval_k}": ndo,
            f"MAP@{cfg.map_k}": apo,
            f"MRR@{cfg.mrr_k}": rro,
            f"Delta_nDCG@{cfg.eval_k}": ndo - nd0,
            f"Delta_MAP@{cfg.map_k}": apo - ap0,
            f"Delta_MRR@{cfg.mrr_k}": rro - rr0,
            "GatedCount": gated_cnt,
            "GatedPct": 100.0 * gated_cnt / max(1, len(cache)),
            "CE_Actions": action_cnt,
            "Tau": tau,
        })

        summary_path = os.path.join(cfg.results_dir, "ra_sce_v4_summary.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"[SAVE] Updated summary: {summary_path}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n[DONE] Final summary: {os.path.join(cfg.results_dir, 'ra_sce_v4_summary.csv')}")


if __name__ == "__main__":
    args = parse_args()

    # 1) File overrides (base)
    cfg = CFG
    if getattr(args, "override_yaml", None):
        od = load_override_file(args.override_yaml)
        cfg = apply_dict_overrides(cfg, od)

    # 2) CLI overrides (highest priority)
    cfg = apply_overrides(cfg, args)

    os.makedirs(cfg.results_dir, exist_ok=True)
    main(cfg)

