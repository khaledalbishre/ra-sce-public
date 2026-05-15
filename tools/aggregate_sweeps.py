#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Stats
from scipy import stats


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps_dir", type=str, default="./results/sweeps",
                    help="Path to sweeps directory (contains dl19_safe, dl19_medium, ...)")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Output directory. Default: <sweeps_dir>/_aggregate")
    ap.add_argument("--prefer_base", type=str, default="k1", choices=["k0", "k1"],
                    help="Which base to use when both base_k0 and base_k1 exist in diagnostics.")
    ap.add_argument("--write_xlsx", action="store_true",
                    help="Also write Excel .xlsx (requires openpyxl).")
    return ap.parse_args()


def safe_read_csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def parse_pytrec_eval(path: Path) -> dict:
    """
    Parses tools/eval_pytrec.py output saved as pytrec_eval.txt.
    Expects lines: 'metric: value'
    """
    if not path.exists():
        return {}
    d = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        v = v.strip()
        try:
            d[k] = float(v)
        except Exception:
            pass
    return d


def find_ndcg_cols(df: pd.DataFrame, prefer_base="k1"):
    """
    Heuristics: find base ndcg column (k1 preferred), ours ndcg column, or delta column.
    Works with various naming conventions.

    Returns: (base_col or None, ours_col or None, delta_col or None)
    """
    cols = list(df.columns)

    # delta candidates
    delta_candidates = [c for c in cols if "delta" in c.lower() and ("ndcg" in c.lower() or "net" in c.lower())]
    delta_col = None
    if delta_candidates:
        dn = [c for c in delta_candidates if re.search(r"delta(_)?net", c, re.I)]
        delta_col = dn[0] if dn else delta_candidates[0]

    # ours candidates
    ours_candidates = [c for c in cols if re.search(r"(ours|ra_sce|final|after)", c, re.I) and ("ndcg" in c.lower())]
    ours_col = ours_candidates[0] if ours_candidates else None

    base_k1 = [c for c in cols if re.search(r"base(_)?k1", c, re.I) and ("ndcg" in c.lower())]
    base_k0 = [c for c in cols if re.search(r"base(_)?k0", c, re.I) and ("ndcg" in c.lower())]
    base_generic = [c for c in cols if re.search(r"\bbase\b", c, re.I) and ("ndcg" in c.lower())]

    if prefer_base == "k1":
        base_col = base_k1[0] if base_k1 else (base_k0[0] if base_k0 else (base_generic[0] if base_generic else None))
    else:
        base_col = base_k0[0] if base_k0 else (base_k1[0] if base_k1 else (base_generic[0] if base_generic else None))

    return base_col, ours_col, delta_col


def paired_tests(delta: np.ndarray):
    """
    Paired tests against 0: one-sample t-test and Wilcoxon signed-rank.
    """
    delta = delta[np.isfinite(delta)]
    n = len(delta)

    out = {
        "n_queries": n,
        "mean_delta": float(np.mean(delta)) if n else np.nan,
        "median_delta": float(np.median(delta)) if n else np.nan,
        "std_delta": float(np.std(delta, ddof=1)) if n > 1 else np.nan,
        "helped_pct": float(np.mean(delta > 1e-12) * 100) if n else np.nan,
        "neutral_pct": float(np.mean(np.abs(delta) <= 1e-12) * 100) if n else np.nan,
        "harmed_pct": float(np.mean(delta < -1e-12) * 100) if n else np.nan,
        "ttest_p": np.nan,
        "wilcoxon_p": np.nan,
    }

    if n < 5:
        return out

    try:
        _, tp = stats.ttest_1samp(delta, 0.0)
        out["ttest_p"] = float(tp)
    except Exception:
        pass

    try:
        _, wp = stats.wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
        out["wilcoxon_p"] = float(wp)
    except Exception:
        pass

    return out


def load_ce_actions(exp_dir: Path):
    """
    Try to extract ce_actions from meta.json or summary csv.
    """
    # meta.json
    meta_path = exp_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            for k in ["ce_actions", "ce_actions_count", "ce_action_count", "ce_actions_total"]:
                if k in meta:
                    return float(meta[k])
        except Exception:
            pass

    # summary csv (if your code writes ce_actions column)
    sum_path = exp_dir / "ra_sce_v4_summary.csv"
    if sum_path.exists():
        df = safe_read_csv(sum_path)
        if df is not None and not df.empty:
            for c in df.columns:
                if re.search(r"ce.*action", c, re.I):
                    try:
                        return float(df[c].iloc[0])
                    except Exception:
                        pass

    return np.nan


def main():
    args = parse_args()
    sweeps_dir = Path(args.sweeps_dir)
    if not sweeps_dir.exists():
        raise FileNotFoundError(f"sweeps_dir not found: {sweeps_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else (sweeps_dir / "_aggregate")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    per_query_rows = []

    exp_dirs = sorted([p for p in sweeps_dir.iterdir() if p.is_dir()])
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name

        # dataset by which diagnostics exist
        diag19 = exp_dir / "diagnostics_TREC_DL_19.csv"
        diag20 = exp_dir / "diagnostics_TREC_DL_20.csv"

        datasets = []
        if diag19.exists():
            datasets.append(("TREC_DL_19", diag19))
        if diag20.exists():
            datasets.append(("TREC_DL_20", diag20))

        if not datasets:
            continue

        # read pytrec metrics if available
        pytrec = parse_pytrec_eval(exp_dir / "pytrec_eval.txt")

        ce_actions = load_ce_actions(exp_dir)

        for ds_key, diag_path in datasets:
            df = safe_read_csv(diag_path)
            if df is None or df.empty:
                continue

            base_col, ours_col, delta_col = find_ndcg_cols(df, prefer_base=args.prefer_base)

            if delta_col is not None:
                delta = df[delta_col].to_numpy(dtype=float)
            elif base_col is not None and ours_col is not None:
                delta = (df[ours_col].to_numpy(dtype=float) - df[base_col].to_numpy(dtype=float))
            else:
                # Can't compute deltas; skip stats but still record pytrec metrics
                delta = np.array([])

            tests = paired_tests(delta)
            n_q = tests["n_queries"]
            ce_pct = (ce_actions / n_q * 100) if (n_q and np.isfinite(ce_actions)) else np.nan

            # headline metrics from pytrec_eval.txt
            ndcg10 = pytrec.get("ndcg_cut.10", np.nan)
            ndcg20 = pytrec.get("ndcg_cut.20", np.nan)
            map1000 = pytrec.get("map_cut.1000", np.nan)
            recall1000 = pytrec.get("recall.1000", np.nan)
            recip_rank = pytrec.get("recip_rank", np.nan)
            mrr10 = pytrec.get("mrr@10 (manual cutoff)", np.nan)
            if not np.isfinite(mrr10):
                mrr10 = recip_rank  # fallback

            summary_rows.append({
                "experiment": exp_name,
                "dataset": ds_key,
                "n_queries": n_q,
                "ce_actions": ce_actions,
                "ce_usage_pct": ce_pct,
                "ndcg@10": ndcg10,
                "ndcg@20": ndcg20,
                "map@1000": map1000,
                "mrr@10": mrr10,
                "recall@1000": recall1000,
                "mean_delta": tests["mean_delta"],
                "median_delta": tests["median_delta"],
                "std_delta": tests["std_delta"],
                "helped_pct": tests["helped_pct"],
                "neutral_pct": tests["neutral_pct"],
                "harmed_pct": tests["harmed_pct"],
                "ttest_p": tests["ttest_p"],
                "wilcoxon_p": tests["wilcoxon_p"],
            })

            # per-query deltas file for plotting later
            if len(delta) and ("qid" in df.columns):
                pq = pd.DataFrame({
                    "experiment": exp_name,
                    "dataset": ds_key,
                    "qid": df["qid"],
                    "delta_ndcg": delta
                })
                per_query_rows.append(pq)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise RuntimeError("No experiments found / no diagnostics parsed. Check sweeps_dir structure.")

    summary_df = summary_df.sort_values(["dataset", "ce_usage_pct", "experiment"], na_position="last")

    per_query_df = pd.concat(per_query_rows, ignore_index=True) if per_query_rows else pd.DataFrame()

    # write outputs
    out_csv = out_dir / "aggregate_results.csv"
    out_pq = out_dir / "per_query_deltas.csv"
    summary_df.to_csv(out_csv, index=False)
    per_query_df.to_csv(out_pq, index=False)

    print(f"[OK] Wrote: {out_csv}")
    print(f"[OK] Wrote: {out_pq}")

    if args.write_xlsx:
        out_xlsx = out_dir / "aggregate_results.xlsx"
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="summary", index=False)
            per_query_df.to_excel(writer, sheet_name="per_query_deltas", index=False)
        print(f"[OK] Wrote: {out_xlsx}")


if __name__ == "__main__":
    main()
