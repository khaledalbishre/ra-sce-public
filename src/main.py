#!/usr/bin/env python3
"""
YAML-driven experiment runner for RA-SCE.

Features:
- YAML includes (defaults + datasets + experiments)
- Deep-merge config patches
- Experiment registry + parameter sweeps
- Per-experiment output folder with config snapshot
- Calls existing benchmark entrypoint: benchmark_ra_sce_v4.main(Config)

Expected YAML schema (high level):
- includes: [paths...]
- datasets: { datasets: [ {name, ir_datasets_id}, ... ] }   OR  datasets: [ ... ]
- experiments: { <exp_name>: {patch: {...}, notes: "..."} , ... }
- runs: { <run_name>: {experiments:[...], output_tag:"..."} , ... }
- sweeps (optional): via runs.<run_name>.experiments_from_sweep

Usage:
  python -m src.main --config configs/experiments.yaml --run ra_sce_main
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


# -------------------------
# YAML loading + deep merge
# -------------------------

def deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `patch` into `base` (without mutating inputs)."""
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml_with_includes(path: str, _base_dir: Path | None = None) -> Dict[str, Any]:
    """
    Load a YAML config that may contain `includes: [...]`.

    Includes are resolved relative to the YAML file's directory (recommended),
    or to `_base_dir` if provided (used internally for recursion).
    """
    p = Path(path)
    if not p.is_absolute():
        p = (_base_dir or Path.cwd()) / p
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    doc = yaml.safe_load(p.read_text()) or {}
    cfg: Dict[str, Any] = {}

    base_dir = p.parent
    for inc in doc.get("includes", []) or []:
        inc_cfg = load_yaml_with_includes(inc, _base_dir=base_dir)
        cfg = deep_merge(cfg, inc_cfg)

    doc.pop("includes", None)
    cfg = deep_merge(cfg, doc)
    return cfg


def set_by_dotted_path(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    """Set cfg['a']['b']['c'] given 'a.b.c'."""
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def get_by_dotted_path(cfg: Dict[str, Any], dotted: str) -> Any:
    keys = dotted.split(".")
    cur: Any = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


# -------------------------
# Experiment expansion
# -------------------------

def normalize_datasets(cfg: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Return list[(dataset_name, ir_datasets_id)].

    Supported YAML schemas:
      1) datasets: { datasets: [ {name, ir_datasets_id}, ... ] }
      2) datasets: [ {name, ir_datasets_id}, ... ]
    """
    ds_block = cfg.get("datasets", {})
    if isinstance(ds_block, list):
        ds_list = ds_block
    elif isinstance(ds_block, dict):
        ds_list = ds_block.get("datasets", ds_block.get("list", []))
    else:
        ds_list = []

    if not ds_list:
        raise ValueError("No datasets found. Expected cfg['datasets'] to be a list or a dict containing 'datasets'.")

    out: List[Tuple[str, str]] = []
    for d in ds_list:
        name = d["name"]
        did = d["ir_datasets_id"]
        out.append((name, did))
    return out


def build_experiment_cfg(base_cfg: Dict[str, Any], exp_patch: Dict[str, Any]) -> Dict[str, Any]:
    """Apply experiment patch over defaults."""
    return deep_merge(base_cfg, exp_patch)


def expand_sweep(
    cfg_all: Dict[str, Any],
    run_spec: Dict[str, Any],
    experiments: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """
    Expand runs.<run_name>.experiments_from_sweep into concrete experiments.

    Example:
      experiments_from_sweep:
        base: "ra_sce_full"
        param: "policy.rho"
        values: [0.1,0.2,0.3]
    """
    sweep = run_spec["experiments_from_sweep"]
    base_name = sweep["base"]
    param = sweep["param"]
    values = sweep["values"]

    if base_name not in experiments:
        raise KeyError(f"Sweep base experiment '{base_name}' not found in experiments registry.")

    base_patch = experiments[base_name].get("patch", {}) or {}

    expanded: Dict[str, Dict[str, Any]] = {}
    for v in values:
        exp_name = f"{base_name}__{param.replace('.', '_')}={v}"
        expanded[exp_name] = {
            "__patch__": copy.deepcopy(base_patch),
            "__dotted_overrides__": {param: v},
            "__notes__": f"sweep {param}={v}",
        }
    return expanded


# -------------------------
# Benchmark adapter
# -------------------------

def adapt_yaml_to_benchmark_kwargs(cfg: Dict[str, Any], datasets: List[Tuple[str, str]], results_dir: str) -> Dict[str, Any]:
    paths = cfg.get("paths", {}) or {}
    models = cfg.get("models", {}) or {}
    retrieval = cfg.get("retrieval", {}) or {}
    rerank = cfg.get("rerank", {}) or {}
    policy = cfg.get("policy", {}) or {}
    eval_cfg = cfg.get("evaluation", {}) or {}
    adaptive = (policy.get("adaptive_blend", {}) or {})

    kwargs = {
        "index_path": paths.get("index_path"),
        "id_path": paths.get("id_path"),
        "results_dir": results_dir,
        "model_name": models.get("biencoder"),
        "ce_model_name": models.get("cross_encoder"),
        "k0": retrieval.get("k0"),
        "k1": retrieval.get("k1"),
        "kg": retrieval.get("kg"),
        "rho": policy.get("rho"),
        "use_cross_encoder": rerank.get("use_cross_encoder"),
        "ce_rerank_k": rerank.get("ce_rerank_k"),
        "confidence_lock": policy.get("confidence_lock"),
        "ent_lock_quantile": policy.get("ent_lock_quantile"),
        "mnorm_lock_quantile": policy.get("mnorm_lock_quantile"),
        "agreement_gate": policy.get("agreement_gate"),
        "agree_k": policy.get("agree_k"),
        "agree_min_overlap": policy.get("agree_min_overlap"),
        "base_alpha": adaptive.get("base_alpha", policy.get("base_alpha")),
        "alpha_high_conf": adaptive.get("alpha_high_conf", policy.get("alpha_high_conf")),
        "alpha_low_conf": adaptive.get("alpha_low_conf", policy.get("alpha_low_conf")),
        "topn_lock": policy.get("topn_lock"),
        "topn_lock_quantile": policy.get("topn_lock_quantile"),
        "topn": policy.get("topn"),
        "eval_k": eval_cfg.get("primary_k", eval_cfg.get("eval_k", 10)),
        # ✅ IMPORTANT: pass ALL datasets in a single benchmark call to avoid summary overwrite
        "datasets": datasets,
    }

    return {k: v for k, v in kwargs.items() if v is not None}


def run_one_experiment_on_all_datasets(
    *,
    cfg_all: Dict[str, Any],
    exp_name: str,
    exp_cfg: Dict[str, Any],
    datasets: List[Tuple[str, str]],
    out_root: str,
) -> None:
    """
    Run one experiment ONCE with all datasets (prevents ra_sce_v4_summary.csv overwrite).
    Writes outputs under: <out_root>/<exp_name>/

    Note: indexes are read from cfg paths (e.g., artifacts/indexes/...) and are not deleted.
    """
    import benchmark_ra_sce_v4 as bench

    if not datasets:
        raise ValueError("No datasets provided. Check cfg['datasets'].")

    exp_dir = Path(out_root) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save per-experiment snapshots (useful for reproducibility)
    (exp_dir / "experiment_config.yaml").write_text(yaml.safe_dump(exp_cfg, sort_keys=False))
    meta = {
        "exp_name": exp_name,
        "datasets": [{"name": n, "ir_datasets_id": did} for (n, did) in datasets],
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    (exp_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    kwargs = adapt_yaml_to_benchmark_kwargs(exp_cfg, datasets, results_dir=str(exp_dir))
    cfg_obj = bench.Config(**kwargs)

    print(f"\n=== Running: {exp_name} | Datasets: {[d[0] for d in datasets]} | results_dir={exp_dir} ===")
    bench.main(cfg_obj)


# -------------------------
# CLI
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (e.g., configs/experiments.yaml)")
    ap.add_argument("--run", required=True, help="Run name under cfg['runs']")
    ap.add_argument("--results-root", default=None, help="Override paths.results_root (optional)")
    args = ap.parse_args()

    cfg_all = load_yaml_with_includes(args.config)

    if args.run not in cfg_all.get("runs", {}):
        raise KeyError(f"Run '{args.run}' not found in cfg['runs'].")

    base_cfg = copy.deepcopy(cfg_all)

    datasets = normalize_datasets(cfg_all)
    experiments_registry = cfg_all.get("experiments", {}) or {}
    run_spec = cfg_all["runs"][args.run]

    # Determine output root folder
    # Prefer: CLI override > YAML paths.results_root > ./results
    paths = cfg_all.get("paths", {}) or {}
    results_root = args.results_root or paths.get("results_root") or "./results"
    output_tag = run_spec.get("output_tag", args.run)
    out_root = Path(results_root) / output_tag
    out_root.mkdir(parents=True, exist_ok=True)

    # Save the "run-level" config snapshot
    (out_root / "run_config.yaml").write_text(yaml.safe_dump(cfg_all, sort_keys=False))

    # Collect experiments list
    exp_cfgs: Dict[str, Dict[str, Any]] = {}

    if "experiments_from_sweep" in run_spec:
        expanded = expand_sweep(cfg_all, run_spec, experiments_registry)
        for exp_name, pack in expanded.items():
            patch = pack.get("__patch__", {}) or {}
            exp_cfg = build_experiment_cfg(base_cfg, patch)
            for dotted, v in (pack.get("__dotted_overrides__", {}) or {}).items():
                set_by_dotted_path(exp_cfg, dotted, v)
            exp_cfgs[exp_name] = exp_cfg
    else:
        exp_list = run_spec.get("experiments", [])
        if not exp_list:
            raise ValueError(f"Run '{args.run}' has no experiments and no sweep definition.")
        for exp_name in exp_list:
            if exp_name not in experiments_registry:
                raise KeyError(f"Experiment '{exp_name}' not found in experiments registry.")
            patch = experiments_registry[exp_name].get("patch", {}) or {}
            exp_cfgs[exp_name] = build_experiment_cfg(base_cfg, patch)

    # Execute
    for exp_name, exp_cfg in exp_cfgs.items():
        run_one_experiment_on_all_datasets(
            cfg_all=cfg_all,
            exp_name=exp_name,
            exp_cfg=exp_cfg,
            datasets=datasets,
            out_root=str(out_root),
        )

    print(f"\n✅ Done. Results saved under: {out_root.resolve()}")


if __name__ == "__main__":
    main()
