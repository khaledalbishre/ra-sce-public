# src/experiments/config.py
import copy, yaml
from pathlib import Path

def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def load_yaml_with_includes(path: str) -> dict:
    cfg = {}
    doc = yaml.safe_load(Path(path).read_text())
    for inc in doc.get("includes", []):
        inc_doc = load_yaml_with_includes(inc)
        cfg = deep_merge(cfg, inc_doc)
    doc.pop("includes", None)
    cfg = deep_merge(cfg, doc)
    return cfg

def set_by_dotted_path(cfg: dict, dotted: str, value):
    keys = dotted.split(".")
    cur = cfg
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
