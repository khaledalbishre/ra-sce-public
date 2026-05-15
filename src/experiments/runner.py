# src/experiments/runner.py
import copy


def expand_sweep(run_cfg: dict, experiments: dict) -> dict:
    # returns a dict experiment_name -> patch
    sweep = run_cfg["experiments_from_sweep"]
    base_name = sweep["base"]
    param = sweep["param"]
    values = sweep["values"]

    base_patch = experiments[base_name].get("patch", {})
    expanded = {}
    for v in values:
        name = f"{base_name}__{param.replace('.','_')}={v}"
        patch = copy.deepcopy(base_patch)
        # apply dotted param override
        # (wrap it as a dict patch if you prefer, or set on merged config later)
        expanded[name] = {"__dotted_overrides__": {param: v}, **patch}
    return expanded
