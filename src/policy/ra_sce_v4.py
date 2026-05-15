# src/policy/ra_sce_v4.py
"""
RA-SCE v4 policy module (inference-time, training-free).

Design goals
- Single entry: apply_policy_for_query(...)
- Deterministic, stateless per query (except thresholds passed in)
- Rich metadata for diagnostics CSV
- Easy to ablate via config flags (agreement_gate, confidence_lock, topn_lock, adaptive_blend)

Expected inputs
- base_scores: dict[docid] -> float (baseline scores for candidate pool, ideally k1 depth)
- ce_scores:   dict[docid] -> float (cross-encoder scores for subset; can be empty if not used)
- thresholds:  precomputed quantile-based thresholds for this dataset/split
- cfg:         policy config

Key concepts (matching your paper)
- Risk-budgeted gating: intervene only for top-ρ risk queries (r(q) >= tau_risk)
- Confidence lock: skip if entropy low AND margin_norm high
- Agreement veto: skip if overlap(topK_base, topK_ce) < min_overlap
- Adaptive blend: alpha(q) chosen from {alpha_high_conf, base_alpha, alpha_low_conf}
- Top-N lock: preserve top-N from baseline when margin_norm >= topn_lock_mnorm_threshold
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
import math


ScoreDict = Dict[str, float]


# ----------------------------
# Config + Threshold containers
# ----------------------------

@dataclass(frozen=True)
class AdaptiveBlendCfg:
    base_alpha: float = 0.60
    alpha_high_conf: float = 0.75
    alpha_low_conf: float = 0.45


@dataclass(frozen=True)
class PolicyCfg:
    # Gating
    rho: float = 0.30

    # Confidence lock
    confidence_lock: bool = True
    ent_lock_quantile: float = 0.35
    mnorm_lock_quantile: float = 0.75

    # Agreement gate (veto)
    agreement_gate: bool = True
    agree_k: int = 20
    agree_min_overlap: int = 6

    # Adaptive blend
    adaptive_blend: bool = True
    blend: AdaptiveBlendCfg = AdaptiveBlendCfg()

    # Top-N lock
    topn_lock: bool = True
    topn_lock_quantile: float = 0.85
    topn: int = 2

    # Risk feature weights (lightweight, training-free)
    # risk = w_gap * gap + w_ent * ent - w_disp * disp
    w_gap: float = 1.0
    w_ent: float = 0.25
    w_disp: float = 0.05

    # which top portion to compute features on
    kg: int = 50

    # CE blending behaviors
    default_action: str = "ce_blend"  # for diagnostics labeling


@dataclass(frozen=True)
class Thresholds:
    # risk gating threshold
    tau_risk: float

    # confidence lock thresholds
    ent_lock_max: float
    mnorm_lock_min: float

    # alpha selection thresholds (optional; we derive from same quantiles by default)
    # if not provided, alpha selection uses lock thresholds as proxy
    ent_high_conf_max: Optional[float] = None
    mnorm_high_conf_min: Optional[float] = None
    ent_low_conf_min: Optional[float] = None
    mnorm_low_conf_max: Optional[float] = None

    # topN lock threshold
    mnorm_topn_lock_min: float = 0.0


@dataclass
class PolicyDecision:
    # final ranking
    ranked_docids: List[str]

    # metadata
    risk: float
    ent: float
    mnorm: float
    disp: float
    gap: float

    gated: int
    action_taken: str
    skip_reason: str

    alpha: float
    overlap_topk: int

    topn_locked: int
    locked_n: int


# ----------------------------
# Utilities: ranking + stats
# ----------------------------

def _topk_sorted(scores: ScoreDict, k: int) -> List[Tuple[str, float]]:
    items = list(scores.items())
    items.sort(key=lambda x: (-x[1], x[0]))  # deterministic tie-break
    return items[:k]


def _ranked_docids(scores: ScoreDict) -> List[str]:
    return [d for d, _ in _topk_sorted(scores, k=len(scores))]


def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    if s <= 0:
        # fallback uniform
        return [1.0 / len(xs)] * len(xs)
    return [e / s for e in exps]


def _entropy_from_scores(top_scores: List[float]) -> float:
    """
    Entropy of softmax(scores) over the head, in nats.
    Higher entropy => flatter / less confident.
    """
    p = _softmax(top_scores)
    ent = 0.0
    for pi in p:
        if pi > 1e-12:
            ent -= pi * math.log(pi)
    return ent


def _margin_norm(top_scores: List[float]) -> float:
    """
    Normalized margin between top-1 and top-2, divided by std of head.
    Matches benchmark: mnorm = (head[0] - head[1]) / (std(head) + 1e-6)
    """
    if len(top_scores) < 2:
        return 0.0
    m12 = top_scores[0] - top_scores[1]
    n = len(top_scores)
    mu = sum(top_scores) / n
    var = sum((x - mu) ** 2 for x in top_scores) / max(1, n - 1)
    std = max(var, 0.0) ** 0.5
    return m12 / (std + 1e-6)


def _dispersion(top_scores: List[float]) -> float:
    """
    Simple dispersion proxy: stddev of head scores.
    """
    n = len(top_scores)
    if n <= 1:
        return 0.0
    mu = sum(top_scores) / n
    var = sum((x - mu) ** 2 for x in top_scores) / (n - 1)
    return math.sqrt(max(0.0, var))


def _gap(top_scores: List[float]) -> float:
    """
    Position-weighted log-gap feature, matching benchmark's weighted_gap_feature.

    Computes log1p(|diff|) between consecutive scores, weighted by 1/log2(rank+2).
    Uses score diffs as proxy for distance diffs (monotonically related).
    """
    n = len(top_scores)
    if n < 2:
        return 0.0
    # consecutive diffs (last element: diff with itself = 0)
    diffs = [top_scores[i] - top_scores[i + 1] for i in range(n - 1)]
    diffs.append(0.0)
    geo = [math.log1p(abs(d)) for d in diffs]
    weights = [1.0 / math.log2(i + 2.0) for i in range(n)]
    w_sum = sum(weights)
    if w_sum < 1e-9:
        return 0.0
    return sum(g * w for g, w in zip(geo, weights)) / w_sum


def _z_normalize_on_keys(scores: ScoreDict, keys: List[str]) -> Dict[str, float]:
    vals = [scores.get(k, 0.0) for k in keys]
    n = len(vals)
    if n == 0:
        return {k: 0.0 for k in keys}
    mu = sum(vals) / n
    var = sum((x - mu) ** 2 for x in vals) / max(1, n - 1)
    std = math.sqrt(max(1e-12, var))
    return {k: (scores.get(k, 0.0) - mu) / std for k in keys}


def _overlap_topk(a_scores: ScoreDict, b_scores: ScoreDict, k: int) -> int:
    a_top = {d for d, _ in _topk_sorted(a_scores, k)}
    b_top = {d for d, _ in _topk_sorted(b_scores, k)}
    return len(a_top & b_top)


# ----------------------------
# Feature extraction
# ----------------------------

@dataclass(frozen=True)
class QuerySignals:
    risk: float
    ent: float
    mnorm: float
    disp: float
    gap: float


def compute_query_signals(base_scores: ScoreDict, cfg: PolicyCfg) -> QuerySignals:
    head = _topk_sorted(base_scores, cfg.kg)
    head_scores = [s for _, s in head]

    ent = _entropy_from_scores(head_scores)
    mnorm = _margin_norm(head_scores)
    disp = _dispersion(head_scores)
    gap = _gap(head_scores)

    # risk score: higher means "more likely to benefit"
    # You can tune weights; these defaults mirror your earlier style (gap + 0.25*ent - 0.05*disp).
    risk = (cfg.w_gap * gap) + (cfg.w_ent * ent) - (cfg.w_disp * disp)

    return QuerySignals(risk=risk, ent=ent, mnorm=mnorm, disp=disp, gap=gap)


# ----------------------------
# Main policy function
# ----------------------------

def choose_alpha(signals: QuerySignals, thresholds: Thresholds, cfg: PolicyCfg) -> float:
    """
    Adaptive alpha selection:
    - high-confidence => trust baseline more (alpha_high_conf)
    - low-confidence  => trust CE more (alpha_low_conf)
    - else => base_alpha
    """
    if not cfg.adaptive_blend:
        return cfg.blend.base_alpha

    ent_hi = thresholds.ent_high_conf_max if thresholds.ent_high_conf_max is not None else thresholds.ent_lock_max
    mnorm_hi = thresholds.mnorm_high_conf_min if thresholds.mnorm_high_conf_min is not None else thresholds.mnorm_lock_min

    # Low confidence thresholds are optional. If absent, we infer low confidence as "not high and quite flat".
    ent_lo = thresholds.ent_low_conf_min
    mnorm_lo = thresholds.mnorm_low_conf_max

    is_high = (signals.ent <= ent_hi) and (signals.mnorm >= mnorm_hi)
    if is_high:
        return cfg.blend.alpha_high_conf

    if ent_lo is not None and mnorm_lo is not None:
        is_low = (signals.ent >= ent_lo) and (signals.mnorm <= mnorm_lo)
        if is_low:
            return cfg.blend.alpha_low_conf
    else:
        # heuristic: very high entropy and very low margin implies low confidence
        if (signals.ent >= ent_hi * 1.10) and (signals.mnorm <= mnorm_hi * 0.60):
            return cfg.blend.alpha_low_conf

    return cfg.blend.base_alpha


def apply_policy_for_query(
    qid: str,
    base_scores_k1: ScoreDict,
    ce_scores: Optional[ScoreDict],
    thresholds: Thresholds,
    cfg: PolicyCfg,
) -> PolicyDecision:
    """
    Apply RA-SCE v4 to a single query.

    Parameters
    ----------
    qid : str
        query id (for diagnostics; not used in logic)
    base_scores_k1 : dict
        baseline scores over the full candidate pool (ideally k1 docs).
    ce_scores : dict or None
        cross-encoder scores for some subset of docids (often top ce_rerank_k from base).
        If None/empty, policy will skip CE actions gracefully.
    thresholds : Thresholds
        dataset-level thresholds computed in pass1.
    cfg : PolicyCfg
        policy config.

    Returns
    -------
    PolicyDecision with final ranked_docids and metadata.
    """
    ce_scores = ce_scores or {}

    # Signals computed on baseline head (k0/k1 head doesn't matter; should be same set of docids for head)
    signals = compute_query_signals(base_scores_k1, cfg)

    # (1) Risk-budgeted gating
    gated = int(signals.risk >= thresholds.tau_risk)
    if not gated:
        ranked = _ranked_docids(base_scores_k1)
        return PolicyDecision(
            ranked_docids=ranked,
            risk=signals.risk, ent=signals.ent, mnorm=signals.mnorm, disp=signals.disp, gap=signals.gap,
            gated=0,
            action_taken="none",
            skip_reason="risk_below_tau",
            alpha=1.0,
            overlap_topk=0,
            topn_locked=0,
            locked_n=0,
        )

    # (2) Confidence lock (suppress intervention if already stable)
    if cfg.confidence_lock and (signals.ent <= thresholds.ent_lock_max) and (signals.mnorm >= thresholds.mnorm_lock_min):
        ranked = _ranked_docids(base_scores_k1)
        return PolicyDecision(
            ranked_docids=ranked,
            risk=signals.risk, ent=signals.ent, mnorm=signals.mnorm, disp=signals.disp, gap=signals.gap,
            gated=1,
            action_taken="none",
            skip_reason="confidence_lock",
            alpha=1.0,
            overlap_topk=0,
            topn_locked=0,
            locked_n=0,
        )

    # If no CE scores available, we cannot refine; return baseline
    if not ce_scores:
        ranked = _ranked_docids(base_scores_k1)
        return PolicyDecision(
            ranked_docids=ranked,
            risk=signals.risk, ent=signals.ent, mnorm=signals.mnorm, disp=signals.disp, gap=signals.gap,
            gated=1,
            action_taken="none",
            skip_reason="no_ce_scores",
            alpha=1.0,
            overlap_topk=0,
            topn_locked=0,
            locked_n=0,
        )

    # Agreement is computed on a safe top-k: cannot exceed either list.
    # This avoids silent changes when ce_rerank_k < agree_k.
    ak = min(int(cfg.agree_k), len(ce_scores), len(base_scores_k1))
    overlap = _overlap_topk(base_scores_k1, ce_scores, ak)

    # (3) Agreement veto
    if cfg.agreement_gate and overlap < int(cfg.agree_min_overlap):
        ranked = _ranked_docids(base_scores_k1)
        return PolicyDecision(
            ranked_docids=ranked,
            risk=signals.risk, ent=signals.ent, mnorm=signals.mnorm, disp=signals.disp, gap=signals.gap,
            gated=1,
            action_taken="none",
            skip_reason="agreement_veto",
            alpha=1.0,
            overlap_topk=overlap,
            topn_locked=0,
            locked_n=0,
        )

    # (4) Controlled integration (z-normalize over CE-scored keys)
    keys = list(ce_scores.keys())
    zB = _z_normalize_on_keys(base_scores_k1, keys)
    zC = _z_normalize_on_keys(ce_scores, keys)

    alpha = choose_alpha(signals, thresholds, cfg)

    blended_scores: ScoreDict = {}
    for d in keys:
        blended_scores[d] = alpha * zB.get(d, 0.0) + (1.0 - alpha) * zC.get(d, 0.0)

    # Create final ranking:
    # - Sort CE-scored subset by blended score
    # - Append the rest of baseline docs (not CE-scored) in original baseline order
    blended_ranked = [d for d, _ in _topk_sorted(blended_scores, k=len(blended_scores))]
    base_ranked_full = _ranked_docids(base_scores_k1)
    rest = [d for d in base_ranked_full if d not in blended_scores]
    ranked = blended_ranked + rest

    # (5) Top-N lock: preserve top-N baseline docs if high confidence by mnorm threshold
    topn_locked = 0
    locked_n = 0
    if cfg.topn_lock and (signals.mnorm >= thresholds.mnorm_topn_lock_min):
        locked_n = max(0, int(cfg.topn))
        if locked_n > 0:
            locked_prefix = base_ranked_full[:locked_n]
            # Remove locked docs from ranked and prepend them
            ranked = locked_prefix + [d for d in ranked if d not in set(locked_prefix)]
            topn_locked = 1

    return PolicyDecision(
        ranked_docids=ranked,
        risk=signals.risk, ent=signals.ent, mnorm=signals.mnorm, disp=signals.disp, gap=signals.gap,
        gated=1,
        action_taken=f"{cfg.default_action}(alpha={alpha:.2f})",
        skip_reason="",
        alpha=alpha,
        overlap_topk=overlap,
        topn_locked=topn_locked,
        locked_n=locked_n,
    )


# ----------------------------
# Optional: threshold estimation helper
# ----------------------------

def compute_thresholds_from_query_signals(
    signals: List[QuerySignals],
    cfg: PolicyCfg,
) -> Thresholds:
    """
    Utility to compute quantile thresholds from pass1 signals.
    This keeps policy module self-contained. You can compute these elsewhere too.
    """
    if not signals:
        # safe fallbacks
        return Thresholds(
            tau_risk=float("inf"),
            ent_lock_max=float("-inf"),
            mnorm_lock_min=float("inf"),
            mnorm_topn_lock_min=float("inf"),
        )

    risks = sorted(s.risk for s in signals)
    ents = sorted(s.ent for s in signals)
    mns = sorted(s.mnorm for s in signals)

    def qtile(arr: List[float], q: float) -> float:
        q = min(max(q, 0.0), 1.0)
        idx = int(round(q * (len(arr) - 1)))
        return arr[idx]

    tau_risk = qtile(risks, 1.0 - cfg.rho)
    ent_lock_max = qtile(ents, cfg.ent_lock_quantile)
    mnorm_lock_min = qtile(mns, cfg.mnorm_lock_quantile)
    mnorm_topn_lock_min = qtile(mns, cfg.topn_lock_quantile)

    return Thresholds(
        tau_risk=tau_risk,
        ent_lock_max=ent_lock_max,
        mnorm_lock_min=mnorm_lock_min,
        mnorm_topn_lock_min=mnorm_topn_lock_min,
    )
