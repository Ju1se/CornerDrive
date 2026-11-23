"""Read unit files and turn the per-update logs into tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .cifar_golden import ROLES
from .detectors import FRAUD, HONEST, NOISE, RARITY, UNAUDITED, VERDICTS, VerdictPolicy, dual_verdict

TRUTHS = ("HONEST", "RARITY", "NOISE", "FRAUD")
# Update types as the tables name them: the three benign roles and the three
# fraud families.
TYPES = ("honest", "rarity", "noise", "sign_flip", "label_flip", "corner_harm")
TYPE_LABELS = {
    "honest": "Honest",
    "rarity": "Rarity (rare-rich honest)",
    "noise": "Noise",
    "sign_flip": "Fraud: sign-flip",
    "label_flip": "Fraud: label-flip",
    "corner_harm": "Fraud: corner-harm",
}


def load_unit(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        unit = {key: payload[key] for key in payload.files if key != "meta"}
        unit["meta"] = json.loads(str(payload["meta"]))
    unit["path"] = str(path)
    return unit


def load_units(folder: Path, **where: Any) -> list[dict[str, Any]]:
    """All units of a folder whose meta matches ``where`` (value or collection)."""
    units = []
    for path in sorted(Path(folder).glob("unit_*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        unit = load_unit(path)
        keep = True
        for key, wanted in where.items():
            value = unit["meta"].get(key)
            allowed = wanted if isinstance(wanted, (list, tuple, set)) else (wanted,)
            keep &= value in allowed
        if keep:
            units.append(unit)
    return units


def type_of(unit: dict[str, Any]) -> np.ndarray:
    """Type name of every logged update, [T, K]."""
    return np.array(ROLES, dtype=object)[unit["u_role"]]


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    """P(score of a positive > score of a negative), ties counted half."""
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if len(positive) == 0 or len(negative) == 0:
        return float("nan")
    scores = np.concatenate([positive, negative])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):                       # average the ranks of ties
        stop = start
        while stop + 1 < len(scores) and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        if stop > start:
            ranks[order[start:stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    rank_sum = ranks[:len(positive)].sum()
    return float((rank_sum - len(positive) * (len(positive) + 1) / 2.0)
                 / (len(positive) * len(negative)))


def drifts(unit: dict[str, Any], mode: str = "relative", step: float = 1.0,
           space: str = "loss") -> tuple[np.ndarray, np.ndarray]:
    """(drift on the golden normal set, drift on the golden rare set), [T, K].

    space "loss": cross-entropy drift of the replica; "embedding": drift of
    the prototype read-out on the replica's own embedding.
    """
    if space == "embedding":
        dm, dc = unit["u_e_main"], unit["u_e_corner"]
        bm, bc = unit["r_base_e_main"], unit["r_base_e_corner"]
    else:
        suffix = "" if step == 1.0 else f"_eta{step:g}"
        dm, dc = unit[f"u_d_main{suffix}"], unit[f"u_d_corner{suffix}"]
        bm, bc = unit["r_base_main"], unit["r_base_corner"]
    if mode == "relative":
        return dm / bm[:, None], dc / bc[:, None]
    if mode == "absolute":
        return dm, dc
    raise ValueError(mode)


def exhaustive_verdicts(unit: dict[str, Any], policy: VerdictPolicy, mode: str,
                        step: float = 1.0, nominated: np.ndarray | None = None,
                        space: str = "loss") -> np.ndarray:
    """Verdict the dual test gives every logged update when all are audited."""
    dm, dc = drifts(unit, mode, step, space)
    if nominated is None:
        nominated = np.zeros(dm.shape, dtype=bool)
    flat = dual_verdict(dm.ravel(), dc.ravel(), nominated.ravel(), policy)
    return flat.reshape(dm.shape)


def confusion(types: np.ndarray, verdicts: np.ndarray) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for name in TYPES:
        mask = types == name
        table[name] = {v: int((verdicts[mask] == v).sum()) for v in VERDICTS}
        table[name]["total"] = int(mask.sum())
    return table


def rarity_discovery(types: np.ndarray, verdicts: np.ndarray) -> dict[str, float]:
    """Recognition (recall), precision and false-rarity counts of the Rarity verdict."""
    is_rarity = types == "rarity"
    said_rarity = verdicts == RARITY
    true_pos = int((is_rarity & said_rarity).sum())
    fraud = np.isin(types, ("sign_flip", "label_flip", "corner_harm"))
    return {
        "rarity_total": int(is_rarity.sum()),
        "said_rarity": int(said_rarity.sum()),
        "recognition": true_pos / max(int(is_rarity.sum()), 1),
        "precision": true_pos / max(int(said_rarity.sum()), 1),
        "fraud_named_rarity": int((fraud & said_rarity).sum()),
        "corner_harm_named_rarity": int(((types == "corner_harm") & said_rarity).sum()),
        "honest_named_rarity": int(((types == "honest") & said_rarity).sum()),
    }


def weight_share(unit: dict[str, Any], rounds: slice = slice(None)) -> dict[str, float]:
    """Mean aggregation share of every update type, relative to an equal
    share (1.0 = the weight FedAvg would give), and the fraction kept at all."""
    types = type_of(unit)[rounds]
    weight = unit["u_weight"][rounds]
    relative = weight * weight.shape[1]
    out: dict[str, float] = {}
    for name in TYPES:
        mask = types == name
        if mask.any():
            out[f"share_{name}"] = float(relative[mask].mean())
            out[f"kept_{name}"] = float((weight[mask] > 0).mean())
    return out


def tail_accuracy(unit: dict[str, Any], last: int = 10) -> dict[str, float]:
    return {
        "acc_normal": float(unit["r_acc_normal"][-last:].mean()),
        "acc_rare": float(unit["r_acc_rare"][-last:].mean()),
        "acc_all": float(unit["r_acc_all"][-last:].mean()),
    }


def mean_ci(values: Iterable[float]) -> tuple[float, float]:
    """Mean and half-width of the 95% t-interval over seeds."""
    values = np.asarray(list(values), dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return float(values[0]), float("nan")
    t95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
           9: 2.306, 10: 2.262}.get(len(values), 1.96)
    return float(values.mean()), float(t95 * values.std(ddof=1) / np.sqrt(len(values)))


__all__ = [
    "FRAUD", "HONEST", "NOISE", "RARITY", "UNAUDITED", "TRUTHS", "TYPES", "TYPE_LABELS",
    "auroc", "confusion", "drifts", "exhaustive_verdicts", "load_unit", "load_units",
    "mean_ci", "rarity_discovery", "tail_accuracy", "type_of", "weight_share",
]
