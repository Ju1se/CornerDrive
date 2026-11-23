"""CornerDrive L1 routing signals and the L2 dual-dataset verdict.

L1 reproduces the adaptive multi-signal profile of thesis Table 4.1 with the
same statistics as ``l1_linear_defense.scoring`` (geometric-median cosine
deviation, log-norm MAD score, top-k sign disagreement), computed in torch on
the GPU. L1 only decides which updates L2 sees.

L2 is the ordered rule of thesis Table 4.2 on the two loss drifts measured on
the golden normal set (D_main) and the golden rare set (D_corner).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

FRAUD, RARITY, HONEST, NOISE, UNAUDITED = "FRAUD", "RARITY", "HONEST", "NOISE", "UNAUDITED"
VERDICTS = (FRAUD, RARITY, HONEST, NOISE, UNAUDITED)
REASONS = ("bypass", "cosine_screening", "norm_mad_screening", "sign_screening",
           "probabilistic_recheck", "exhaustive")


# --------------------------------------------------------------------------
# L1
# --------------------------------------------------------------------------
def geometric_median(
    updates: torch.Tensor, max_iter: int = 100, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weiszfeld iteration from the mean. Returns (median [P], weights [C]),
    the weights being the normalised inverse distances at the solution."""
    point = updates.mean(dim=0)
    inverse = torch.ones(updates.shape[0], device=updates.device)
    for _ in range(max_iter):
        inverse = 1.0 / (updates - point).norm(dim=1).clamp_min(eps)
        new_point = (updates * inverse[:, None]).sum(dim=0) / inverse.sum()
        change = float((new_point - point).norm())
        point = new_point
        if change < eps:
            break
    return point, inverse / inverse.sum()


@dataclass(frozen=True)
class L1Signals:
    cosine_deviation: np.ndarray
    norm_mad: np.ndarray
    sign_disagreement: np.ndarray
    norm: np.ndarray


def l1_signals(
    updates: torch.Tensor, sign_topk_ratio: float = 0.10, eps: float = 1e-12
) -> L1Signals:
    reference, _ = geometric_median(updates)
    norms = updates.norm(dim=1)
    ref_norm = reference.norm()
    cosine = (updates @ reference) / (norms * ref_norm).clamp_min(1e-10)
    cosine = torch.where((norms < 1e-10) | (ref_norm < 1e-10), torch.zeros_like(cosine), cosine)

    log_norm = torch.log(norms + eps)
    deviation = (log_norm - torch.quantile(log_norm, 0.5)).abs()
    mad = torch.quantile(deviation, 0.5)
    norm_mad = deviation / mad if float(mad) > eps else torch.zeros_like(deviation)

    nonzero = int((reference.abs() > eps).sum())
    if nonzero == 0:
        sign_dis = torch.zeros_like(norms)
    else:
        topk = max(1, min(nonzero, int(math.ceil(nonzero * sign_topk_ratio))))
        top = torch.topk(reference.abs(), topk).indices
        sign_dis = (torch.sign(updates[:, top]) != torch.sign(reference[top])).float().mean(dim=1)

    return L1Signals(
        cosine_deviation=(1.0 - cosine).cpu().numpy().astype(np.float64),
        norm_mad=norm_mad.cpu().numpy().astype(np.float64),
        sign_disagreement=sign_dis.cpu().numpy().astype(np.float64),
        norm=norms.cpu().numpy().astype(np.float64),
    )


@dataclass(frozen=True)
class RoutingPolicy:
    """Thesis Table 4.1, adaptive multi-signal profile."""

    tau_cos: float = 0.60
    tau_norm_mad: float = 2.5
    tau_sign: float = 0.55
    p_recheck: float = 0.25
    exhaustive: bool = False


def route(
    signals: L1Signals, policy: RoutingPolicy, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (routed, nominated, reason).

    ``nominated`` is the thesis's s_i: the update was sent to L2 because one
    of the deviation signals fired. A recheck or an exhaustive audit makes an
    update visible without nominating it.
    """
    count = len(signals.norm)
    draws = rng.random(count)
    routed = np.zeros(count, dtype=bool)
    nominated = np.zeros(count, dtype=bool)
    reasons: list[str] = []
    for i in range(count):
        if signals.cosine_deviation[i] > policy.tau_cos:
            reason = "cosine_screening"
        elif signals.norm_mad[i] > policy.tau_norm_mad:
            reason = "norm_mad_screening"
        elif signals.sign_disagreement[i] > policy.tau_sign:
            reason = "sign_screening"
        else:
            reason = "bypass"
        nominated[i] = reason != "bypass"
        if reason == "bypass":
            if policy.exhaustive:
                reason = "exhaustive"
            elif draws[i] < policy.p_recheck:
                reason = "probabilistic_recheck"
        routed[i] = reason != "bypass"
        reasons.append(reason)
    return routed, nominated, reasons


# --------------------------------------------------------------------------
# L2
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VerdictPolicy:
    """Thresholds of the dual-dataset test.

    theta_tol          main non-harm tolerance      (drift on D_main above it -> Fraud)
    theta_corner_harm  corner non-harm tolerance    (drift on D_corner above it -> Fraud)
    theta_rare         corner benefit, negative     (drift on D_corner at or below it -> Rarity)
    theta_util         utility floor for nominated neutral updates (below it -> Noise)
    mode               "dual" | "main_only" | "corner_only"
    """

    theta_tol: float
    theta_rare: float
    theta_corner_harm: float
    theta_util: float
    mode: str = "dual"


def dual_verdict(
    drift_main: np.ndarray,
    drift_corner: np.ndarray,
    nominated: np.ndarray,
    policy: VerdictPolicy,
) -> np.ndarray:
    """Ordered rule of thesis Table 4.2 (and its single-objective reductions,
    Section 4.6). Returns an array of verdict strings."""
    dm = np.asarray(drift_main, dtype=np.float64)
    dc = np.asarray(drift_corner, dtype=np.float64)
    nominated = np.asarray(nominated, dtype=bool)
    out = np.empty(len(dm), dtype=object)
    for i in range(len(dm)):
        if policy.mode == "dual":
            if dm[i] > policy.theta_tol:
                verdict = FRAUD                      # main-task harm
            elif dc[i] > policy.theta_corner_harm:
                verdict = FRAUD                      # corner-case harm
            elif dc[i] <= policy.theta_rare:
                verdict = RARITY                     # beneficial rarity
            else:
                utility = max(-dm[i], 0.0) + max(-dc[i], 0.0)
                verdict = NOISE if (nominated[i] and utility < policy.theta_util) else HONEST
        elif policy.mode == "main_only":
            if dm[i] > policy.theta_tol:
                verdict = FRAUD
            else:
                utility = max(-dm[i], 0.0)
                verdict = NOISE if (nominated[i] and utility < policy.theta_util) else HONEST
        elif policy.mode == "corner_only":
            if dc[i] > policy.theta_corner_harm:
                verdict = FRAUD
            elif dc[i] <= policy.theta_rare:
                verdict = RARITY
            else:
                utility = max(-dc[i], 0.0)
                verdict = NOISE if (nominated[i] and utility < policy.theta_util) else HONEST
        else:
            raise ValueError(f"unknown verdict mode {policy.mode!r}")
        out[i] = verdict
    return out


def prototype_readout(
    features: torch.Tensor,
    labels: torch.Tensor,
    support: torch.Tensor,
    num_classes: int,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Embedding-space read-out of C models on the same golden draw.

    Class prototypes are the mean unit-norm embedding of the ``support``
    images; the remaining (query) images are scored by cosine similarity to
    the prototypes. Returns per-example query loss and correctness, [C, Nq].
    The read-out ignores the linear head, so it measures what an update did to
    the representation and not what it did to the output prior.
    """
    unit = F.normalize(features, dim=2)
    onehot = F.one_hot(labels, num_classes).float() * support[:, None].float()     # [N, K]
    proto = torch.einsum("cnf,nk->ckf", unit, onehot) / onehot.sum(dim=0).clamp_min(1.0)[None, :, None]
    proto = F.normalize(proto, dim=2)
    query = ~support
    logits = torch.einsum("cnf,ckf->cnk", unit[:, query], proto) / temperature
    target = labels[query].unsqueeze(0).expand(features.shape[0], -1)
    loss = F.cross_entropy(logits.reshape(-1, num_classes), target.reshape(-1),
                           reduction="none").view(features.shape[0], -1)
    return loss, logits.argmax(dim=2) == target


def cosine_to(updates: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(updates, reference.unsqueeze(0), dim=1)
