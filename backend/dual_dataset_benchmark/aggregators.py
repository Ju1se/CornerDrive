"""Baseline aggregation rules on client updates.

Each rule returns ``(delta [P], weight [C])``: the aggregated update and the
effective share every client received (non-negative, summing to one, zero for
a rejected client). The share is what the survival / retention metrics read.

    fedavg    mean                                         McMahan et al. 2017
    geomed    geometric median                             Pillutla et al. 2022
    krum      Multi-Krum                                   Blanchard et al. 2017
    fltrust   ReLU-cosine trust against the server update  Cao et al. 2021
    zeno      rank by validation descent, drop the b worst Xie et al. 2019
    zenopp    accept on a first-order validation estimate  Xie et al. 2020
    fedval    per-class validation score, formula (6)      Valadi et al. 2023
"""

from __future__ import annotations

import torch

from .detectors import geometric_median


def _uniform(mask: torch.Tensor) -> torch.Tensor:
    weight = mask.float()
    total = weight.sum()
    return weight / total if float(total) > 0 else weight


def fedavg(updates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.full((updates.shape[0],), 1.0 / updates.shape[0], device=updates.device)
    return updates.mean(dim=0), weight


def geomed(updates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return geometric_median(updates)


def krum_scores(updates: torch.Tensor, byzantine: int) -> torch.Tensor:
    """Sum of squared distances to the n - f - 2 nearest neighbours."""
    count = updates.shape[0]
    dist2 = torch.cdist(updates, updates) ** 2
    dist2.fill_diagonal_(float("inf"))
    neighbours = max(1, count - byzantine - 2)
    return torch.topk(dist2, neighbours, dim=1, largest=False).values.sum(dim=1)


def multi_krum(updates: torch.Tensor, byzantine: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Keeps the n - f - 2 best-scored updates, as the repository's
    ``_multi_krum_indices`` does, and averages them."""
    count = updates.shape[0]
    scores = krum_scores(updates, byzantine)
    keep = torch.topk(scores, max(1, count - byzantine - 2), largest=False).indices
    mask = torch.zeros(count, dtype=torch.bool, device=updates.device)
    mask[keep] = True
    weight = _uniform(mask)
    return (updates * weight[:, None]).sum(dim=0), weight


def fltrust(updates: torch.Tensor, root: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Trust = ReLU(cos(update, server update)); every update is rescaled to
    the server update's norm before the trust-weighted mean."""
    root_norm = root.norm().clamp_min(1e-12)
    norms = updates.norm(dim=1).clamp_min(1e-12)
    trust = torch.relu((updates @ root) / (norms * root_norm))
    if float(trust.sum()) <= 0:
        return torch.zeros_like(root), torch.zeros_like(trust)
    weight = trust / trust.sum()
    delta = ((weight * root_norm / norms)[:, None] * updates).sum(dim=0)
    return delta, weight


def zeno_scores(base_loss: torch.Tensor, client_loss: torch.Tensor,
                updates: torch.Tensor, rho: float) -> torch.Tensor:
    """Validation descent minus a magnitude penalty (losses are mean losses on
    the pooled validation draw)."""
    return base_loss - client_loss - rho * updates.pow(2).sum(dim=1)


def zeno(updates: torch.Tensor, scores: torch.Tensor, byzantine: int
         ) -> tuple[torch.Tensor, torch.Tensor]:
    count = updates.shape[0]
    keep = torch.topk(scores, max(1, count - byzantine), largest=True).indices
    mask = torch.zeros(count, dtype=torch.bool, device=updates.device)
    mask[keep] = True
    weight = _uniform(mask)
    return (updates * weight[:, None]).sum(dim=0), weight


def zenopp_scores(updates: torch.Tensor, validation_gradient: torch.Tensor,
                  rho: float) -> torch.Tensor:
    """First-order estimate of the validation descent, -<grad, u> - rho |u|^2."""
    return -(updates @ validation_gradient) - rho * updates.pow(2).sum(dim=1)


def zenopp(updates: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Accept an update iff its first-order score is positive; accepted
    updates are brought to the round's median norm and averaged. Nothing
    accepted means no step."""
    accept = scores > 0
    weight = _uniform(accept)
    if not bool(accept.any()):
        return torch.zeros_like(updates[0]), weight
    norms = updates.norm(dim=1).clamp_min(1e-12)
    scale = norms.median() / norms
    return ((weight * scale)[:, None] * updates).sum(dim=0), weight


def fedval_scores(
    class_loss: torch.Tensor,
    mean_loss: torch.Tensor,
    s2: float,
    s1: float = 3.0,
    s1_avg: float = 5.0,
    c_const: float = 3.0,
) -> torch.Tensor:
    """FedVal formula (6).

    class_loss [C, K]  validation loss of client model d on label k
    mean_loss  [C]     validation loss of client model d on the whole set

        S_d = sum_k ( max(1, (Lbar_k / Lbar_avg)^s2) * s1 * div_kd / MAD_k + C s1 )
              + s1_avg * div_avg,d / MAD_avg + C s1_avg
        div_kd = Lbar_k - L_kd, Lbar and MAD taken over the round's clients.
    """
    eps = 1e-12
    overall = mean_loss.mean()
    mad_avg = (mean_loss - overall).abs().mean().clamp_min(eps)
    score = s1_avg * (overall - mean_loss) / mad_avg + c_const * s1_avg
    label_mean = class_loss.mean(dim=0)                                   # [K]
    mad = (class_loss - label_mean).abs().mean(dim=0).clamp_min(eps)      # [K]
    boost = torch.clamp((label_mean / overall.clamp_min(eps)) ** s2, min=1.0, max=1e6)
    per_label = boost * s1 * (label_mean - class_loss) / mad + c_const * s1
    return score + per_label.sum(dim=1)


def fedval_s2_candidates(s2: float) -> list[float]:
    """The paper checks {s2, s2 +/- 0.5, s2 +/- 5} every round."""
    return sorted({max(0.0, v) for v in (s2, s2 + 0.5, s2 - 0.5, s2 + 5.0, s2 - 5.0)})


def fedval_weights(scores: torch.Tensor) -> torch.Tensor:
    weight = scores.clamp_min(0.0)
    total = weight.sum()
    if float(total) <= 0:
        return torch.full_like(weight, 1.0 / weight.numel())
    return weight / total
