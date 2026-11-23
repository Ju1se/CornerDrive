"""Model backends and the loops that are the same for both of them.

A backend bundles what differs between the two federated models:

    cnn        a small CNN trained from scratch on the raw images
    r50head    an MLP head on frozen ResNet-50 embeddings

Everything else (local SGD for C clients at once, per-example losses of C
replicas on a shared golden draw, the validation gradient) is written once.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch
import torch.nn.functional as F

from . import batched_cnn, batched_head

NUM_CLASSES = 10


@dataclass(frozen=True)
class Backend:
    name: str
    num_params: int
    init_params: Callable        # (seed, device) -> [P]
    forward: Callable            # (flat [C, P], inputs, return_features) -> logits (, features)
    prepare: Callable            # (raw pool rows, flip mask or None) -> inputs
    audit_chunk: int
    eval_chunk: int


CNN = Backend(
    name="cnn",
    num_params=batched_cnn.NUM_PARAMS,
    init_params=batched_cnn.init_params,
    forward=batched_cnn.forward,
    prepare=batched_cnn.prepare,
    audit_chunk=250,
    eval_chunk=2500,
)


def head_backend(input_dim: int, name: str = "r50head") -> Backend:
    return Backend(
        name=name,
        num_params=batched_head.num_params(input_dim),
        init_params=partial(batched_head.init_params, input_dim=input_dim),
        forward=partial(batched_head.forward, input_dim=input_dim),
        prepare=batched_head.prepare,
        audit_chunk=1000,
        eval_chunk=5000,
    )


def local_train(
    backend: Backend,
    start: torch.Tensor,
    x: torch.Tensor,
    shard_index: torch.Tensor,
    shard_labels: torch.Tensor,
    positions: torch.Tensor,
    flips: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """Plain local SGD for C clients at once; returns the updates [C, P].

    start         [P]         the global model
    shard_index   [C, n]      pool rows of every client's examples
    shard_labels  [C, n]      training labels (already poisoned where relevant)
    positions     [S, C, B]   which shard positions form each step's batch
    flips         [S, C, B]   augmentation mask (used by the image backend)
    """
    count = shard_index.shape[0]
    current = start.unsqueeze(0).repeat(count, 1)
    for step in range(positions.shape[0]):
        current.requires_grad_(True)
        picked = shard_index.gather(1, positions[step])
        xb = backend.prepare(x[picked], flips[step])
        yb = shard_labels.gather(1, positions[step])
        logits = backend.forward(current, xb)
        loss = F.cross_entropy(
            logits.reshape(-1, NUM_CLASSES), yb.reshape(-1), reduction="none"
        ).view(count, -1).mean(dim=1).sum()
        (grad,) = torch.autograd.grad(loss, current)
        current = (current - lr * grad).detach()
    return current - start.unsqueeze(0)


@torch.no_grad()
def example_losses(
    backend: Backend,
    flat: torch.Tensor,
    x: torch.Tensor,
    index: torch.Tensor,
    labels: torch.Tensor,
    chunk: int,
    with_features: bool = False,
):
    """Per-example cross-entropy [C, N] and correctness [C, N] of C models on
    the same N examples; with ``with_features`` also their embeddings."""
    count = flat.shape[0]
    losses, correct, feats = [], [], []
    for start in range(0, index.numel(), chunk):
        sel = index[start:start + chunk]
        y = labels[start:start + chunk]
        logits, features = backend.forward(flat, backend.prepare(x[sel]), True)
        target = y.unsqueeze(0).expand(count, -1)
        losses.append(F.cross_entropy(
            logits.reshape(-1, NUM_CLASSES), target.reshape(-1), reduction="none"
        ).view(count, -1))
        correct.append(logits.argmax(dim=2) == target)
        if with_features:
            feats.append(features)
    if with_features:
        return torch.cat(losses, dim=1), torch.cat(correct, dim=1), torch.cat(feats, dim=1)
    return torch.cat(losses, dim=1), torch.cat(correct, dim=1)


def loss_gradient(
    backend: Backend,
    flat_one: torch.Tensor,
    x: torch.Tensor,
    index: torch.Tensor,
    labels: torch.Tensor,
    chunk: int = 500,
) -> torch.Tensor:
    """Gradient [P] of the mean loss of one model on N examples."""
    param = flat_one.detach().clone().unsqueeze(0).requires_grad_(True)
    total = torch.zeros((), device=flat_one.device)
    for start in range(0, index.numel(), chunk):
        sel = index[start:start + chunk]
        logits = backend.forward(param, backend.prepare(x[sel]))
        total = total + F.cross_entropy(logits[0], labels[start:start + chunk], reduction="sum")
    (grad,) = torch.autograd.grad(total / index.numel(), param)
    return grad[0].detach()
