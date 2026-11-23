"""Federated MLP head on a frozen embedding model (model B).

The embedding model (an ImageNet ResNet-50, see
``scripts/extract_cifar10_embeddings.py``) is a fixed public encoder: nobody
trains it. Clients and the server exchange only the head

    linear(D -> 128)  ReLU  linear(128 -> 10)

whose 128-d hidden layer plays the role of the model's own embedding in the
prototype read-out. As in ``batched_cnn``, a stack [C, P] of flat parameter
vectors is run for C clients at once.
"""

from __future__ import annotations

import math

import torch

NUM_CLASSES = 10
HIDDEN = 128


def layers(input_dim: int, hidden: int = HIDDEN) -> tuple[tuple[str, tuple[int, ...]], ...]:
    return (
        ("fc1.weight", (hidden, input_dim)), ("fc1.bias", (hidden,)),
        ("fc2.weight", (NUM_CLASSES, hidden)), ("fc2.bias", (NUM_CLASSES,)),
    )


def num_params(input_dim: int, hidden: int = HIDDEN) -> int:
    return sum(math.prod(shape) for _name, shape in layers(input_dim, hidden))


def init_params(seed: int, device, input_dim: int, hidden: int = HIDDEN) -> torch.Tensor:
    """torch.nn.Linear's default initialisation, U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    gen = torch.Generator().manual_seed(int(seed))
    parts = []
    for name, shape in layers(input_dim, hidden):
        fan_in = input_dim if name.startswith("fc1") else hidden
        bound = 1.0 / math.sqrt(fan_in)
        parts.append(((torch.rand(shape, generator=gen) * 2 - 1) * bound).reshape(-1))
    return torch.cat(parts).to(device)


def forward(flat: torch.Tensor, x: torch.Tensor, return_features: bool = False,
            *, input_dim: int, hidden: int = HIDDEN):
    """Logits [C, B, 10]. ``x`` is [C, B, D] (own batches) or [B, D] (shared)."""
    count = flat.shape[0]
    a = hidden * input_dim
    w1 = flat[:, :a].view(count, hidden, input_dim)
    b1 = flat[:, a:a + hidden]
    w2 = flat[:, a + hidden:a + hidden + NUM_CLASSES * hidden].view(count, NUM_CLASSES, hidden)
    b2 = flat[:, a + hidden + NUM_CLASSES * hidden:]
    if x.dim() == 3:
        pre = torch.baddbmm(b1[:, None, :], x, w1.transpose(1, 2))
    else:
        pre = (w1.reshape(count * hidden, input_dim) @ x.t()).view(count, hidden, -1)
        pre = pre.transpose(1, 2) + b1[:, None, :]
    features = torch.relu(pre)
    logits = torch.baddbmm(b2[:, None, :], features, w2.transpose(1, 2))
    if return_features:
        return logits, features
    return logits


def prepare(x: torch.Tensor, flips: torch.Tensor | None = None) -> torch.Tensor:
    """Embeddings are already standardised; there is no augmentation."""
    return x
