"""One small CNN, many clients, one grouped-convolution pass (model A).

Every model is a flat vector of P parameters. A stack ``flat`` of shape
[C, P] holds C models; ``forward`` runs all of them at once by folding the
client axis into the channel axis (``groups=C``), so twenty clients train in
a single cuDNN call. GroupNorm keeps the models independent of each other and
of the batch composition.

    conv3x3(3->32)  GN(4)  ReLU  maxpool
    conv3x3(32->64) GN(8)  ReLU  maxpool
    conv3x3(64->128) GN(8) ReLU  global-average-pool
    linear(128->10)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

NUM_CLASSES = 10
LAYERS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("conv1.weight", (32, 3, 3, 3)), ("conv1.bias", (32,)),
    ("gn1.weight", (32,)), ("gn1.bias", (32,)),
    ("conv2.weight", (64, 32, 3, 3)), ("conv2.bias", (64,)),
    ("gn2.weight", (64,)), ("gn2.bias", (64,)),
    ("conv3.weight", (128, 64, 3, 3)), ("conv3.bias", (128,)),
    ("gn3.weight", (128,)), ("gn3.bias", (128,)),
    ("fc.weight", (NUM_CLASSES, 128)), ("fc.bias", (NUM_CLASSES,)),
)
GN_GROUPS = (4, 8, 8)
POOL_AFTER = (True, True, False)

_SIZES = [math.prod(shape) for _name, shape in LAYERS]
NUM_PARAMS = sum(_SIZES)
_OFFSETS = [sum(_SIZES[:i]) for i in range(len(_SIZES))]

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def init_params(seed: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """He-normal convolutions, default-uniform linear head, unit GroupNorm."""
    gen = torch.Generator().manual_seed(int(seed))
    parts = []
    for name, shape in LAYERS:
        if name.startswith("conv") and name.endswith("weight"):
            fan_in = shape[1] * shape[2] * shape[3]
            parts.append(torch.randn(shape, generator=gen) * math.sqrt(2.0 / fan_in))
        elif name == "fc.weight":
            bound = 1.0 / math.sqrt(shape[1])
            parts.append((torch.rand(shape, generator=gen) * 2 - 1) * bound)
        elif name.startswith("gn") and name.endswith("weight"):
            parts.append(torch.ones(shape))
        else:
            parts.append(torch.zeros(shape))
    return torch.cat([p.reshape(-1) for p in parts]).to(device)


def unflatten(flat: torch.Tensor) -> dict[str, torch.Tensor]:
    """[C, P] -> {name: [C, *shape]} (views, no copy)."""
    count = flat.shape[0]
    return {
        name: flat[:, off:off + size].reshape(count, *shape)
        for (name, shape), off, size in zip(LAYERS, _OFFSETS, _SIZES)
    }


def normalize(x_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [..., 3, 32, 32] -> standardised float32."""
    mean = torch.tensor(CIFAR_MEAN, device=x_u8.device).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD, device=x_u8.device).view(3, 1, 1)
    return (x_u8.float() / 255.0 - mean) / std


def forward(flat: torch.Tensor, x: torch.Tensor, return_features: bool = False):
    """Logits [C, B, 10]; with ``return_features`` also the 128-d embedding
    [C, B, 128] that feeds the linear head.

    ``x`` is either [C, B, 3, 32, 32] (each model sees its own batch) or
    [B, 3, 32, 32] (all models see the same batch).
    """
    count = flat.shape[0]
    params = unflatten(flat)
    if x.dim() == 5:
        batch = x.shape[1]
        h = x.permute(1, 0, 2, 3, 4).reshape(batch, count * 3, 32, 32)
    else:
        batch = x.shape[0]
        h = x.unsqueeze(1).expand(batch, count, 3, 32, 32).reshape(batch, count * 3, 32, 32)
    for layer, (groups, pool) in enumerate(zip(GN_GROUPS, POOL_AFTER), start=1):
        weight = params[f"conv{layer}.weight"]
        out_channels = weight.shape[1]
        h = F.conv2d(
            h,
            weight.reshape(count * out_channels, *weight.shape[2:]),
            params[f"conv{layer}.bias"].reshape(-1),
            padding=1,
            groups=count,
        )
        h = F.group_norm(
            h,
            count * groups,
            params[f"gn{layer}.weight"].reshape(-1),
            params[f"gn{layer}.bias"].reshape(-1),
        )
        h = F.relu(h)
        if pool:
            h = F.max_pool2d(h, 2)
    features = h.mean(dim=(2, 3)).view(batch, count, -1)
    logits = torch.einsum("bcf,ckf->bck", features, params["fc.weight"])
    logits = (logits + params["fc.bias"].unsqueeze(0)).permute(1, 0, 2)
    if return_features:
        return logits, features.permute(1, 0, 2)
    return logits


def prepare(x_u8: torch.Tensor, flips: torch.Tensor | None = None) -> torch.Tensor:
    """Raw pool rows -> model input; ``flips`` is the horizontal-flip mask of a
    training batch."""
    x = normalize(x_u8)
    if flips is not None:
        x = torch.where(flips[..., None, None, None], x.flip(-1), x)
    return x
