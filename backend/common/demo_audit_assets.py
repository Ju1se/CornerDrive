"""
Shared demo assets for the placeholder L2 audit pipeline.

These helpers keep the demo model, datasets, and gradient dimensionality
stable across the worker and local demo-data generator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEMO_MODEL_SEED = 1337
DEMO_MAIN_DATASET_SEED = 1338
DEMO_CORNER_DATASET_SEED = 1339
DEMO_CLASS_CENTER_SEED = 1340
DEMO_CLASS_STYLE_SEED = 1341

DEMO_INPUT_DIM = 100
DEMO_HIDDEN_DIM = 32
DEMO_OUTPUT_DIM = 10

DEMO_MAIN_DATASET_SIZE = 500
DEMO_CORNER_DATASET_SIZE = 100

DEMO_GRADIENT_DIM = (
    DEMO_INPUT_DIM * DEMO_HIDDEN_DIM
    + DEMO_HIDDEN_DIM
    + DEMO_HIDDEN_DIM * DEMO_OUTPUT_DIM
    + DEMO_OUTPUT_DIM
)


class SimpleMLP(nn.Module):
    """Small deterministic MLP used by the placeholder auditor."""

    def __init__(
        self,
        input_dim: int = DEMO_INPUT_DIM,
        hidden_dim: int = DEMO_HIDDEN_DIM,
        output_dim: int = DEMO_OUTPUT_DIM,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class PlaceholderDataset(torch.utils.data.Dataset):
    """Structured synthetic dataset used for the demo audit pipeline."""

    def __init__(
        self,
        size: int,
        input_dim: int = DEMO_INPUT_DIM,
        num_classes: int = DEMO_OUTPUT_DIM,
        seed: int = DEMO_MAIN_DATASET_SEED,
        variant: str = "main",
        class_centers: torch.Tensor | None = None,
        class_styles: torch.Tensor | None = None,
    ):
        generator = torch.Generator().manual_seed(seed)

        if class_centers is None:
            centers_generator = torch.Generator().manual_seed(DEMO_CLASS_CENTER_SEED)
            class_centers = F.normalize(
                torch.randn(num_classes, input_dim, generator=centers_generator),
                dim=1,
            ) * 2.5

        if class_styles is None:
            styles_generator = torch.Generator().manual_seed(DEMO_CLASS_STYLE_SEED)
            class_styles = F.normalize(
                torch.randn(num_classes, input_dim, generator=styles_generator),
                dim=1,
            )

        if variant == "corner":
            rare_class_ids = torch.tensor([1, 3, 7, 9], dtype=torch.long)
            targets = rare_class_ids[torch.arange(size) % len(rare_class_ids)]
            targets = targets[torch.randperm(size, generator=generator)]
        else:
            targets = torch.arange(size, dtype=torch.long) % num_classes
            targets = targets[torch.randperm(size, generator=generator)]

        if variant == "corner":
            neighbor_targets = (targets + 1) % num_classes
            high_blend_mask = torch.isin(targets, torch.tensor([1, 7], dtype=torch.long)).unsqueeze(1)
            blend = torch.where(
                high_blend_mask,
                torch.full((size, 1), 0.42),
                torch.full((size, 1), 0.30),
            )
            noise = 0.28 * torch.randn(size, input_dim, generator=generator)
            data = (
                (1.0 - blend) * class_centers[targets]
                + blend * class_centers[neighbor_targets]
                + 0.55 * class_styles[targets]
                - 0.12 * class_styles[neighbor_targets]
                + noise
            )
        else:
            noise = 0.20 * torch.randn(size, input_dim, generator=generator)
            data = (
                class_centers[targets]
                + 0.18 * class_styles[targets]
                + noise
            )

        self.data = data.float()
        self.targets = targets.long()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def build_demo_audit_bundle():
    """Build the deterministic placeholder model and datasets."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(DEMO_MODEL_SEED)
        model = SimpleMLP()

    centers_generator = torch.Generator().manual_seed(DEMO_CLASS_CENTER_SEED)
    class_centers = F.normalize(
        torch.randn(DEMO_OUTPUT_DIM, DEMO_INPUT_DIM, generator=centers_generator),
        dim=1,
    ) * 2.5

    styles_generator = torch.Generator().manual_seed(DEMO_CLASS_STYLE_SEED)
    class_styles = F.normalize(
        torch.randn(DEMO_OUTPUT_DIM, DEMO_INPUT_DIM, generator=styles_generator),
        dim=1,
    )

    main_dataset = PlaceholderDataset(
        size=DEMO_MAIN_DATASET_SIZE,
        seed=DEMO_MAIN_DATASET_SEED,
        variant="main",
        class_centers=class_centers,
        class_styles=class_styles,
    )
    corner_dataset = PlaceholderDataset(
        size=DEMO_CORNER_DATASET_SIZE,
        seed=DEMO_CORNER_DATASET_SEED,
        variant="corner",
        class_centers=class_centers,
        class_styles=class_styles,
    )

    return model, main_dataset, corner_dataset
