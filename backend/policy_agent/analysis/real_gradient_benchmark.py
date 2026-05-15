"""Real-data gradient benchmark for CornerDrive.

The synthetic benchmark is useful for controlled attack families, but it can
look circular because gradients are generated from hand-built archetypes. This
module builds client gradients from real image datasets instead:

- preferred: LEAF/FEMNIST processed JSON, preserving real client partitions;
- fallback: torchvision MNIST/FashionMNIST with deterministic non-IID shards.

Raw public gradient traces are rare because gradients can leak private data, so
the reproducible path is to load public client data and derive gradients with a
fixed model/checkpoint.
"""

from __future__ import annotations

import copy
import csv
import json
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = PROJECT_ROOT / "backend"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

for candidate in (PROJECT_ROOT, BACKEND_DIR, SCRIPTS_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from common.config import L2_LEARNING_RATE  # noqa: E402
from common.schemas import DEFAULT_POLICY, Policy  # noqa: E402
from l1_linear_defense.aggregation import filter_suspects, geometric_median  # noqa: E402
from l1_linear_defense.config import L1RouterConfig, make_l1_router_config  # noqa: E402
from l2_dual_audit.classifier import DualChannelAuditor  # noqa: E402


DEFAULT_CORNER_LABELS = (1, 7, 9)
REAL_DATA_ADAPTIVE_POLICY_UPDATES: dict[str, float] = {
    "theta_tol": 0.02,
    "theta_rare": -0.005,
    "cosine_filter_threshold": 0.60,
    "recheck_probability": 0.25,
}


@dataclass(frozen=True)
class RealClient:
    client_id: str
    inputs: torch.Tensor
    targets: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def label_histogram(self) -> dict[int, int]:
        labels = [int(label) for label in self.targets.tolist()]
        return dict(Counter(labels))

    @property
    def size(self) -> int:
        return int(self.targets.numel())


@dataclass(frozen=True)
class ClientGradient:
    client_id: str
    gradient: np.ndarray
    ground_truth: str
    attack_family: str
    sample_count: int
    label_histogram: dict[int, int]


@dataclass(frozen=True)
class RealGradientDataBundle:
    """Leakage-safe data surfaces for the real-gradient benchmark."""

    clients: list[RealClient]
    audit_main_dataset: torch.utils.data.Dataset
    audit_corner_dataset: torch.utils.data.Dataset
    eval_main_dataset: torch.utils.data.Dataset
    eval_corner_dataset: torch.utils.data.Dataset
    dataset_info: dict[str, Any]


@dataclass(frozen=True)
class RealGradientBenchmarkConfig:
    source: str = "auto"
    leaf_data_dir: str = "data/real/femnist"
    bdd_data_dir: str = "data/real/bdd100k"
    bdd_label_file: str = ""
    bdd_image_dir: str = ""
    bdd_image_size: int = 32
    bdd_target_attribute: str = "weather"
    bdd_client_group: str = "weather_timeofday"
    bdd_corner_values: str = "rainy,snowy,foggy"
    data_dir: str = "data/real"
    download: bool = False
    max_clients: int = 80
    min_samples_per_client: int = 8
    max_samples_per_client: int = 32
    clients_per_round: int = 16
    rounds: int = 8
    seed: int = 20260507
    pretrain_steps: int = 40
    local_batch_size: int = 16
    reference_split_fraction: float = 0.50
    max_reference_samples: int = 4096
    max_evaluation_samples: int = 4096
    attack_fraction: float = 0.20
    corner_harm_fraction: float = 0.05
    noise_fraction: float = 0.05
    rarity_label_fraction_threshold: float = 0.30
    sign_flip_scale: float = 3.0
    corner_harm_scale: float = 2.0
    zeno_score_penalty: float = 1e-4
    zenopp_score_temperature: float = 0.05
    cornerdrive_l1_mode: str = "v25_cosine_fixed"
    cornerdrive_l1_norm_mad_threshold: float = 3.0
    cornerdrive_l1_sign_threshold: float = 0.65
    cornerdrive_l1_sign_topk_ratio: float = 0.10
    cornerdrive_l1_queue_budget_ratio: float = 0.35
    cornerdrive_l1_random_recheck_ratio: float = 0.05


class TinyImageMLP(nn.Module):
    """Small model used to derive reproducible client gradients."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 64, output_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1).float())


def _normalize_images(raw_inputs: Any) -> torch.Tensor:
    tensor = torch.tensor(raw_inputs, dtype=torch.float32)
    if tensor.ndim > 2:
        tensor = tensor.view(tensor.size(0), -1)
    if tensor.numel() and float(tensor.max()) > 2.0:
        tensor = tensor / 255.0
    return tensor


def _client_from_arrays(
    client_id: str,
    raw_inputs: Any,
    raw_targets: Any,
    *,
    max_samples: int,
) -> RealClient | None:
    inputs = _normalize_images(raw_inputs)
    targets = torch.tensor(raw_targets, dtype=torch.long)

    if inputs.ndim != 2 or targets.ndim != 1 or inputs.size(0) != targets.size(0):
        return None
    if inputs.size(0) == 0:
        return None

    if inputs.size(0) > max_samples:
        inputs = inputs[:max_samples]
        targets = targets[:max_samples]

    return RealClient(client_id=client_id, inputs=inputs, targets=targets)


def _leaf_json_files(root: Path, *, split: str | None = None) -> list[Path]:
    search_root = root / split if split and (root / split).exists() else root
    candidates: list[Path] = []
    if split in {"train", "test"}:
        patterns = ("**/all_data*.json", f"**/*{split}*.json")
    else:
        patterns = ("**/all_data*.json", "**/*train*.json", "**/*test*.json")
    for pattern in patterns:
        candidates.extend(search_root.glob(pattern))
    return sorted(set(candidates))


def load_leaf_femnist_clients(
    root: Path,
    *,
    max_clients: int,
    min_samples_per_client: int,
    max_samples_per_client: int,
    split: str | None = "train",
) -> tuple[list[RealClient], dict[str, Any]]:
    """Load LEAF/FEMNIST processed JSON clients."""
    if not root.exists():
        raise FileNotFoundError(f"LEAF data directory does not exist: {root}")

    clients: list[RealClient] = []
    json_files = _leaf_json_files(root, split=split)
    for json_file in json_files:
        payload = json.loads(json_file.read_text())
        users = payload.get("users", [])
        user_data = payload.get("user_data", {})
        for user in users:
            record = user_data.get(user, {})
            client = _client_from_arrays(
                str(user),
                record.get("x", []),
                record.get("y", []),
                max_samples=max_samples_per_client,
            )
            if client is None or client.size < min_samples_per_client:
                continue
            clients.append(client)
            if len(clients) >= max_clients:
                return clients, {
                    "source": "leaf_femnist",
                    "root": str(root),
                    "files_scanned": len(json_files),
                    "split": split or "all",
                    "real_client_partitions": True,
                    "corner_labels": list(DEFAULT_CORNER_LABELS),
                }

    if not clients:
        raise ValueError(f"No usable LEAF clients found under {root}")

    return clients, {
        "source": "leaf_femnist",
        "root": str(root),
        "files_scanned": len(json_files),
        "split": split or "all",
        "real_client_partitions": True,
        "corner_labels": list(DEFAULT_CORNER_LABELS),
    }


def _load_torchvision_dataset(name: str, root: Path, download: bool, *, train: bool = True):
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for the torchvision fallback; use LEAF JSON "
            "or install torchvision."
        ) from exc

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda tensor: tensor.view(-1)),
    ])
    dataset_name = name.lower()
    if dataset_name in {"mnist", "torchvision_mnist"}:
        return datasets.MNIST(root=str(root), train=train, transform=transform, download=download)
    if dataset_name in {"fashionmnist", "fashion_mnist", "torchvision_fashionmnist"}:
        return datasets.FashionMNIST(root=str(root), train=train, transform=transform, download=download)
    raise ValueError(f"Unsupported torchvision dataset source: {name}")


def load_torchvision_clients(
    name: str,
    root: Path,
    *,
    download: bool,
    max_clients: int,
    min_samples_per_client: int,
    max_samples_per_client: int,
    seed: int,
) -> tuple[list[RealClient], dict[str, Any]]:
    """Build deterministic non-IID clients from a real torchvision dataset."""
    dataset = _load_torchvision_dataset(name, root, download, train=True)
    rng = random.Random(seed)

    indices_by_label: dict[int, list[int]] = {}
    for idx in range(len(dataset)):
        _input, target = dataset[idx]
        indices_by_label.setdefault(int(target), []).append(idx)

    for indices in indices_by_label.values():
        rng.shuffle(indices)

    clients: list[RealClient] = []
    labels = sorted(indices_by_label)
    label_cursor = 0
    while len(clients) < max_clients:
        client_labels = [
            labels[label_cursor % len(labels)],
            labels[(label_cursor + 3) % len(labels)],
        ]
        label_cursor += 1
        selected: list[int] = []
        per_label = max(1, max_samples_per_client // len(client_labels))
        for label in client_labels:
            bucket = indices_by_label[label]
            if len(bucket) < per_label:
                continue
            selected.extend(bucket[:per_label])
            del bucket[:per_label]

        if len(selected) < min_samples_per_client:
            break

        rng.shuffle(selected)
        xs: list[torch.Tensor] = []
        ys: list[int] = []
        for idx in selected[:max_samples_per_client]:
            sample, target = dataset[idx]
            xs.append(sample.view(-1).float())
            ys.append(int(target))

        clients.append(
            RealClient(
                client_id=f"{name}:client:{len(clients):04d}",
                inputs=torch.stack(xs),
                targets=torch.tensor(ys, dtype=torch.long),
            )
        )

    if not clients:
        raise ValueError(f"No torchvision clients could be built for {name}")

    return clients, {
        "source": name,
        "root": str(root),
        "real_client_partitions": False,
        "partition_note": "real samples with deterministic non-IID client shards",
        "corner_labels": list(DEFAULT_CORNER_LABELS),
    }


BDD_ATTRIBUTE_CHOICES = {"weather", "timeofday", "scene"}


def _resolve_repo_path(raw: str) -> Path:
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _bdd_label_files(root: Path, explicit: str) -> list[Path]:
    if explicit:
        candidate = _resolve_repo_path(explicit)
        if not candidate.exists():
            raise FileNotFoundError(f"BDD100K label file does not exist: {candidate}")
        return [candidate]

    patterns = [
        "labels/bdd100k_labels_images_train.json",
        "labels/bdd100k_labels_images_val.json",
        "labels/*labels*images*.json",
        "labels/100k/train.json",
        "labels/100k/val.json",
        "labels/**/*.json",
        "**/*labels*images*.json",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted(set(path for path in files if path.is_file()))


def _bdd_image_roots(root: Path, explicit: str) -> list[Path]:
    if explicit:
        candidate = _resolve_repo_path(explicit)
        if not candidate.exists():
            raise FileNotFoundError(f"BDD100K image directory does not exist: {candidate}")
        return [candidate]

    candidates = [
        root / "images" / "100k" / "train",
        root / "images" / "100k" / "val",
        root / "images" / "10k" / "train",
        root / "images" / "10k" / "val",
        root / "images" / "train",
        root / "images" / "val",
        root / "images",
        root,
    ]
    return [path for path in candidates if path.exists()]


def _bdd_group_key(attributes: dict[str, Any], client_group: str) -> str:
    fields = [part.strip() for part in client_group.split("_") if part.strip()]
    if not fields:
        fields = ["weather", "timeofday"]
    values = [str(attributes.get(field, "unknown")).strip().lower() for field in fields]
    return "|".join(value or "unknown" for value in values)


def _bdd_image_tensor(path: Path, image_size: int) -> torch.Tensor:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to load BDD100K image files") from exc

    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).view(-1)


def _find_bdd_image(
    name: str,
    image_roots: list[Path],
    image_index: dict[str, Path],
) -> Path | None:
    named_path = Path(name)
    for root in image_roots:
        candidate = root / named_path
        if candidate.exists():
            return candidate
        candidate = root / named_path.name
        if candidate.exists():
            return candidate

    if not image_index:
        for root in image_roots:
            for path in root.rglob("*"):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    image_index.setdefault(path.name, path)
    return image_index.get(named_path.name)


def _load_bdd_label_records(label_files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label_file in label_files:
        payload = json.loads(label_file.read_text())
        if isinstance(payload, list):
            records.extend(record for record in payload if isinstance(record, dict))
        elif isinstance(payload, dict):
            for key in ("frames", "labels", "annotations"):
                values = payload.get(key)
                if isinstance(values, list):
                    records.extend(record for record in values if isinstance(record, dict))
                    break
    return records


def _parse_corner_values(raw: str) -> set[str]:
    return {
        part.strip().lower()
        for part in raw.split(",")
        if part.strip()
    }


def load_bdd100k_clients(
    root: Path,
    *,
    label_file: str = "",
    image_dir: str = "",
    image_size: int = 32,
    target_attribute: str = "weather",
    client_group: str = "weather_timeofday",
    corner_values: str = "rainy,snowy,foggy",
    max_clients: int,
    min_samples_per_client: int,
    max_samples_per_client: int,
    seed: int,
) -> tuple[list[RealClient], dict[str, Any]]:
    """Load BDD100K image labels as pseudo-clients for IoV gradient calibration.

    BDD100K does not ship federated client ids. This adapter constructs
    deterministic pseudo-clients from frame attributes such as weather,
    time-of-day, and scene. That is a domain-relevance calibration path, not a
    claim of real vehicle-level federation.
    """

    if not root.exists():
        raise FileNotFoundError(f"BDD100K data directory does not exist: {root}")

    target = target_attribute.strip().lower()
    if target not in BDD_ATTRIBUTE_CHOICES:
        supported = ", ".join(sorted(BDD_ATTRIBUTE_CHOICES))
        raise ValueError(f"Unsupported BDD100K target attribute {target_attribute!r}: {supported}")

    label_files = _bdd_label_files(root, label_file)
    if not label_files:
        raise FileNotFoundError(
            "No BDD100K label JSON found. Expected files such as "
            "labels/bdd100k_labels_images_train.json under "
            f"{root} or pass --bdd-label-file."
        )
    image_roots = _bdd_image_roots(root, image_dir)
    if not image_roots:
        raise FileNotFoundError(
            f"No BDD100K image directory found under {root}; pass --bdd-image-dir."
        )

    records = _load_bdd_label_records(label_files)
    rng = random.Random(seed)
    rng.shuffle(records)
    image_index: dict[str, Path] = {}
    class_to_id: dict[str, int] = {}
    grouped: dict[str, list[tuple[Path, int, dict[str, Any]]]] = {}

    for record in records:
        name = str(record.get("name", "")).strip()
        attributes = record.get("attributes", {})
        if not name or not isinstance(attributes, dict):
            continue
        target_value = str(attributes.get(target, "")).strip().lower()
        if not target_value or target_value in {"undefined", "unknown", "none"}:
            continue
        image_path = _find_bdd_image(name, image_roots, image_index)
        if image_path is None:
            continue

        class_id = class_to_id.setdefault(target_value, len(class_to_id))
        group_key = _bdd_group_key(attributes, client_group)
        grouped.setdefault(group_key, []).append(
            (
                image_path,
                class_id,
                {
                    "name": name,
                    "attributes": {
                        key: str(value)
                        for key, value in attributes.items()
                    },
                    "target_value": target_value,
                    "image_path": str(image_path),
                },
            )
        )

    clients: list[RealClient] = []
    for group_key, items in sorted(grouped.items()):
        rng.shuffle(items)
        for chunk_id, start in enumerate(range(0, len(items), max_samples_per_client)):
            chunk = items[start:start + max_samples_per_client]
            if len(chunk) < min_samples_per_client:
                continue
            inputs = torch.stack([_bdd_image_tensor(item[0], image_size) for item in chunk])
            targets = torch.tensor([item[1] for item in chunk], dtype=torch.long)
            clients.append(
                RealClient(
                    client_id=f"bdd100k:{client_group}:{group_key}:{chunk_id:03d}",
                    inputs=inputs,
                    targets=targets,
                    metadata={
                        "client_group": client_group,
                        "group_key": group_key,
                        "records": [item[2] for item in chunk],
                    },
                )
            )
            if len(clients) >= max_clients:
                break
        if len(clients) >= max_clients:
            break

    if not clients:
        raise ValueError(
            "No usable BDD100K pseudo-clients were built. Check image paths, "
            "label file, min/max samples, and target attribute."
        )

    corner_value_set = _parse_corner_values(corner_values)
    corner_label_ids = [
        class_id
        for value, class_id in sorted(class_to_id.items(), key=lambda item: item[1])
        if value in corner_value_set
    ]
    if not corner_label_ids and class_to_id:
        # Keep the benchmark runnable even for tiny smoke datasets that do not
        # contain the default rainy/snowy/foggy labels.
        corner_label_ids = [max(class_to_id.values())]

    id_to_class = {
        class_id: value
        for value, class_id in class_to_id.items()
    }
    return clients, {
        "source": "bdd100k",
        "root": str(root),
        "label_files": [str(path) for path in label_files],
        "image_roots": [str(path) for path in image_roots],
        "real_client_partitions": False,
        "partition_note": (
            "BDD100K image frames grouped into deterministic pseudo-clients by "
            f"{client_group}; use as IoV/domain calibration evidence."
        ),
        "target_attribute": target,
        "client_group": client_group,
        "image_size": image_size,
        "class_to_id": class_to_id,
        "id_to_class": id_to_class,
        "corner_values": sorted(corner_value_set),
        "corner_labels": corner_label_ids,
    }


def load_real_clients(config: RealGradientBenchmarkConfig) -> tuple[list[RealClient], dict[str, Any]]:
    source = config.source.lower()
    leaf_root = PROJECT_ROOT / config.leaf_data_dir
    bdd_root = PROJECT_ROOT / config.bdd_data_dir
    data_root = PROJECT_ROOT / config.data_dir

    if source in {"auto", "leaf", "leaf_femnist", "femnist"} and leaf_root.exists():
        try:
            return load_leaf_femnist_clients(
                leaf_root,
                max_clients=config.max_clients,
                min_samples_per_client=config.min_samples_per_client,
                max_samples_per_client=config.max_samples_per_client,
                split="train",
            )
        except Exception:
            if source != "auto":
                raise

    if source in {"leaf", "leaf_femnist", "femnist"}:
        return load_leaf_femnist_clients(
            leaf_root,
            max_clients=config.max_clients,
            min_samples_per_client=config.min_samples_per_client,
            max_samples_per_client=config.max_samples_per_client,
            split="train",
        )

    if source in {"auto", "bdd", "bdd100k"} and bdd_root.exists():
        try:
            return load_bdd100k_clients(
                bdd_root,
                label_file=config.bdd_label_file,
                image_dir=config.bdd_image_dir,
                image_size=config.bdd_image_size,
                target_attribute=config.bdd_target_attribute,
                client_group=config.bdd_client_group,
                corner_values=config.bdd_corner_values,
                max_clients=config.max_clients,
                min_samples_per_client=config.min_samples_per_client,
                max_samples_per_client=config.max_samples_per_client,
                seed=config.seed,
            )
        except Exception:
            if source != "auto":
                raise

    if source in {"bdd", "bdd100k"}:
        return load_bdd100k_clients(
            bdd_root,
            label_file=config.bdd_label_file,
            image_dir=config.bdd_image_dir,
            image_size=config.bdd_image_size,
            target_attribute=config.bdd_target_attribute,
            client_group=config.bdd_client_group,
            corner_values=config.bdd_corner_values,
            max_clients=config.max_clients,
            min_samples_per_client=config.min_samples_per_client,
            max_samples_per_client=config.max_samples_per_client,
            seed=config.seed,
        )

    fallback_source = "mnist" if source == "auto" else source
    return load_torchvision_clients(
        fallback_source,
        data_root,
        download=config.download,
        max_clients=config.max_clients,
        min_samples_per_client=config.min_samples_per_client,
        max_samples_per_client=config.max_samples_per_client,
        seed=config.seed,
    )


def _cap_indices(indices: Sequence[int], limit: int) -> list[int]:
    if limit <= 0:
        return list(indices)
    return list(indices[:limit])


def _tensor_dataset_from_source(
    dataset: torch.utils.data.Dataset,
    indices: Sequence[int],
) -> torch.utils.data.TensorDataset:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for idx in indices:
        sample, target = dataset[idx]
        xs.append(torch.as_tensor(sample).view(-1).float())
        ys.append(torch.as_tensor(target, dtype=torch.long).view(()))
    if not xs:
        raise ValueError("Cannot build a TensorDataset from an empty index set")
    return torch.utils.data.TensorDataset(torch.stack(xs), torch.stack(ys).long())


def _split_dataset_by_corner(
    dataset: torch.utils.data.TensorDataset,
    corner_labels: tuple[int, ...],
) -> tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    inputs, targets = dataset.tensors
    corner_mask = torch.zeros_like(targets, dtype=torch.bool)
    for label in corner_labels:
        corner_mask |= targets == label

    if int(corner_mask.sum().item()) < 4:
        corner_mask[: max(4, min(16, targets.numel()))] = True

    return dataset, torch.utils.data.TensorDataset(inputs[corner_mask], targets[corner_mask])


def _split_clients_deterministically(
    clients: list[RealClient],
    *,
    seed: int,
    reference_fraction: float,
) -> tuple[list[RealClient], list[RealClient]]:
    if len(clients) < 2:
        return clients, clients
    shuffled = list(clients)
    random.Random(seed).shuffle(shuffled)
    split_at = int(round(len(shuffled) * reference_fraction))
    split_at = max(1, min(split_at, len(shuffled) - 1))
    return shuffled[:split_at], shuffled[split_at:]


def _partition_client_pool(
    clients: list[RealClient],
    *,
    seed: int,
) -> tuple[list[RealClient], list[RealClient], list[RealClient], str]:
    """Fallback partition for sources without a clean official train/test split."""

    if len(clients) < 3:
        return clients, clients, clients, "shared_tiny_fixture"

    shuffled = list(clients)
    random.Random(seed).shuffle(shuffled)
    eval_count = max(1, len(shuffled) // 5)
    reference_count = max(1, len(shuffled) // 5)
    update_count = len(shuffled) - reference_count - eval_count
    if update_count < 1:
        return clients, clients, clients, "shared_tiny_fixture"

    update_clients = shuffled[:update_count]
    reference_clients = shuffled[update_count:update_count + reference_count]
    eval_clients = shuffled[update_count + reference_count:]
    return update_clients, reference_clients, eval_clients, "deterministic_client_partition"


def _load_torchvision_reference_eval_datasets(
    name: str,
    root: Path,
    *,
    download: bool,
    seed: int,
    corner_labels: tuple[int, ...],
    reference_fraction: float,
    max_reference_samples: int,
    max_evaluation_samples: int,
) -> tuple[
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    torch.utils.data.Dataset,
    dict[str, Any],
]:
    dataset = _load_torchvision_dataset(name, root, download, train=False)
    indices = list(range(len(dataset)))
    random.Random(seed + 17).shuffle(indices)
    split_at = int(round(len(indices) * reference_fraction))
    split_at = max(1, min(split_at, len(indices) - 1))

    reference_indices = _cap_indices(indices[:split_at], max_reference_samples)
    evaluation_indices = _cap_indices(indices[split_at:], max_evaluation_samples)
    reference_dataset = _tensor_dataset_from_source(dataset, reference_indices)
    evaluation_dataset = _tensor_dataset_from_source(dataset, evaluation_indices)
    audit_main, audit_corner = _split_dataset_by_corner(reference_dataset, corner_labels)
    eval_main, eval_corner = _split_dataset_by_corner(evaluation_dataset, corner_labels)

    return audit_main, audit_corner, eval_main, eval_corner, {
        "split_protocol": "official_train_clients_test_reference_eval",
        "update_split": "torchvision_train",
        "reference_split": "torchvision_test_reference",
        "evaluation_split": "torchvision_test_evaluation",
        "reference_sample_count": len(reference_dataset),
        "evaluation_sample_count": len(evaluation_dataset),
    }


def _make_tensor_dataset(clients: Iterable[RealClient]) -> torch.utils.data.TensorDataset:
    inputs = torch.cat([client.inputs for client in clients], dim=0)
    targets = torch.cat([client.targets for client in clients], dim=0)
    return torch.utils.data.TensorDataset(inputs, targets)


def _split_reference_clients(
    clients: list[RealClient],
    corner_labels: tuple[int, ...] = DEFAULT_CORNER_LABELS,
) -> tuple[torch.utils.data.TensorDataset, torch.utils.data.TensorDataset]:
    all_inputs = torch.cat([client.inputs for client in clients], dim=0)
    all_targets = torch.cat([client.targets for client in clients], dim=0)
    corner_mask = torch.zeros_like(all_targets, dtype=torch.bool)
    for label in corner_labels:
        corner_mask |= all_targets == label

    if int(corner_mask.sum().item()) < 4:
        corner_mask[: max(4, min(16, all_targets.numel()))] = True

    main_dataset = torch.utils.data.TensorDataset(all_inputs, all_targets)
    corner_dataset = torch.utils.data.TensorDataset(all_inputs[corner_mask], all_targets[corner_mask])
    return main_dataset, corner_dataset


def build_real_gradient_data_bundle(config: RealGradientBenchmarkConfig) -> RealGradientDataBundle:
    clients, dataset_info = load_real_clients(config)
    data_root = PROJECT_ROOT / config.data_dir
    leaf_root = PROJECT_ROOT / config.leaf_data_dir
    corner_labels = tuple(
        int(label)
        for label in dataset_info.get("corner_labels", list(DEFAULT_CORNER_LABELS))
    )
    dataset_source = str(dataset_info.get("source", config.source)).lower()

    split_info: dict[str, Any]
    if dataset_source in {"mnist", "torchvision_mnist", "fashionmnist", "torchvision_fashionmnist"}:
        audit_main, audit_corner, eval_main, eval_corner, split_info = _load_torchvision_reference_eval_datasets(
            dataset_source,
            data_root,
            download=config.download,
            seed=config.seed,
            corner_labels=corner_labels,
            reference_fraction=config.reference_split_fraction,
            max_reference_samples=config.max_reference_samples,
            max_evaluation_samples=config.max_evaluation_samples,
        )
    elif dataset_source == "leaf_femnist":
        try:
            heldout_clients, heldout_info = load_leaf_femnist_clients(
                leaf_root,
                max_clients=max(config.max_clients, 2),
                min_samples_per_client=config.min_samples_per_client,
                max_samples_per_client=config.max_samples_per_client,
                split="test",
            )
            reference_clients, eval_clients = _split_clients_deterministically(
                heldout_clients,
                seed=config.seed + 17,
                reference_fraction=config.reference_split_fraction,
            )
            audit_main, audit_corner = _split_reference_clients(reference_clients, corner_labels)
            eval_main, eval_corner = _split_reference_clients(eval_clients, corner_labels)
            split_info = {
                "split_protocol": "leaf_train_clients_test_reference_eval",
                "update_split": "leaf_train",
                "reference_split": "leaf_test_reference_clients",
                "evaluation_split": "leaf_test_evaluation_clients",
                "heldout_files_scanned": heldout_info.get("files_scanned", 0),
                "heldout_client_count": len(heldout_clients),
            }
        except Exception as exc:
            update_clients, reference_clients, eval_clients, strategy = _partition_client_pool(
                clients,
                seed=config.seed + 17,
            )
            clients = update_clients
            audit_main, audit_corner = _split_reference_clients(reference_clients, corner_labels)
            eval_main, eval_corner = _split_reference_clients(eval_clients, corner_labels)
            split_info = {
                "split_protocol": strategy,
                "split_warning": f"FEMNIST test split unavailable: {exc}",
                "update_split": "leaf_partition_update",
                "reference_split": "leaf_partition_reference",
                "evaluation_split": "leaf_partition_evaluation",
            }
    else:
        update_clients, reference_clients, eval_clients, strategy = _partition_client_pool(
            clients,
            seed=config.seed + 17,
        )
        clients = update_clients
        audit_main, audit_corner = _split_reference_clients(reference_clients, corner_labels)
        eval_main, eval_corner = _split_reference_clients(eval_clients, corner_labels)
        split_info = {
            "split_protocol": strategy,
            "update_split": f"{dataset_source}_partition_update",
            "reference_split": f"{dataset_source}_partition_reference",
            "evaluation_split": f"{dataset_source}_partition_evaluation",
        }

    dataset_info = {
        **dataset_info,
        **split_info,
        "data_leakage_guard": (
            "client/update gradients, audit/reference gradients, and final "
            "evaluation metrics are constructed from separate deterministic "
            "surfaces when the source exposes enough data."
        ),
        "update_client_count": len(clients),
        "audit_main_sample_count": len(audit_main),
        "audit_corner_sample_count": len(audit_corner),
        "eval_main_sample_count": len(eval_main),
        "eval_corner_sample_count": len(eval_corner),
        "reference_split_fraction": config.reference_split_fraction,
        "max_reference_samples": config.max_reference_samples,
        "max_evaluation_samples": config.max_evaluation_samples,
    }
    return RealGradientDataBundle(
        clients=clients,
        audit_main_dataset=audit_main,
        audit_corner_dataset=audit_corner,
        eval_main_dataset=eval_main,
        eval_corner_dataset=eval_corner,
        dataset_info=dataset_info,
    )


def _flat_gradient(model: nn.Module) -> np.ndarray:
    parts = []
    for parameter in model.parameters():
        grad = parameter.grad
        if grad is None:
            parts.append(torch.zeros_like(parameter.data).view(-1))
        else:
            parts.append(grad.detach().view(-1))
    return torch.cat(parts).cpu().numpy()


def compute_client_gradient(
    model: nn.Module,
    client: RealClient,
    *,
    batch_size: int,
) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_samples = 0

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(client.inputs, client.targets),
        batch_size=batch_size,
        shuffle=False,
    )
    for inputs, targets in loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss * inputs.size(0)
        total_samples += int(inputs.size(0))

    if total_samples == 0:
        return np.zeros(sum(parameter.numel() for parameter in model.parameters()))

    (total_loss / total_samples).backward()
    return _flat_gradient(model)


def _gradient_for_dataset(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
) -> np.ndarray:
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for sample, target in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
        inputs.append(sample)
        targets.append(target)
        if sum(chunk.size(0) for chunk in inputs) >= batch_size * 4:
            break
    client = RealClient("reference", torch.cat(inputs), torch.cat(targets))
    return compute_client_gradient(model, client, batch_size=batch_size)


def _max_target_in_dataset(dataset: torch.utils.data.Dataset) -> int:
    if isinstance(dataset, torch.utils.data.TensorDataset):
        return int(dataset.tensors[1].max().item())
    max_target = 0
    for _sample, target in torch.utils.data.DataLoader(dataset, batch_size=256):
        max_target = max(max_target, int(torch.as_tensor(target).max().item()))
    return max_target


def _pretrain_model(
    clients: list[RealClient],
    *,
    input_dim: int,
    output_dim: int,
    steps: int,
    batch_size: int,
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    model = TinyImageMLP(input_dim=input_dim, output_dim=output_dim)
    if steps <= 0:
        return model

    pooled = _make_tensor_dataset(clients)
    loader = torch.utils.data.DataLoader(
        pooled,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    iterator = iter(loader)
    for _ in range(steps):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            inputs, targets = next(iterator)

        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    return model


def _evaluate_model(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
) -> dict[str, float]:
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            predictions = outputs.argmax(dim=1)
            total_loss += float(loss.item()) * inputs.size(0)
            total_correct += int((predictions == targets).sum().item())
            total_samples += int(inputs.size(0))
    model.train()
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def _apply_gradient(model: nn.Module, gradient: np.ndarray, learning_rate: float) -> nn.Module:
    model_copy = copy.deepcopy(model)
    flat_params = torch.cat([parameter.data.view(-1) for parameter in model_copy.parameters()])
    gradient_tensor = torch.tensor(gradient, dtype=torch.float32)
    if flat_params.numel() != gradient_tensor.numel():
        raise ValueError("Gradient dimension does not match model")
    flat_params -= learning_rate * gradient_tensor
    offset = 0
    for parameter in model_copy.parameters():
        param_size = parameter.numel()
        parameter.data = flat_params[offset:offset + param_size].view(parameter.shape)
        offset += param_size
    return model_copy


def _mean_gradient(gradients: list[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(gradients), axis=0)


def _multi_krum_indices(gradients: list[np.ndarray], byzantine_budget: int = 2) -> list[int]:
    n = len(gradients)
    if n <= 2:
        return list(range(n))
    f = max(1, min(byzantine_budget, (n - 3) // 2))
    neighbor_count = max(1, n - f - 2)
    selection_count = max(1, n - f - 2)
    matrix = np.stack(gradients)
    distances = np.sum((matrix[:, None, :] - matrix[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distances, np.inf)
    scores = [
        (float(np.sum(np.partition(distances[idx], neighbor_count - 1)[:neighbor_count])), idx)
        for idx in range(n)
    ]
    scores.sort(key=lambda item: (item[0], item[1]))
    return [idx for _score, idx in scores[:selection_count]]


def _estimated_byzantine_budget(config: RealGradientBenchmarkConfig, gradient_count: int) -> int:
    expected = int(round(gradient_count * (config.attack_fraction + config.corner_harm_fraction)))
    return max(1, min(expected, max(gradient_count - 1, 1)))


def _zero_gradient_like(gradients: list[np.ndarray]) -> np.ndarray:
    return np.zeros_like(gradients[0])


def _fltrust_aggregate(
    gradients: list[np.ndarray],
    root_gradient: np.ndarray,
) -> tuple[np.ndarray, set[int], list[float]]:
    """Aggregate client gradients with FLTrust-style cosine trust bootstrapping."""
    eps = 1e-12
    root_norm = float(np.linalg.norm(root_gradient))
    if root_norm <= eps:
        return _zero_gradient_like(gradients), set(), [0.0 for _ in gradients]

    scaled_gradients: list[np.ndarray] = []
    trust_scores: list[float] = []
    selected_indices: set[int] = set()
    for idx, gradient in enumerate(gradients):
        grad_norm = float(np.linalg.norm(gradient))
        if grad_norm <= eps:
            trust_scores.append(0.0)
            scaled_gradients.append(np.zeros_like(gradient))
            continue

        cosine = float(np.dot(gradient, root_gradient) / (grad_norm * root_norm + eps))
        trust = max(cosine, 0.0)
        trust_scores.append(trust)
        if trust > 0.0:
            selected_indices.add(idx)
        scaled_gradients.append((gradient / grad_norm) * root_norm)

    total_trust = sum(trust_scores)
    if total_trust <= eps:
        return _zero_gradient_like(gradients), selected_indices, trust_scores

    aggregated = sum(
        trust * scaled
        for trust, scaled in zip(trust_scores, scaled_gradients)
    ) / total_trust
    return aggregated, selected_indices, trust_scores


def _loss_on_dataset(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
) -> float:
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets in torch.utils.data.DataLoader(dataset, batch_size=batch_size):
            outputs = model(inputs)
            total_loss += float(criterion(outputs, targets).item())
            total_samples += int(inputs.size(0))
    return total_loss / max(total_samples, 1)


def _zeno_scores(
    model: nn.Module,
    gradients: list[np.ndarray],
    validation_dataset: torch.utils.data.Dataset,
    *,
    batch_size: int,
    learning_rate: float,
    score_penalty: float,
) -> list[float]:
    """Score gradients by validation loss decrease minus a norm penalty."""
    before_loss = _loss_on_dataset(model, validation_dataset, batch_size=batch_size)
    scores: list[float] = []
    for gradient in gradients:
        candidate = _apply_gradient(model, gradient, learning_rate)
        after_loss = _loss_on_dataset(candidate, validation_dataset, batch_size=batch_size)
        norm_penalty = score_penalty * float(np.dot(gradient, gradient))
        scores.append(before_loss - after_loss - norm_penalty)
    return scores


def _zeno_aggregate(
    gradients: list[np.ndarray],
    scores: list[float],
    *,
    byzantine_budget: int,
) -> tuple[np.ndarray, set[int]]:
    selection_count = max(1, len(gradients) - byzantine_budget)
    ranked = sorted(range(len(gradients)), key=lambda idx: (scores[idx], -idx), reverse=True)
    selected_indices = set(ranked[:selection_count])
    return _mean_gradient([gradients[idx] for idx in sorted(selected_indices)]), selected_indices


def _zenopp_aggregate(
    gradients: list[np.ndarray],
    scores: list[float],
    *,
    temperature: float,
) -> tuple[np.ndarray, set[int], list[float]]:
    """Synchronous Zeno++-style score-weighted aggregation for benchmark parity."""
    positive_scores = np.array([max(score, 0.0) for score in scores], dtype=np.float64)
    selected_indices = {idx for idx, score in enumerate(positive_scores) if score > 0.0}
    if not selected_indices:
        best_idx = max(range(len(scores)), key=lambda idx: (scores[idx], -idx))
        return gradients[best_idx].copy(), {best_idx}, [
            1.0 if idx == best_idx else 0.0
            for idx in range(len(scores))
        ]

    temp = max(float(temperature), 1e-6)
    logits = positive_scores / temp
    logits -= float(np.max(logits))
    raw_weights = np.exp(logits)
    raw_weights[positive_scores <= 0.0] = 0.0
    weights = raw_weights / max(float(np.sum(raw_weights)), 1e-12)
    aggregated = sum(float(weight) * gradient for weight, gradient in zip(weights, gradients))
    return aggregated, selected_indices, [float(weight) for weight in weights]


def _client_is_corner_heavy(
    client: RealClient,
    corner_labels: tuple[int, ...],
    threshold: float = 0.30,
) -> bool:
    labels = client.targets.tolist()
    if not labels:
        return False
    corner_count = sum(1 for label in labels if int(label) in corner_labels)
    return corner_count / len(labels) >= threshold


def _build_attack_plan(
    *,
    client_count: int,
    rng: random.Random,
    config: RealGradientBenchmarkConfig,
) -> dict[int, str]:
    indices = list(range(client_count))
    rng.shuffle(indices)
    sign_flip_count = int(round(client_count * config.attack_fraction))
    corner_harm_count = int(round(client_count * config.corner_harm_fraction))
    noise_count = int(round(client_count * config.noise_fraction))

    plan: dict[int, str] = {}
    for idx in indices[:sign_flip_count]:
        plan[idx] = "sign_flip_proxy"

    start = sign_flip_count
    for idx in indices[start:start + corner_harm_count]:
        plan[idx] = "corner_harm"

    start += corner_harm_count
    for idx in indices[start:start + noise_count]:
        plan[idx] = "benign_noise"

    return plan


def _round_truth(
    *,
    round_clients: list[RealClient],
    attack_plan: dict[int, str],
    corner_labels: tuple[int, ...],
    rarity_threshold: float,
) -> list[str]:
    truth: list[str] = []
    for idx, client in enumerate(round_clients):
        attack_family = attack_plan.get(idx, "none")
        if attack_family in {"sign_flip_proxy", "corner_harm"}:
            truth.append("FRAUD")
        elif attack_family == "benign_noise":
            truth.append("NOISE")
        else:
            truth.append(
                "RARITY"
                if _client_is_corner_heavy(client, corner_labels, rarity_threshold)
                else "HONEST"
            )
    return truth


def _build_round_gradients(
    *,
    round_clients: list[RealClient],
    model: nn.Module,
    main_reference: np.ndarray,
    corner_reference: np.ndarray,
    attack_plan: dict[int, str],
    config: RealGradientBenchmarkConfig,
    corner_labels: tuple[int, ...],
    noise_seed: int,
) -> list[ClientGradient]:
    gradients: list[ClientGradient] = []
    for idx, client in enumerate(round_clients):
        ground_truth = (
            "RARITY"
            if _client_is_corner_heavy(
                client,
                corner_labels,
                config.rarity_label_fraction_threshold,
            )
            else "HONEST"
        )
        attack_family = attack_plan.get(idx, "none")
        gradient = compute_client_gradient(model, client, batch_size=config.local_batch_size)
        if attack_family == "sign_flip_proxy":
            gradient = -config.sign_flip_scale * main_reference
            ground_truth = "FRAUD"
        elif attack_family == "corner_harm":
            gradient = 0.35 * main_reference - config.corner_harm_scale * corner_reference
            ground_truth = "FRAUD"
        elif attack_family == "benign_noise":
            noise_rng = np.random.default_rng(noise_seed + idx)
            gradient = gradient + noise_rng.normal(0.0, 0.05, size=gradient.shape)
            ground_truth = "NOISE"

        gradients.append(
            ClientGradient(
                client_id=client.client_id,
                gradient=gradient,
                ground_truth=ground_truth,
                attack_family=attack_family,
                sample_count=client.size,
                label_histogram=client.label_histogram,
            )
        )

    return gradients


def _summarize_classification(
    ground_truth: list[str],
    predicted: list[str],
    target: str,
) -> dict[str, float | int]:
    support = sum(1 for label in ground_truth if label == target)
    predicted_count = sum(1 for label in predicted if label == target)
    true_positive = sum(
        1 for truth, pred in zip(ground_truth, predicted) if truth == target and pred == target
    )
    return {
        "support": support,
        "precision": true_positive / predicted_count if predicted_count else 0.0,
        "recall": true_positive / support if support else 0.0,
    }


def make_real_data_adaptive_policy(base_policy: Policy | None = None) -> Policy:
    """Policy profile calibrated from the current MNIST/Fashion/FEMNIST traces."""

    return (base_policy or DEFAULT_POLICY).model_copy(update=REAL_DATA_ADAPTIVE_POLICY_UPDATES)


def _cornerdrive_l1_router_config(config: RealGradientBenchmarkConfig) -> L1RouterConfig | None:
    if config.cornerdrive_l1_mode == "v25_cosine_fixed":
        return None
    return make_l1_router_config(
        config.cornerdrive_l1_mode,
        norm_mad_threshold=config.cornerdrive_l1_norm_mad_threshold,
        sign_threshold=config.cornerdrive_l1_sign_threshold,
        sign_topk_ratio=config.cornerdrive_l1_sign_topk_ratio,
        queue_budget_ratio=config.cornerdrive_l1_queue_budget_ratio,
        random_recheck_ratio=config.cornerdrive_l1_random_recheck_ratio,
    )


def run_real_gradient_benchmark(
    config: RealGradientBenchmarkConfig | None = None,
    policy: Policy | None = None,
) -> dict[str, Any]:
    config = config or RealGradientBenchmarkConfig()
    policy = policy or DEFAULT_POLICY
    data_bundle = build_real_gradient_data_bundle(config)
    clients = data_bundle.clients
    dataset_info = data_bundle.dataset_info
    rng = random.Random(config.seed)

    input_dim = int(clients[0].inputs.view(clients[0].inputs.size(0), -1).size(1))
    corner_labels = tuple(
        int(label)
        for label in dataset_info.get("corner_labels", list(DEFAULT_CORNER_LABELS))
    )
    output_dim = int(
        max(
            max(int(client.targets.max().item()) for client in clients),
            _max_target_in_dataset(data_bundle.audit_main_dataset),
            _max_target_in_dataset(data_bundle.eval_main_dataset),
        ) + 1
    )
    initial_model = _pretrain_model(
        clients[: max(config.clients_per_round, 8)],
        input_dim=input_dim,
        output_dim=output_dim,
        steps=config.pretrain_steps,
        batch_size=config.local_batch_size,
        seed=config.seed,
    )
    audit_main_dataset = data_bundle.audit_main_dataset
    audit_corner_dataset = data_bundle.audit_corner_dataset
    eval_main_dataset = data_bundle.eval_main_dataset
    eval_corner_dataset = data_bundle.eval_corner_dataset
    auditor = DualChannelAuditor(
        model=copy.deepcopy(initial_model),
        main_dataset=audit_main_dataset,
        corner_dataset=audit_corner_dataset,
    )
    auditor.apply_policy(policy)

    methods = {
        "fedavg": {"label": "FedAvg", "model": copy.deepcopy(initial_model), "rounds": []},
        "geomed": {"label": "GeoMed", "model": copy.deepcopy(initial_model), "rounds": []},
        "krum": {"label": "Multi-Krum", "model": copy.deepcopy(initial_model), "rounds": []},
        "fltrust": {"label": "FLTrust", "model": copy.deepcopy(initial_model), "rounds": []},
        "zeno": {"label": "Zeno", "model": copy.deepcopy(initial_model), "rounds": []},
        "zenopp": {"label": "Zeno++", "model": copy.deepcopy(initial_model), "rounds": []},
        "cornerdrive": {"label": "CornerDrive", "model": copy.deepcopy(initial_model), "rounds": []},
    }
    cornerdrive_router_config = _cornerdrive_l1_router_config(config)
    all_predictions: dict[str, list[str]] = {"cornerdrive": []}
    all_truth: list[str] = []

    for round_index in range(config.rounds):
        round_clients = rng.sample(clients, k=min(config.clients_per_round, len(clients)))
        attack_plan = _build_attack_plan(
            client_count=len(round_clients),
            rng=rng,
            config=config,
        )
        truth = _round_truth(
            round_clients=round_clients,
            attack_plan=attack_plan,
            corner_labels=corner_labels,
            rarity_threshold=config.rarity_label_fraction_threshold,
        )
        vehicle_ids = [
            "0x" + f"{round_index:04x}{idx:036x}"[-40:]
            for idx in range(len(round_clients))
        ]
        all_truth.extend(truth)

        for method_id, method in methods.items():
            model = method["model"]
            before_main = _evaluate_model(model, eval_main_dataset, batch_size=config.local_batch_size)
            before_corner = _evaluate_model(model, eval_corner_dataset, batch_size=config.local_batch_size)
            main_reference = _gradient_for_dataset(
                model,
                audit_main_dataset,
                batch_size=config.local_batch_size,
            )
            corner_reference = _gradient_for_dataset(
                model,
                audit_corner_dataset,
                batch_size=config.local_batch_size,
            )
            round_gradients = _build_round_gradients(
                round_clients=round_clients,
                model=model,
                main_reference=main_reference,
                corner_reference=corner_reference,
                attack_plan=attack_plan,
                config=config,
                corner_labels=corner_labels,
                noise_seed=config.seed + round_index * 1009,
            )
            raw_gradients = [item.gradient for item in round_gradients]

            selected_indices: set[int]
            predicted = ["ACCEPTED" for _ in raw_gradients]
            l1_suspect_total = 0
            l1_router_mode = None
            l1_routing_reasons: dict[str, int] = {}
            if method_id == "fedavg":
                selected_indices = set(range(len(raw_gradients)))
                aggregated = _mean_gradient(raw_gradients)
            elif method_id == "geomed":
                selected_indices = set(range(len(raw_gradients)))
                aggregated, _iterations = geometric_median(raw_gradients)
            elif method_id == "krum":
                selected_indices = set(
                    _multi_krum_indices(
                        raw_gradients,
                        byzantine_budget=_estimated_byzantine_budget(config, len(raw_gradients)),
                    )
                )
                aggregated = _mean_gradient([raw_gradients[idx] for idx in sorted(selected_indices)])
                predicted = [
                    "ACCEPTED" if idx in selected_indices else "REJECTED"
                    for idx in range(len(raw_gradients))
                ]
            elif method_id == "fltrust":
                aggregated, selected_indices, _trust_scores = _fltrust_aggregate(
                    raw_gradients,
                    main_reference,
                )
                predicted = [
                    "ACCEPTED" if idx in selected_indices else "REJECTED"
                    for idx in range(len(raw_gradients))
                ]
            elif method_id == "zeno":
                scores = _zeno_scores(
                    model,
                    raw_gradients,
                    audit_main_dataset,
                    batch_size=config.local_batch_size,
                    learning_rate=L2_LEARNING_RATE,
                    score_penalty=config.zeno_score_penalty,
                )
                aggregated, selected_indices = _zeno_aggregate(
                    raw_gradients,
                    scores,
                    byzantine_budget=_estimated_byzantine_budget(config, len(raw_gradients)),
                )
                predicted = [
                    "ACCEPTED" if idx in selected_indices else "REJECTED"
                    for idx in range(len(raw_gradients))
                ]
            elif method_id == "zenopp":
                scores = _zeno_scores(
                    model,
                    raw_gradients,
                    audit_main_dataset,
                    batch_size=config.local_batch_size,
                    learning_rate=L2_LEARNING_RATE,
                    score_penalty=config.zeno_score_penalty,
                )
                aggregated, selected_indices, _weights = _zenopp_aggregate(
                    raw_gradients,
                    scores,
                    temperature=config.zenopp_score_temperature,
                )
                predicted = [
                    "ACCEPTED" if idx in selected_indices else "REJECTED"
                    for idx in range(len(raw_gradients))
                ]
            else:
                l1 = filter_suspects(
                    raw_gradients,
                    vehicle_ids,
                    threshold=policy.cosine_filter_threshold,
                    recheck_probability=policy.recheck_probability,
                    rng=random.Random(config.seed + round_index),
                    router_config=cornerdrive_router_config,
                    current_round=round_index,
                )
                suspect_indices = set(l1.suspect_indices)
                l1_suspect_total = len(suspect_indices)
                l1_router_mode = l1.router_mode
                l1_routing_reasons = dict(Counter(l1.routing_reasons.values()))
                selected_indices = set(range(len(raw_gradients))) - suspect_indices
                predicted = ["HONEST" for _ in raw_gradients]
                runtime_auditor = DualChannelAuditor(
                    model=copy.deepcopy(model),
                    main_dataset=audit_main_dataset,
                    corner_dataset=audit_corner_dataset,
                )
                runtime_auditor.apply_policy(policy)
                for idx in sorted(suspect_indices):
                    audit = runtime_auditor.audit(vehicle_ids[idx], raw_gradients[idx])
                    predicted[idx] = audit.classification.value
                    if audit.include_in_aggregation:
                        selected_indices.add(idx)
                aggregated = (
                    _mean_gradient([raw_gradients[idx] for idx in sorted(selected_indices)])
                    if selected_indices
                    else np.zeros_like(raw_gradients[0])
                )
                all_predictions["cornerdrive"].extend(predicted)

            method["model"] = _apply_gradient(model, aggregated, L2_LEARNING_RATE)
            after_main = _evaluate_model(method["model"], eval_main_dataset, batch_size=config.local_batch_size)
            after_corner = _evaluate_model(method["model"], eval_corner_dataset, batch_size=config.local_batch_size)
            selected_truth = Counter(truth[idx] for idx in selected_indices)
            fraud_total = max(sum(1 for label in truth if label == "FRAUD"), 1)
            rarity_total = max(sum(1 for label in truth if label == "RARITY"), 1)
            round_record = {
                "round": round_index,
                "main_accuracy": after_main["accuracy"],
                "corner_accuracy": after_corner["accuracy"],
                "main_loss_delta": after_main["loss"] - before_main["loss"],
                "corner_loss_delta": after_corner["loss"] - before_corner["loss"],
                "selected_total": len(selected_indices),
                "selected_fraud": int(selected_truth.get("FRAUD", 0)),
                "selected_rarity": int(selected_truth.get("RARITY", 0)),
                "fraud_survival_rate": selected_truth.get("FRAUD", 0) / fraud_total,
                "rarity_retention_rate": selected_truth.get("RARITY", 0) / rarity_total,
                "truth_counts": dict(Counter(truth)),
                "predicted_counts": dict(Counter(predicted)),
            }
            if method_id == "cornerdrive":
                fraud_family_total = Counter(
                    item.attack_family
                    for item, label in zip(round_gradients, truth)
                    if label == "FRAUD"
                )
                fraud_family_selected = Counter(
                    round_gradients[idx].attack_family
                    for idx in selected_indices
                    if truth[idx] == "FRAUD"
                )
                round_record.update({
                    "l1_router_mode": l1_router_mode,
                    "l1_suspect_total": l1_suspect_total,
                    "l1_review_rate": l1_suspect_total / len(raw_gradients),
                    "l1_routing_reasons": l1_routing_reasons,
                    "fraud_truth_attack_families": dict(fraud_family_total),
                    "selected_fraud_attack_families": dict(fraud_family_selected),
                })
            method["rounds"].append(round_record)

    results_by_method: dict[str, Any] = {}
    for method_id, method in methods.items():
        rows = method["rounds"]
        summary = {
            "main_accuracy_avg": mean(row["main_accuracy"] for row in rows),
            "corner_accuracy_avg": mean(row["corner_accuracy"] for row in rows),
            "fraud_survival_rate_avg": mean(row["fraud_survival_rate"] for row in rows),
            "rarity_retention_rate_avg": mean(row["rarity_retention_rate"] for row in rows),
            "selected_total_avg": mean(row["selected_total"] for row in rows),
        }
        if method_id == "cornerdrive":
            summary["fraud_classification"] = _summarize_classification(
                all_truth,
                all_predictions["cornerdrive"],
                "FRAUD",
            )
            summary["rarity_classification"] = _summarize_classification(
                all_truth,
                all_predictions["cornerdrive"],
                "RARITY",
            )
            summary["l1_suspect_total_avg"] = mean(
                row.get("l1_suspect_total", 0) for row in rows
            )
            summary["l1_review_rate_avg"] = mean(
                row.get("l1_review_rate", 0.0) for row in rows
            )
            fraud_family_total: Counter[str] = Counter()
            fraud_family_selected: Counter[str] = Counter()
            for row in rows:
                fraud_family_total.update(row.get("fraud_truth_attack_families", {}))
                fraud_family_selected.update(row.get("selected_fraud_attack_families", {}))
            summary["fraud_survival_by_attack_family"] = {
                family: fraud_family_selected.get(family, 0) / total
                for family, total in fraud_family_total.items()
                if total
            }
        results_by_method[method_id] = {
            "id": method_id,
            "label": method["label"],
            "summary": summary,
            "round_records": rows,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "real_gradient",
        "dataset": {
            **dataset_info,
            "client_count": len(clients),
            "client_sample_count": sum(client.size for client in clients),
            "client_sample_min": min(client.size for client in clients),
            "client_sample_max": max(client.size for client in clients),
            "client_sample_avg": mean(client.size for client in clients),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "corner_labels": list(corner_labels),
        },
        "config": asdict(config),
        "policy": policy.model_dump(mode="json"),
        "methods": results_by_method,
    }


def write_real_gradient_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "real_gradient_benchmark_summary.json").write_text(
        json.dumps(result, indent=2),
    )

    rows: list[dict[str, Any]] = []
    for method_id, payload in result["methods"].items():
        for row in payload["round_records"]:
            rows.append({"method": method_id, **row})

    if rows:
        with (output_dir / "real_gradient_rounds.csv").open("w", newline="") as handle:
            fieldnames = sorted({key for row in rows for key in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
