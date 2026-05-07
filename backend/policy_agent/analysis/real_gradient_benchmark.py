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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

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
from l2_dual_audit.classifier import DualChannelAuditor  # noqa: E402


DEFAULT_CORNER_LABELS = (1, 7, 9)


@dataclass(frozen=True)
class RealClient:
    client_id: str
    inputs: torch.Tensor
    targets: torch.Tensor

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
class RealGradientBenchmarkConfig:
    source: str = "auto"
    leaf_data_dir: str = "data/real/femnist"
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
    attack_fraction: float = 0.20
    corner_harm_fraction: float = 0.05
    noise_fraction: float = 0.05
    sign_flip_scale: float = 3.0
    corner_harm_scale: float = 2.0


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


def _leaf_json_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for pattern in ("**/all_data*.json", "**/*train*.json", "**/*test*.json"):
        candidates.extend(root.glob(pattern))
    return sorted(set(candidates))


def load_leaf_femnist_clients(
    root: Path,
    *,
    max_clients: int,
    min_samples_per_client: int,
    max_samples_per_client: int,
) -> tuple[list[RealClient], dict[str, Any]]:
    """Load LEAF/FEMNIST processed JSON clients."""
    if not root.exists():
        raise FileNotFoundError(f"LEAF data directory does not exist: {root}")

    clients: list[RealClient] = []
    for json_file in _leaf_json_files(root):
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
                    "files_scanned": len(_leaf_json_files(root)),
                    "real_client_partitions": True,
                }

    if not clients:
        raise ValueError(f"No usable LEAF clients found under {root}")

    return clients, {
        "source": "leaf_femnist",
        "root": str(root),
        "files_scanned": len(_leaf_json_files(root)),
        "real_client_partitions": True,
    }


def _load_torchvision_dataset(name: str, root: Path, download: bool):
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
        return datasets.MNIST(root=str(root), train=True, transform=transform, download=download)
    if dataset_name in {"fashionmnist", "fashion_mnist", "torchvision_fashionmnist"}:
        return datasets.FashionMNIST(root=str(root), train=True, transform=transform, download=download)
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
    dataset = _load_torchvision_dataset(name, root, download)
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
    }


def load_real_clients(config: RealGradientBenchmarkConfig) -> tuple[list[RealClient], dict[str, Any]]:
    source = config.source.lower()
    leaf_root = PROJECT_ROOT / config.leaf_data_dir
    data_root = PROJECT_ROOT / config.data_dir

    if source in {"auto", "leaf", "leaf_femnist", "femnist"} and leaf_root.exists():
        try:
            return load_leaf_femnist_clients(
                leaf_root,
                max_clients=config.max_clients,
                min_samples_per_client=config.min_samples_per_client,
                max_samples_per_client=config.max_samples_per_client,
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


def _client_is_corner_heavy(client: RealClient, corner_labels: tuple[int, ...]) -> bool:
    labels = client.targets.tolist()
    if not labels:
        return False
    corner_count = sum(1 for label in labels if int(label) in corner_labels)
    return corner_count / len(labels) >= 0.60


def _build_round_gradients(
    *,
    round_clients: list[RealClient],
    model: nn.Module,
    main_reference: np.ndarray,
    corner_reference: np.ndarray,
    rng: random.Random,
    config: RealGradientBenchmarkConfig,
) -> list[ClientGradient]:
    gradients: list[ClientGradient] = []
    for client in round_clients:
        ground_truth = "RARITY" if _client_is_corner_heavy(client, DEFAULT_CORNER_LABELS) else "HONEST"
        gradients.append(
            ClientGradient(
                client_id=client.client_id,
                gradient=compute_client_gradient(model, client, batch_size=config.local_batch_size),
                ground_truth=ground_truth,
                attack_family="none",
                sample_count=client.size,
                label_histogram=client.label_histogram,
            )
        )

    indices = list(range(len(gradients)))
    rng.shuffle(indices)
    sign_flip_count = int(round(len(gradients) * config.attack_fraction))
    corner_harm_count = int(round(len(gradients) * config.corner_harm_fraction))
    noise_count = int(round(len(gradients) * config.noise_fraction))

    for idx in indices[:sign_flip_count]:
        item = gradients[idx]
        gradients[idx] = ClientGradient(
            client_id=item.client_id,
            gradient=-config.sign_flip_scale * main_reference,
            ground_truth="FRAUD",
            attack_family="sign_flip_proxy",
            sample_count=item.sample_count,
            label_histogram=item.label_histogram,
        )

    start = sign_flip_count
    for idx in indices[start:start + corner_harm_count]:
        item = gradients[idx]
        harmful = 0.35 * main_reference - config.corner_harm_scale * corner_reference
        gradients[idx] = ClientGradient(
            client_id=item.client_id,
            gradient=harmful,
            ground_truth="FRAUD",
            attack_family="corner_harm",
            sample_count=item.sample_count,
            label_histogram=item.label_histogram,
        )

    start += corner_harm_count
    for idx in indices[start:start + noise_count]:
        item = gradients[idx]
        noise_rng = np.random.default_rng(config.seed + idx)
        noise = noise_rng.normal(0.0, 0.05, size=item.gradient.shape)
        gradients[idx] = ClientGradient(
            client_id=item.client_id,
            gradient=item.gradient + noise,
            ground_truth="NOISE",
            attack_family="benign_noise",
            sample_count=item.sample_count,
            label_histogram=item.label_histogram,
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


def run_real_gradient_benchmark(
    config: RealGradientBenchmarkConfig | None = None,
    policy: Policy | None = None,
) -> dict[str, Any]:
    config = config or RealGradientBenchmarkConfig()
    policy = policy or DEFAULT_POLICY
    clients, dataset_info = load_real_clients(config)
    rng = random.Random(config.seed)

    input_dim = int(clients[0].inputs.view(clients[0].inputs.size(0), -1).size(1))
    output_dim = int(max(int(client.targets.max().item()) for client in clients) + 1)
    initial_model = _pretrain_model(
        clients[: max(config.clients_per_round, 8)],
        input_dim=input_dim,
        output_dim=output_dim,
        steps=config.pretrain_steps,
        batch_size=config.local_batch_size,
        seed=config.seed,
    )
    main_dataset, corner_dataset = _split_reference_clients(clients)
    main_reference = _gradient_for_dataset(
        initial_model,
        main_dataset,
        batch_size=config.local_batch_size,
    )
    corner_reference = _gradient_for_dataset(
        initial_model,
        corner_dataset,
        batch_size=config.local_batch_size,
    )
    auditor = DualChannelAuditor(
        model=copy.deepcopy(initial_model),
        main_dataset=main_dataset,
        corner_dataset=corner_dataset,
    )
    auditor.apply_policy(policy)

    methods = {
        "fedavg": {"label": "FedAvg", "model": copy.deepcopy(initial_model), "rounds": []},
        "geomed": {"label": "GeoMed", "model": copy.deepcopy(initial_model), "rounds": []},
        "krum": {"label": "Multi-Krum", "model": copy.deepcopy(initial_model), "rounds": []},
        "cornerdrive": {"label": "CornerDrive", "model": copy.deepcopy(initial_model), "rounds": []},
    }
    all_predictions: dict[str, list[str]] = {"cornerdrive": []}
    all_truth: list[str] = []

    for round_index in range(config.rounds):
        round_clients = rng.sample(clients, k=min(config.clients_per_round, len(clients)))
        round_gradients = _build_round_gradients(
            round_clients=round_clients,
            model=initial_model,
            main_reference=main_reference,
            corner_reference=corner_reference,
            rng=rng,
            config=config,
        )
        raw_gradients = [item.gradient for item in round_gradients]
        vehicle_ids = [
            "0x" + f"{round_index:04x}{idx:036x}"[-40:]
            for idx in range(len(round_gradients))
        ]
        truth = [item.ground_truth for item in round_gradients]
        all_truth.extend(truth)

        for method_id, method in methods.items():
            model = method["model"]
            before_main = _evaluate_model(model, main_dataset, batch_size=config.local_batch_size)
            before_corner = _evaluate_model(model, corner_dataset, batch_size=config.local_batch_size)

            selected_indices: set[int]
            predicted = ["ACCEPTED" for _ in raw_gradients]
            if method_id == "fedavg":
                selected_indices = set(range(len(raw_gradients)))
                aggregated = _mean_gradient(raw_gradients)
            elif method_id == "geomed":
                selected_indices = set(range(len(raw_gradients)))
                aggregated, _iterations = geometric_median(raw_gradients)
            elif method_id == "krum":
                selected_indices = set(_multi_krum_indices(raw_gradients))
                aggregated = _mean_gradient([raw_gradients[idx] for idx in sorted(selected_indices)])
            else:
                l1 = filter_suspects(
                    raw_gradients,
                    vehicle_ids,
                    threshold=policy.cosine_filter_threshold,
                    recheck_probability=policy.recheck_probability,
                    rng=random.Random(config.seed + round_index),
                )
                suspect_indices = set(l1.suspect_indices)
                selected_indices = set(range(len(raw_gradients))) - suspect_indices
                predicted = ["HONEST" for _ in raw_gradients]
                runtime_auditor = DualChannelAuditor(
                    model=copy.deepcopy(model),
                    main_dataset=main_dataset,
                    corner_dataset=corner_dataset,
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
            after_main = _evaluate_model(method["model"], main_dataset, batch_size=config.local_batch_size)
            after_corner = _evaluate_model(method["model"], corner_dataset, batch_size=config.local_batch_size)
            selected_truth = Counter(truth[idx] for idx in selected_indices)
            fraud_total = max(sum(1 for label in truth if label == "FRAUD"), 1)
            rarity_total = max(sum(1 for label in truth if label == "RARITY"), 1)
            method["rounds"].append({
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
            })

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
            "input_dim": input_dim,
            "output_dim": output_dim,
            "corner_labels": list(DEFAULT_CORNER_LABELS),
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
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
