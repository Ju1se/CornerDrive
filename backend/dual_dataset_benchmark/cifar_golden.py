"""CIFAR-10 golden-dataset annotation and the federation built on the rest.

Split of the 60,000 CIFAR-10 images (official train and test pooled),
stratified by class with a fixed annotation seed:

    golden  1/4  (1,500 per class, 15,000 images)  server-held, annotated
        golden_normal   classes outside RARE_LABELS   -> D_main   (10,500)
        golden_rare     classes in RARE_LABELS        -> D_corner  (4,500)
    test    3/4  (4,500 per class, 45,000 images)  never annotated
        client_pool     4,000 per class   client shards that produce the
                                           updates under test
        eval            500 per class     held-out accuracy read-out

The server only ever sees the golden quarter. Rarity ground truth on the
test three quarters is used for scoring the detectors, never for a decision.

The rare classes mirror ``DEFAULT_CORNER_LABELS = (1, 7, 9)`` of
``policy_agent.analysis.real_gradient_benchmark``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
NUM_CLASSES = 10
RARE_LABELS = (1, 7, 9)
NORMAL_LABELS = tuple(k for k in range(NUM_CLASSES) if k not in RARE_LABELS)

ANNOTATION_SEED = 20261123
GOLDEN_FRACTION = 0.25
EVAL_PER_CLASS = 500

# md5 of the official python-version batches (same values torchvision checks).
CIFAR10_MD5 = {
    "data_batch_1": "c99cafc152244af753f735de768cd75f",
    "data_batch_2": "d4bba439e000b95fd0a9bffe97cbabec",
    "data_batch_3": "54ebc095f3ab1f0389bbae665268c751",
    "data_batch_4": "634d18415352ddfa80567beed471001a",
    "data_batch_5": "482c414d41f54cd18b22e5b47cb7c3cb",
    "test_batch": "40351d587109b95175f43aff81a1287e",
}
TRAIN_FILES = tuple(f"data_batch_{i}" for i in range(1, 6))

# Data-poisoning maps used by the attacker roles.
LABEL_FLIP_MAP = tuple(9 - k for k in range(NUM_CLASSES))        # y -> 9 - y
CORNER_HARM_MAP = {1: 8, 7: 4, 9: 0}    # automobile->ship, horse->deer, truck->airplane

ROLE_TRUTH = {
    "honest": "HONEST",
    "rarity": "RARITY",
    "noise": "NOISE",
    "sign_flip": "FRAUD",
    "label_flip": "FRAUD",
    "corner_harm": "FRAUD",
}
ROLES = tuple(ROLE_TRUTH)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cifar10(root: Path, *, verify: bool = True) -> dict[str, Any]:
    """Load all 60,000 images as one pool: uint8 [60000, 3, 32, 32] + labels.

    Pool index 0..49999 is the official train split in file order, 50000..59999
    the official test split. Files are checked against the official md5 before
    they are unpickled.
    """
    root = Path(root)
    images, labels, md5s = [], [], {}
    for name in (*TRAIN_FILES, "test_batch"):
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"CIFAR-10 batch not found: {path}")
        md5s[name] = _md5(path)
        if verify and md5s[name] != CIFAR10_MD5[name]:
            raise ValueError(f"md5 mismatch for {path}: {md5s[name]}")
        with path.open("rb") as handle:
            batch = pickle.load(handle, encoding="latin1")
        images.append(np.asarray(batch["data"], dtype=np.uint8).reshape(-1, 3, 32, 32))
        labels.append(np.asarray(batch["labels"], dtype=np.int64))
    x = np.concatenate(images)
    y = np.concatenate(labels)
    source_split = np.array(["train"] * 50000 + ["test"] * 10000)
    source_index = np.concatenate([np.arange(50000), np.arange(10000)])
    return {"x": x, "y": y, "source_split": source_split,
            "source_index": source_index, "md5": md5s}


# --------------------------------------------------------------------------
# golden annotation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GoldenAnnotation:
    """Pool indices of every surface."""

    golden_normal: np.ndarray       # D_main
    golden_rare: np.ndarray         # D_corner
    client_pool: np.ndarray
    eval_normal: np.ndarray
    eval_rare: np.ndarray
    seed: int
    golden_fraction: float

    @property
    def golden(self) -> np.ndarray:
        return np.concatenate([self.golden_normal, self.golden_rare])

    @property
    def eval(self) -> np.ndarray:
        return np.concatenate([self.eval_normal, self.eval_rare])


def annotate_golden(
    y: np.ndarray,
    *,
    seed: int = ANNOTATION_SEED,
    golden_fraction: float = GOLDEN_FRACTION,
    eval_per_class: int = EVAL_PER_CLASS,
    rare_labels: tuple[int, ...] = RARE_LABELS,
) -> GoldenAnnotation:
    """Stratified 1/4 golden vs 3/4 test split, then normal/rare tags on golden."""
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[np.ndarray]] = {
        name: [] for name in
        ("golden_normal", "golden_rare", "client_pool", "eval_normal", "eval_rare")
    }
    for k in range(NUM_CLASSES):
        idx = rng.permutation(np.flatnonzero(y == k))
        n_golden = int(round(len(idx) * golden_fraction))
        golden, rest = idx[:n_golden], idx[n_golden:]
        held_out, pool = rest[:eval_per_class], rest[eval_per_class:]
        rare = k in rare_labels
        buckets["golden_rare" if rare else "golden_normal"].append(golden)
        buckets["eval_rare" if rare else "eval_normal"].append(held_out)
        buckets["client_pool"].append(pool)
    merged = {name: np.sort(np.concatenate(parts)) for name, parts in buckets.items()}
    return GoldenAnnotation(seed=seed, golden_fraction=golden_fraction, **merged)


def write_manifest(data: dict[str, Any], ann: GoldenAnnotation, out_dir: Path) -> dict[str, Any]:
    """Write manifest.csv (one row per image) and summary.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y = data["y"]
    n = len(y)
    partition = np.full(n, "", dtype=object)
    golden_label = np.full(n, "", dtype=object)
    test_role = np.full(n, "", dtype=object)
    partition[ann.golden] = "golden"
    golden_label[ann.golden_normal] = "normal"
    golden_label[ann.golden_rare] = "rare"
    partition[ann.client_pool] = "test"
    test_role[ann.client_pool] = "client_pool"
    partition[ann.eval] = "test"
    test_role[ann.eval] = "eval"
    if (partition == "").any():
        raise RuntimeError("annotation does not cover every image")

    with (out_dir / "manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pool_index", "source_split", "source_index", "label",
                         "class_name", "partition", "golden_label", "test_role"])
        for i in range(n):
            writer.writerow([i, data["source_split"][i], int(data["source_index"][i]),
                             int(y[i]), CLASS_NAMES[int(y[i])], partition[i],
                             golden_label[i], test_role[i]])

    def per_class(indices: np.ndarray) -> dict[str, int]:
        counts = np.bincount(y[indices], minlength=NUM_CLASSES)
        return {CLASS_NAMES[k]: int(counts[k]) for k in range(NUM_CLASSES) if counts[k]}

    summary = {
        "dataset": "CIFAR-10 (official train + test pooled, 60,000 images)",
        "annotation_seed": ann.seed,
        "golden_fraction": ann.golden_fraction,
        "rare_labels": {int(k): CLASS_NAMES[k] for k in RARE_LABELS},
        "normal_labels": {int(k): CLASS_NAMES[k] for k in NORMAL_LABELS},
        "rule": (
            "Stratified by class. A quarter of every class is annotated as the "
            "golden dataset: images of a rare class are tagged 'rare' (D_corner), "
            "all others 'normal' (D_main). The other three quarters are left "
            "unannotated and used for testing: client shards that produce the "
            "updates under test, and a held-out evaluation set."
        ),
        "counts": {
            "golden_total": int(len(ann.golden)),
            "golden_normal (D_main)": int(len(ann.golden_normal)),
            "golden_rare (D_corner)": int(len(ann.golden_rare)),
            "test_total": int(len(ann.client_pool) + len(ann.eval)),
            "test_client_pool": int(len(ann.client_pool)),
            "test_eval_normal": int(len(ann.eval_normal)),
            "test_eval_rare": int(len(ann.eval_rare)),
        },
        "per_class": {
            "golden_normal": per_class(ann.golden_normal),
            "golden_rare": per_class(ann.golden_rare),
            "client_pool": per_class(ann.client_pool),
            "eval": per_class(ann.eval),
        },
        "source_md5": data["md5"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# --------------------------------------------------------------------------
# federation on the unannotated three quarters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FederationConfig:
    """Client population. Role counts follow the thesis mix (Table A.1):
    main-hostile fraud 0.20, corner-harm 0.05, noise 0.05; rare-rich honest 0.10."""

    num_clients: int = 120
    samples_per_client: int = 224
    n_rarity: int = 12
    n_corner_harm: int = 6
    n_sign_flip: int = 12
    n_label_flip: int = 12
    n_noise: int = 6
    rare_rich_share: float = 0.60        # rare images inside a rare-rich shard
    ordinary_rare_share: float = 0.02    # rare images inside every other shard
    dirichlet_alpha_normal: float = 0.5  # label skew over the normal classes
    dirichlet_alpha_rare: float = 1.0    # label skew over the rare classes
    rarity_truth_threshold: float = 0.30  # thesis rule: >= 30% corner-label samples
    scenario: str = "mixed"              # "mixed" | "no_attack"


@dataclass(frozen=True)
class Federation:
    client_index: np.ndarray      # [N, n] pool indices
    true_labels: np.ndarray       # [N, n]
    train_labels: np.ndarray      # [N, n] after data poisoning
    roles: tuple[str, ...]        # behaviour of every client
    truth: tuple[str, ...]        # HONEST | RARITY | NOISE | FRAUD
    rare_share: np.ndarray        # [N] fraction of rare-class images
    config: FederationConfig

    def describe(self) -> dict[str, Any]:
        roles = np.array(self.roles)
        return {
            "config": asdict(self.config),
            "role_counts": {r: int((roles == r).sum()) for r in ROLES},
            "rare_share_by_role": {
                r: float(self.rare_share[roles == r].mean()) for r in ROLES if (roles == r).any()
            },
            "pool_rare_share_true_labels": float(
                np.isin(self.true_labels, RARE_LABELS).mean()
            ),
        }


def _take(queues: dict[int, np.ndarray], cursor: dict[int, int], group: tuple[int, ...],
          counts: np.ndarray) -> np.ndarray:
    """Pop ``counts[j]`` images of class ``group[j]``; a class that runs dry
    passes its deficit to the class of the same group with the most left."""
    counts = counts.astype(np.int64).copy()
    left = np.array([len(queues[k]) - cursor[k] for k in group])
    for _ in range(len(group)):
        deficit = np.maximum(counts - left, 0)
        if not deficit.any():
            break
        counts -= deficit
        spare = left - counts
        counts[int(np.argmax(spare))] += int(deficit.sum())
    if (counts > left).any():
        raise RuntimeError("client pool exhausted; lower num_clients or samples_per_client")
    picked = []
    for j, k in enumerate(group):
        picked.append(queues[k][cursor[k]:cursor[k] + counts[j]])
        cursor[k] += int(counts[j])
    return np.concatenate(picked)


def build_federation(
    y: np.ndarray,
    client_pool: np.ndarray,
    *,
    seed: int,
    config: FederationConfig | None = None,
) -> Federation:
    """Deterministic long-tailed, non-IID client shards with fixed roles.

    The shards and the role of every client depend on ``seed`` only, so the
    "mixed" and "no_attack" scenarios of one seed are paired: same data, same
    clients, and only the behaviour of the attacker roles differs.
    """
    cfg = config or FederationConfig()
    rng = np.random.default_rng(seed)
    n_clients, n = cfg.num_clients, cfg.samples_per_client
    special = cfg.n_rarity + cfg.n_corner_harm + cfg.n_sign_flip + cfg.n_label_flip + cfg.n_noise
    if special > n_clients:
        raise ValueError("role counts exceed num_clients")

    order = rng.permutation(n_clients)
    base_roles = np.full(n_clients, "honest", dtype=object)
    start = 0
    for role, count in (("rarity", cfg.n_rarity), ("corner_harm", cfg.n_corner_harm),
                        ("sign_flip", cfg.n_sign_flip), ("label_flip", cfg.n_label_flip),
                        ("noise", cfg.n_noise)):
        base_roles[order[start:start + count]] = role
        start += count
    rare_rich = np.isin(base_roles, ("rarity", "corner_harm"))

    pool_y = y[client_pool]
    queues = {k: rng.permutation(client_pool[pool_y == k]) for k in range(NUM_CLASSES)}
    cursor = {k: 0 for k in range(NUM_CLASSES)}

    client_index = np.zeros((n_clients, n), dtype=np.int64)
    for c in range(n_clients):
        share = cfg.rare_rich_share if rare_rich[c] else cfg.ordinary_rare_share
        n_rare = int(round(n * share))
        p_rare = (rng.dirichlet([cfg.dirichlet_alpha_rare] * len(RARE_LABELS))
                  if rare_rich[c] else np.full(len(RARE_LABELS), 1.0 / len(RARE_LABELS)))
        p_normal = rng.dirichlet([cfg.dirichlet_alpha_normal] * len(NORMAL_LABELS))
        rare_idx = _take(queues, cursor, RARE_LABELS, rng.multinomial(n_rare, p_rare))
        normal_idx = _take(queues, cursor, NORMAL_LABELS, rng.multinomial(n - n_rare, p_normal))
        client_index[c] = rng.permutation(np.concatenate([rare_idx, normal_idx]))

    if cfg.scenario == "mixed":
        roles = base_roles.copy()
    elif cfg.scenario == "no_attack":
        roles = np.where(base_roles == "rarity", "rarity", "honest").astype(object)
        roles[base_roles == "corner_harm"] = "rarity"   # same rare-rich data, now honest
    else:
        raise ValueError(f"unknown scenario {cfg.scenario!r}")

    true_labels = y[client_index]
    train_labels = true_labels.copy()
    flip = np.array(LABEL_FLIP_MAP)
    harm = np.arange(NUM_CLASSES)
    for src, dst in CORNER_HARM_MAP.items():
        harm[src] = dst
    for c in range(n_clients):
        if roles[c] == "label_flip":
            train_labels[c] = flip[true_labels[c]]
        elif roles[c] == "corner_harm":
            train_labels[c] = harm[true_labels[c]]

    rare_share = np.isin(true_labels, RARE_LABELS).mean(axis=1)
    truth = []
    for c in range(n_clients):
        label = ROLE_TRUTH[str(roles[c])]
        # thesis rule: an honest shard is Rarity iff it is corner-heavy
        if roles[c] == "rarity" and rare_share[c] < cfg.rarity_truth_threshold:
            label = "HONEST"
        truth.append(label)
    return Federation(
        client_index=client_index,
        true_labels=true_labels,
        train_labels=train_labels,
        roles=tuple(str(r) for r in roles),
        truth=tuple(truth),
        rare_share=rare_share,
        config=cfg,
    )
