# Real-Data Gradient Benchmark

The original V2.5 exporters intentionally stress controlled archetypes, but
they still generate gradients synthetically. For thesis evidence, the stronger
next benchmark is to derive client gradients from public federated datasets and
from IoV-relevant public driving datasets.

## Dataset Search

Raw public gradient traces are uncommon because gradients can leak training
examples. The practical benchmark route is therefore:

1. load a public federated dataset with real client partitions, or load a
   driving dataset and construct deterministic pseudo-clients from scene
   attributes;
2. freeze a deterministic model checkpoint;
3. compute per-client gradients from each client's real examples;
4. inject only the adversarial transformations needed for security stress
   tests, while keeping honest and rarity gradients data-derived.

Candidate datasets:

| Candidate | Why usable | Notes |
| --- | --- | --- |
| [LEAF/FEMNIST](https://leaf.cmu.edu/) and [TalwalkarLab/leaf](https://github.com/TalwalkarLab/leaf) | Real writer/client partitions and FEMNIST image classification. | Preferred source for this repo. Place processed LEAF JSON under `data/real/femnist`. |
| [TensorFlow Federated EMNIST ClientData](https://www.tensorflow.org/federated/tutorials/working_with_client_data) | Official federated EMNIST client API; clients map to writers. | Good external validation path, but it adds TensorFlow/TFF dependencies. |
| [Flower Datasets FEMNIST](https://flower.ai/docs/datasets/) | Provides `flwrlabs/femnist` and partitioning tools for FL experiments. | Useful if the benchmark later moves to Flower-based orchestration. |
| [BDD100K](https://github.com/bdd100k/bdd100k) | Real driving images with frame attributes such as weather, scene, and time of day. | Best IoV/domain-relevance calibration path. BDD100K has pseudo-clients grouped by attributes, not real vehicle ids. |
| [FedScale](https://github.com/SymbioticLab/FedScale) | Large FL benchmark suite with real-world heterogeneous datasets. | Heavier dependency and runtime surface, better for large-scale follow-up. |

## Exporter

Run the new exporter with LEAF/FEMNIST:

```bash
python scripts/export_real_gradient_benchmark.py \
  --source leaf_femnist \
  --leaf-data-dir data/real/femnist \
  --rounds 8 \
  --clients-per-round 16 \
  --output-dir results/real_gradient_benchmark
```

The exporter defaults to `--policy-profile real_data_adaptive`. This profile
comes from the current MNIST, FashionMNIST, and LEAF/FEMNIST real-gradient
calibration runs. It tightens L2 fraud tolerance, relaxes the rarity threshold
enough to preserve mixed real clients, and routes CornerDrive through L1V3
norm/sign screening:

```bash
python scripts/export_real_gradient_benchmark.py \
  --source leaf_femnist \
  --leaf-data-dir data/real/femnist \
  --policy-profile real_data_adaptive \
  --cornerdrive-l1-mode v3_m2_norm_sign_fixed
```

Use `--policy-profile default --cornerdrive-l1-mode v25_cosine_fixed` to
reproduce the original V2.5 cosine-only CornerDrive behavior.

If LEAF data is not present, a real-sample fallback can derive gradients from
torchvision MNIST or FashionMNIST:

```bash
python scripts/export_real_gradient_benchmark.py \
  --source mnist \
  --download \
  --rounds 8 \
  --clients-per-round 16
```

Run with BDD100K after placing the official image labels and images under
`data/real/bdd100k`:

```bash
python scripts/export_real_gradient_benchmark.py \
  --source bdd100k \
  --bdd-data-dir data/real/bdd100k \
  --bdd-target-attribute weather \
  --bdd-client-group weather_timeofday \
  --bdd-corner-values rainy,snowy,foggy \
  --rounds 8 \
  --clients-per-round 16 \
  --output-dir results/real_gradient_benchmark_bdd100k
```

Common explicit BDD100K layout override:

```bash
python scripts/export_real_gradient_benchmark.py \
  --source bdd100k \
  --bdd-label-file data/real/bdd100k/labels/bdd100k_labels_images_train.json \
  --bdd-image-dir data/real/bdd100k/images/100k/train
```

Outputs:

- `real_gradient_benchmark_summary.json`
- `real_gradient_rounds.csv`

## Interpretation

The benchmark reports FedAvg, GeoMed, Multi-Krum, FLTrust, Zeno, Zeno++, and
CornerDrive on the same round schedule. FLTrust uses a root/reference gradient
from the validation slice, Zeno uses validation-loss descent scoring, and
Zeno++ uses the same score as a synchronous score-weighted benchmark variant.
Core metrics are main accuracy, corner accuracy, fraud survival, rarity
retention, and CornerDrive fraud/rarity precision and recall.

For CornerDrive, the round records also include L1 diagnostics:
`l1_router_mode`, `l1_suspect_total`, `l1_review_rate`, `l1_routing_reasons`,
and fraud survival split by attack family. These fields are important for real
data because stealthy sign-flip proxy gradients can sit inside the cosine-only
clean region while still being caught by norm/sign-assisted L1V3 routing.

Current local calibration on MNIST, FashionMNIST, and LEAF/FEMNIST shows why
the adaptive profile is the default for real data:

| CornerDrive profile | Main acc | Corner acc | Fraud survival | Rarity retention |
| --- | ---: | ---: | ---: | ---: |
| Default V2.5 cosine-only | 0.4475 | 0.4723 | 0.6250 | 0.8545 |
| Tuned thresholds, V2.5 L1 | 0.4605 | 0.5495 | 0.3646 | 0.8663 |
| Real-data adaptive L1V3 | 0.4783 | 0.5918 | 0.0521 | 0.8402 |

The trade-off is audit cost: the adaptive profile routes about 80-86% of each
16-client round through L2. This is intentional for the thesis benchmark, where
the goal is evidence-backed fraud suppression on real non-IID gradients; a
larger deployment can lower review coverage with `v3_m3_budgeted` once the
fraud-pressure target is fixed.

`--rarity-label-fraction-threshold` controls when a real client is treated as
corner/rarity-heavy. The default `0.30` is intentionally below a strict
majority so deterministic two-label torchvision shards and mixed FEMNIST writer
clients can contribute rarity-retention evidence.

LEAF/FEMNIST should be treated as the primary federated-client benchmark because
it keeps client boundaries real. BDD100K should be treated as IoV/domain
calibration evidence because it provides real driving images and scene
attributes, but the client boundaries are pseudo-clients rather than real
vehicles. Torchvision fallback is still useful for development, but its client
partitions are deterministic non-IID shards rather than real users.
