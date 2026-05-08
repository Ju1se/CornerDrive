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

The benchmark reports FedAvg, GeoMed, Multi-Krum, and CornerDrive on the same
round schedule. Core metrics are main accuracy, corner accuracy, fraud survival,
rarity retention, and CornerDrive fraud/rarity precision and recall.

LEAF/FEMNIST should be treated as the primary federated-client benchmark because
it keeps client boundaries real. BDD100K should be treated as IoV/domain
calibration evidence because it provides real driving images and scene
attributes, but the client boundaries are pseudo-clients rather than real
vehicles. Torchvision fallback is still useful for development, but its client
partitions are deterministic non-IID shards rather than real users.
