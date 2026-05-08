# Real-Data Gradient Benchmark

The original V2.5 exporters intentionally stress controlled archetypes, but
they still generate gradients synthetically. For thesis evidence, the stronger
next benchmark is to derive client gradients from public federated datasets.

## Dataset Search

Raw public gradient traces are uncommon because gradients can leak training
examples. The practical benchmark route is therefore:

1. load a public federated dataset with real client partitions;
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

Outputs:

- `real_gradient_benchmark_summary.json`
- `real_gradient_rounds.csv`

## Interpretation

The benchmark reports FedAvg, GeoMed, Multi-Krum, and CornerDrive on the same
round schedule. Core metrics are main accuracy, corner accuracy, fraud survival,
rarity retention, and CornerDrive fraud/rarity precision and recall.

LEAF/FEMNIST should be treated as the primary thesis benchmark because it keeps
client boundaries real. Torchvision fallback is still useful for development,
but its client partitions are deterministic non-IID shards rather than real
users.
