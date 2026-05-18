# Real-Data Gradient Benchmark

The original synthetic ALG exporters intentionally stress controlled archetypes, but
they still generate gradients synthetically. For thesis evidence, the stronger
next benchmark is to derive client gradients from public federated datasets and
from IoV-relevant public driving datasets.

## Dataset Search

Raw public gradient traces are uncommon because gradients can leak training
examples. The practical benchmark route is therefore:

1. load a public federated dataset with real client partitions, or load a
   driving dataset and construct deterministic pseudo-clients from scene
   attributes;
2. keep client/update samples separate from server-side audit/reference samples
   and final evaluation samples;
3. start from a deterministic model checkpoint;
4. compute per-client gradients from each client's real examples;
5. inject only the adversarial transformations needed for security stress
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

The exporter defaults to `--policy-profile real_gradient_calibrated`. This profile
comes from the current MNIST, FashionMNIST, and LEAF/FEMNIST real-gradient
calibration runs. It tightens L2 fraud tolerance, relaxes the rarity threshold
enough to preserve mixed real clients, and routes CornerDrive through calibrated
dual-proxy L1 screening:

Current calibrated values:

- `theta_tol = 0.02`
- `theta_rare = -0.005`
- `theta_rarity_main_tol = 0.00925`
- `cosine_filter_threshold = 0.50`
- `recheck_probability = 0.25`
- `cornerdrive_l1_mode = dual_proxy_budgeted`
- `cornerdrive_l1_queue_budget_ratio = 0.80`
- `cornerdrive_l1_random_recheck_ratio = 0.05`
- `norm_mad_threshold = 1.5`
- `sign_threshold = 0.40`
- risk weights: cosine `0.35`, norm `0.20`, sign `0.15`
- dual-proxy route actions: `SAFE_ACCEPT`, `AUDIT`, `QUARANTINE`, `LOW_WEIGHT`

```bash
python scripts/export_real_gradient_benchmark.py \
  --source leaf_femnist \
  --leaf-data-dir data/real/femnist \
  --policy-profile real_gradient_calibrated
```

Use `--policy-profile default --cornerdrive-l1-mode cosine_recheck` to
reproduce the original synthetic ALG cosine-only CornerDrive behavior.

The calibrated profile changes clean RARITY from `delta_main <= theta_tol` to
`delta_main <= theta_rarity_main_tol`. It also adds cheap first-order
main/corner loss-drift proxies at L1:

- `pred_delta_main ~= -eta * <grad_main_val, grad_client>`
- `pred_delta_corner ~= -eta * <grad_corner_val, grad_client>`
- weighted aggregation fields: `effective_fraud_mass_survival` and
  `effective_rarity_mass_retention`
- diagnostic fields: `l1_fraud_recall`,
  `l2_fraud_reject_rate_given_routed`, `fraud_survival_unrouted`, and
  `fraud_survival_l2_accepted`

The calibrated `theta_rarity_main_tol = 0.00925` treats updates that improve
corner loss while introducing positive main-task drift above the stricter band
as conflict or noise rather than clean rarity.
The value was selected by a 20-seed threshold sweep as the largest tested
zero-fraud setting before the FashionMNIST boundary case reappeared.
The frozen calibration record is tracked in
`configs/real_gradient_calibration_manifest.json`; its calibration seeds and
final held-out seeds are disjoint.

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

## Larger Reliability Runs

For thesis-grade evidence, use the reliability exporter instead of relying on
one seed. The default real-data profile now matches the thesis reproducibility
manifest: 120 clients, 48 samples per client, 20 clients per round, 10 rounds,
and 20 held-out seeds per dataset. It exports per-run metrics plus mean, standard
deviation, and 95% confidence intervals:

```bash
python scripts/export_real_gradient_reliability_benchmark.py \
  --sources mnist,fashionmnist,femnist \
  --seeds 20260527,20260528,20260529,20260530,20260531,20260532,20260533,20260534,20260535,20260536,20260537,20260538,20260539,20260540,20260541,20260542,20260543,20260544,20260545,20260546 \
  --max-clients 120 \
  --max-samples-per-client 48 \
  --clients-per-round 20 \
  --rounds 10 \
  --pretrain-steps 50 \
  --reference-split-fraction 0.50 \
  --max-reference-samples 4096 \
  --max-evaluation-samples 4096 \
  --output-dir results/real_gradient_reliability_calibrated_holdout
```

Outputs:

- `real_gradient_reliability_runs.csv`
- `real_gradient_reliability_summary.csv`
- `real_gradient_reliability_summary.json`
- per-run `real_gradient_benchmark_summary.json` and `real_gradient_rounds.csv`
  under `runs/<source>/seed_<seed>/`

The current local expanded run uses leakage-safe split surfaces. MNIST and
FashionMNIST use training samples for pseudo-client gradients and split the
official test data into audit/reference and final evaluation subsets.
LEAF/FEMNIST uses `train/` shards for client gradients and `test/` shards for
audit/reference and final evaluation clients. The final held-out calibrated run uses
seeds `20260527` through `20260546`, which raises the evidence from 128
client-round observations per dataset in the original single-seed run to 4,000
observations per dataset:

| Dataset | CornerDrive main acc | Corner acc | Fraud survival | Rarity retention | L1 review |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | 0.7575 +/- 0.0086 | 0.8313 +/- 0.0207 | 0.0010 +/- 0.0020 | 0.8634 +/- 0.0422 | 0.8500 +/- 0.0000 |
| FashionMNIST | 0.5986 +/- 0.0111 | 0.9127 +/- 0.0048 | 0.0030 +/- 0.0059 | 0.1879 +/- 0.0365 | 0.8500 +/- 0.0000 |
| LEAF/FEMNIST | 0.0827 +/- 0.0082 | 0.3949 +/- 0.0431 | 0.0000 +/- 0.0000 | 0.0788 +/- 0.0367 | 0.8500 +/- 0.0000 |

Across all three datasets, this completed held-out run covers 12,000
client-round observations, including 3,000 fraud observations and 3,782 rarity
observations. With the calibrated dual-proxy router, CornerDrive averages 0.0013 fraud
survival, 0.0011 effective fraud-mass survival, 0.3767 rarity retention, 0.7130
corner accuracy, and 0.8500 L1 review coverage under the held-out evaluation
protocol.

The same held-out seeds produce this macro baseline comparison:

| Method | Main acc | Corner acc | Fraud survival | Rarity retention | Selected clients |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multi-Krum | 0.4637 | 0.6283 | 0.4660 | 0.7608 | 13.00 |
| FLTrust | 0.4923 | 0.6301 | 0.0717 | 0.5388 | 10.26 |
| Zeno | 0.5009 | 0.6713 | 0.1820 | 0.9430 | 15.00 |
| Zeno++ | 0.5031 | 0.6608 | 0.0007 | 0.2318 | 4.73 |
| CornerDrive calibrated | 0.4796 | 0.7130 | 0.0013 | 0.3767 | 3.58 |

## Interpretation

The benchmark reports FedAvg, GeoMed, Multi-Krum, FLTrust, Zeno, Zeno++, and
CornerDrive on the same round schedule. Every method receives the same sampled
clients and attack schedule in a given round, but each method maintains its own
model state; gradients in later rounds are therefore computed against that
method's current checkpoint. FLTrust uses a root/reference gradient from the
validation slice, Zeno uses validation-loss descent scoring, and Zeno++ uses the
same score as a synchronous score-weighted benchmark variant. Multi-Krum uses
the configured attack fractions to estimate its Byzantine budget. These are
controlled benchmark implementations of the baseline principles rather than
full framework reproductions of every original paper. Core metrics are main
accuracy, corner accuracy, fraud survival, rarity retention, and CornerDrive
fraud/rarity precision and recall.

For CornerDrive, the round records also include L1 diagnostics:
`l1_router_mode`, `l1_suspect_total`, `l1_review_rate`, `l1_routing_reasons`,
and fraud survival split by attack family. These fields are important for real
data because stealthy sign-flip proxy gradients can sit inside the cosine-only
clean region while still being caught by calibrated dual-proxy routing and L2
rarity-safety checks.

Held-out calibration on MNIST, FashionMNIST, and LEAF/FEMNIST selects the
calibrated profile as the frozen real-gradient setting:

| CornerDrive profile | Main acc | Corner acc | Fraud survival | Rarity retention | L1 review |
| --- | ---: | ---: | ---: | ---: | ---: |
| Calibrated rarity-main tolerance 0.00925 | 0.4772 | 0.7293 | 0.0000 | 0.3665 | 0.8500 |

Under the held-out protocol, this profile should be treated as a frozen
calibration point rather than a newly tuned optimum. The updated reliability
table is intentionally not retuned on the same final seeds; it exposes the
generalization cost that a reviewer would ask to see. Future tuning should use a
separate calibration split and report only on untouched seeds or datasets.

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
