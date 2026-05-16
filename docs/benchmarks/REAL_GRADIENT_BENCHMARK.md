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
2. keep client/update samples separate from server-side audit/reference samples
   and final evaluation samples;
3. freeze a deterministic model checkpoint;
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

The exporter defaults to `--policy-profile real_data_adaptive`. This profile
comes from the current MNIST, FashionMNIST, and LEAF/FEMNIST real-gradient
calibration runs. It tightens L2 fraud tolerance, relaxes the rarity threshold
enough to preserve mixed real clients, and routes CornerDrive through L1V3
risk-budget screening:

Current calibrated values:

- `theta_tol = 0.02`
- `theta_rare = -0.005`
- `theta_rarity_main_tol = 0.02` for the legacy V3/V4 calibrated profile
- `cosine_filter_threshold = 0.50`
- `recheck_probability = 0.25`
- `cornerdrive_l1_mode = v3_m3_budgeted`
- `cornerdrive_l1_queue_budget_ratio = 0.80`
- `cornerdrive_l1_random_recheck_ratio = 0.05`
- `norm_mad_threshold = 1.5`
- `sign_threshold = 0.40`
- risk weights: cosine `0.35`, norm `0.20`, sign `0.15`

```bash
python scripts/export_real_gradient_benchmark.py \
  --source leaf_femnist \
  --leaf-data-dir data/real/femnist \
  --policy-profile real_data_adaptive \
  --cornerdrive-l1-mode v3_m3_budgeted
```

Use `--policy-profile default --cornerdrive-l1-mode v25_cosine_fixed` to
reproduce the original V2.5 cosine-only CornerDrive behavior.

An experimental V4 profile is also available for the current fraud-survival
diagnostics work:

```bash
python scripts/export_real_gradient_benchmark.py \
  --source leaf_femnist \
  --leaf-data-dir data/real/femnist \
  --policy-profile real_data_adaptive_v4
```

`real_data_adaptive_v4` keeps the same legacy L2 policy values but changes
`cornerdrive_l1_mode` to `v4_m4_dual_proxy_budgeted`. This mode adds cheap
first-order main/corner loss-drift proxies at L1:

- `pred_delta_main ~= -eta * <grad_main_val, grad_client>`
- `pred_delta_corner ~= -eta * <grad_corner_val, grad_client>`
- route actions: `SAFE_ACCEPT`, `AUDIT`, `QUARANTINE`, `LOW_WEIGHT`
- weighted aggregation fields: `effective_fraud_mass_survival` and
  `effective_rarity_mass_retention`
- diagnostic fields: `l1_fraud_recall`,
  `l2_fraud_reject_rate_given_routed`, `fraud_survival_unrouted`, and
  `fraud_survival_l2_accepted`

Use `real_data_adaptive_v41` for the stricter L2 rarity-safety profile:

```bash
python scripts/export_real_gradient_benchmark.py \
  --source leaf_femnist \
  --leaf-data-dir data/real/femnist \
  --policy-profile real_data_adaptive_v41
```

V4.1 keeps V4 routing but changes clean RARITY from
`delta_main <= theta_tol` to `delta_main <= theta_rarity_main_tol`, with
`theta_rarity_main_tol = 0.00925` in the calibrated real-gradient profile. This
treats updates that improve corner loss while introducing positive main-task
drift above the stricter band as conflict or noise rather than clean rarity.
The value was selected by a 20-seed threshold sweep as the largest tested
zero-fraud setting before the FashionMNIST boundary case reappeared.

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
and 10 seeds per dataset. It exports per-run metrics plus mean, standard
deviation, and 95% confidence intervals:

```bash
python scripts/export_real_gradient_reliability_benchmark.py \
  --sources mnist,fashionmnist,femnist \
  --seeds 20260507,20260508,20260509,20260510,20260511,20260512,20260513,20260514,20260515,20260516 \
  --max-clients 120 \
  --max-samples-per-client 48 \
  --clients-per-round 20 \
  --rounds 10 \
  --pretrain-steps 50 \
  --reference-split-fraction 0.50 \
  --max-reference-samples 4096 \
  --max-evaluation-samples 4096 \
  --output-dir results/real_gradient_reliability_medium
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
audit/reference and final evaluation clients. This raises the evidence from 128
client-round observations per dataset in the original single-seed run to 2,000
observations per dataset:

| Dataset | CornerDrive main acc | Corner acc | Fraud survival | Rarity retention | L1 review |
| --- | ---: | ---: | ---: | ---: | ---: |
| MNIST | 0.7326 +/- 0.0134 | 0.8544 +/- 0.0156 | 0.3900 +/- 0.0619 | 0.8125 +/- 0.0725 | 0.8500 +/- 0.0000 |
| FashionMNIST | 0.5941 +/- 0.0117 | 0.9138 +/- 0.0081 | 0.3080 +/- 0.0465 | 0.4937 +/- 0.0418 | 0.8500 +/- 0.0000 |
| LEAF/FEMNIST | 0.0918 +/- 0.0095 | 0.3755 +/- 0.0291 | 0.0200 +/- 0.0202 | 0.2155 +/- 0.0848 | 0.8500 +/- 0.0000 |

Across all three datasets, this completed expanded run covers 6,000
client-round observations, including 1,500 fraud observations and 1,896 rarity
observations. With the M3 risk-budget router, CornerDrive averages 0.2393 fraud
survival, 0.5072 rarity retention, 0.7146 corner accuracy, and 0.8500 L1 review
coverage under the held-out evaluation protocol.

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
clean region while still being caught by risk-budget L1V3 routing.

Held-out calibration on MNIST, FashionMNIST, and LEAF/FEMNIST shows why the M3
risk-budget profile was selected for real data:

| CornerDrive profile | Main acc | Corner acc | Fraud survival | Rarity retention | L1 review |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial held-out profile | 0.4721 | 0.6611 | 0.3444 | 0.5763 | 0.7428 |
| L1 aggressive thresholds | 0.4709 | 0.6940 | 0.2333 | 0.5388 | 0.8189 |
| M3 risk budget 0.80 | 0.4702 | 0.7155 | 0.2267 | 0.5465 | 0.8500 |
| M3 sign-heavy 0.80 | 0.4693 | 0.7086 | 0.2556 | 0.5415 | 0.8500 |

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
