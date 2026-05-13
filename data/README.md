# Data Preparation

The repository does not track full datasets. Runtime data goes under
`data/real/`, which is intentionally ignored by Git.

## MNIST and FashionMNIST

The real-gradient benchmark can download the torchvision datasets automatically:

```bash
python scripts/prepare_data.py --download-torchvision
```

or directly during the benchmark run:

```bash
python scripts/export_real_gradient_reliability_benchmark.py --download --sources mnist,fashionmnist
```

The benchmark creates deterministic non-IID pseudo-clients from image samples.
For the thesis setting, each source uses 120 pseudo-clients, 20 clients per
round, and 10 rounds per seed.

## FEMNIST / LEAF

FEMNIST is expected at:

```text
data/real/femnist/train/
data/real/femnist/test/
```

Use the LEAF preprocessing scripts or an equivalent FEMNIST export that matches
LEAF's JSON client format. Keep the generated files in `data/real/femnist/`; do
not commit them to Git.

## BDD100K Optional Calibration

BDD100K is optional and is used only for domain-relevance calibration, not for
the main thesis tables. If used, place labels and images under:

```text
data/real/bdd100k/labels/
data/real/bdd100k/images/
```

The BDD100K adapter groups frames into deterministic pseudo-clients by
attributes such as weather/time-of-day. It is not a claim of real vehicle-level
federated client IDs.

## D_main and D_corner Construction

CornerDrive audits every suspect update against two validation channels:

- `D_main`: standard-distribution validation samples for the main task.
- `D_corner`: rare or under-represented labels/conditions used to measure
  corner-case benefit.

For MNIST/FashionMNIST, `D_corner` is built from rare label groups configured in
the benchmark code. For FEMNIST, `D_corner` is derived from corner-heavy client
label distributions. The thesis real-gradient configuration is mirrored in
`configs/real_gradient_reliability.yaml`.
