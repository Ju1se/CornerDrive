# Synthetic-Real Gradient Calibration

This exporter addresses the dissertation future-work item on real client-SGD
validation. It is a sanity check, not a deployment benchmark: it asks whether
real client gradients occupy comparable scalar regimes to the controlled ALG
archetypes.

The script compares:

- controlled ALG synthetic gradients, grouped by archetype;
- real client-SGD gradients derived from LEAF/FEMNIST if available, otherwise
  from torchvision MNIST/FashionMNIST non-IID shards.

Because the ALG model and the real-data model can have different parameter
dimensions and semantics, the exporter does **not** compare raw gradient vectors
across models. It compares scalar diagnostics that matter to CornerDrive:

- gradient norm and log norm;
- within-round cosine to the geometric median;
- deviation from the geometric median;
- dual loss drift: `delta_l_main` and `delta_l_corner`.

## Usage

Smoke run with torchvision MNIST:

```bash
python scripts/export_synthetic_real_gradient_calibration.py \
  --download \
  --synthetic-rounds 1 \
  --real-rounds 1 \
  --max-real-clients 8 \
  --real-clients-per-round 4 \
  --pretrain-epochs 1 \
  --real-pretrain-steps 1 \
  --output-dir results/synthetic_real_gradient_calibration_smoke
```

Thesis-scale sanity check:

```bash
python scripts/export_synthetic_real_gradient_calibration.py \
  --source auto \
  --download \
  --synthetic-rounds 24 \
  --real-rounds 8 \
  --max-real-clients 80 \
  --real-clients-per-round 16 \
  --pretrain-epochs 5 \
  --real-pretrain-steps 40 \
  --output-dir results/synthetic_real_gradient_calibration
```

With LEAF/FEMNIST JSON:

```bash
python scripts/export_synthetic_real_gradient_calibration.py \
  --source leaf_femnist \
  --leaf-data-dir data/real/femnist \
  --output-dir results/synthetic_real_gradient_calibration_leaf
```

## Outputs

- `synthetic_real_gradient_features.csv`: per-gradient scalar diagnostics.
- `synthetic_real_gradient_summary.csv`: per-source/per-label means and standard
  deviations.
- `synthetic_real_gradient_distance.csv`: standardized mean differences between
  each real-gradient group and each synthetic archetype.
- `synthetic_real_gradient_calibration_config.json`: data source and run
  configuration.

The distance table is the quickest read. A smaller `mean_smd` means a real
client-gradient group is closer to that synthetic archetype across the scalar
diagnostics. Large distances are useful too: they indicate that the ALG
surrogate is not yet externally calibrated for that real-gradient regime.

## Interpretation

This benchmark should be reported as calibration evidence. It can support
claims such as:

- real client-SGD gradients have comparable or different norm/deviation regimes
  from ALG Honest/Rarity updates;
- real corner-heavy clients are or are not naturally visible to L1;
- ALG thresholds may need recalibration before real deployment.

It should not be reported as a full IoV deployment result. Real vehicular data,
adaptive attackers, and long-horizon client training remain separate validation
steps.
