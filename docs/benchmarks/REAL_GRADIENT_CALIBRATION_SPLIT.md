# Real-Gradient Calibration Split

This table makes threshold selection reviewer-auditable. The calibration sweep
uses seeds `20260507` and `20260508` as the dev split, then reports the chosen
settings on seed `20260509` as the test split. Deprecated router profiles are
intentionally excluded from this helper so the reproduction path stays aligned
with the final calibrated benchmark.

## Usage

```bash
python scripts/export_real_gradient_calibration_split.py \
  --input-root results/real_gradient_threshold_sweep \
  --output-dir results/real_gradient_calibration_split
```

## Outputs

- `real_gradient_calibration_split_long.csv`: one row per profile and split.
- `real_gradient_calibration_split_summary.csv`: one row per profile with dev
  fraud survival and test-set utility/safety metrics side by side.

The thesis should use this table to state that thresholds and L1 routing
profiles are selected on dev seeds, while the reported benchmark numbers are
computed after fixing the parameters on held-out seeds. This does not make the
compact MNIST/FashionMNIST/FEMNIST benchmark a deployment study; it simply
removes the obvious "was this tuned on the test set?" concern.
