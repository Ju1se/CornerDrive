# Real-Gradient FL Learning-Curve Benchmark

This benchmark extends the current one-step real-gradient reliability table into
an end-to-end multi-round FL training curve. It keeps the same leakage-safe data
surfaces: client/update gradients are derived from the client split, while audit
and final evaluation use held-out reference/evaluation surfaces.

## Purpose

The existing real-gradient benchmark answers whether each update helps or hurts
under a fixed round-level audit protocol. A reviewer may also ask whether that
update-level advantage persists over training. This benchmark tracks main and
corner accuracy across 50 or more FL rounds under the same rare-beneficial and
harmful update stream.

## Usage

Smoke run:

```bash
python scripts/export_real_gradient_learning_curve.py \
  --sources mnist \
  --seeds 20260507 \
  --rounds 2 \
  --max-clients 40 \
  --max-samples-per-client 16 \
  --clients-per-round 8 \
  --pretrain-steps 5 \
  --max-reference-samples 256 \
  --max-evaluation-samples 256 \
  --download \
  --output-dir results/real_gradient_learning_curve_smoke
```

Thesis-scale curve:

```bash
python scripts/export_real_gradient_learning_curve.py \
  --sources mnist,fashionmnist,femnist \
  --seeds 20260507,20260508,20260509,20260510,20260511 \
  --rounds 50 \
  --max-clients 120 \
  --max-samples-per-client 48 \
  --clients-per-round 20 \
  --pretrain-steps 50 \
  --max-reference-samples 4096 \
  --max-evaluation-samples 4096 \
  --download \
  --output-dir results/real_gradient_learning_curve
```

For a stronger journal-scale run, add more seeds beyond the current five:

```bash
--seeds 20260507,20260508,20260509,20260510,20260511,20260512,20260513,20260514,20260515,20260516
```

## Outputs

- `real_gradient_learning_curve_rounds.csv`: one row per
  source/seed/method/round.
- `real_gradient_learning_curve_by_round.csv`: mean/std by method and round.
- `real_gradient_learning_curve_final_summary.csv`: final-round summary table.
- `real_gradient_learning_curve.svg`: dependency-free learning-curve figure.
- `real_gradient_learning_curve_config.json`: sources, seeds, and exact run
  configuration.

## Current 50-Round Run

The current thesis-scale five-seed run is stored in
`results/real_gradient_learning_curve`.

| Method | Final main acc. | Final corner acc. | Final fraud survival | Final rarity retention |
| --- | ---: | ---: | ---: | ---: |
| FedAvg | 0.3693 | 0.3464 | 1.0000 | 1.0000 |
| Multi-Krum | 0.4238 | 0.5802 | 0.2000 | 0.8867 |
| FLTrust | 0.4796 | 0.6273 | 0.0667 | 0.5469 |
| Zeno++ | 0.5036 | 0.7002 | 0.0000 | 0.0871 |
| CornerDrive | 0.4262 | 0.6768 | 0.1867 | 0.5022 |

Interpretation: after adding seeds, the learning curve no longer supports a
single-seed claim that CornerDrive matches Zeno++ on final fraud survival.
Instead, it supports a trade-off claim: CornerDrive keeps far more rarity than
Zeno++ (`0.5022` vs. `0.0871`) and much stronger corner accuracy than FedAvg and
Multi-Krum, but Zeno++ remains the strictest low-survival selector. FEMNIST
remains the hardest split and should be discussed as a limitation rather than
hidden by the macro average.

## Thesis Claim

Use this benchmark to support a cautious long-horizon claim:

> CornerDrive's value is not merely one-step filtering; when the same
> rare-beneficial and harmful update process is repeated across FL rounds, the
> method can be evaluated by whether it protects corner accuracy while reducing
> harmful update survival.

Do not overstate this as full IoV deployment. The benchmark still uses compact
public image datasets and one-step client gradients rather than real vehicles,
object detection, or long client-side local training.
