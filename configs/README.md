# Experiment Configs

These YAML files mirror the thesis reproduction settings. The current exporter
scripts still take command-line arguments and a small number of environment
variables; keep the YAML files synchronized with the README commands so readers
can audit the exact parameter choices without searching through source code.

The thesis-matching synthetic ALG setting is `synthetic_alg.yaml`. Its most
important reproducibility controls are:

- `batch_size: 24`
- `vehicle_pool_size: 128`
- five generator seeds: `20260318` through `20260322`
- `p_recheck` sweep: `0.00, 0.05, 0.10, 0.20, 0.30`

The thesis real-gradient setting is `real_gradient_reliability.yaml`. It records
the held-out split protocol used to keep client/update gradients, audit/reference
data, and final evaluation data separate.

`real_gradient_calibration_manifest.json` records the frozen calibrated
CornerDrive profile, the calibration seed range, and the final held-out seed
range. The two seed sets are intentionally disjoint so threshold selection can
be audited separately from final reporting.

`cifar10_dual_dataset.yaml` records the CIFAR-10 dual-dataset rarity benchmark:
the golden-dataset annotation, the client federation, both model back-ends and
the method list. `cifar10_dual_dataset_policy_r50head.json` is the CornerDrive
operating point frozen on development seeds 20261101-20261102 by
`scripts/calibrate_cifar10_dual_dataset.py`; the confirmatory seeds
20261110-20261114 use it unchanged.
