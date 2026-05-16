# Legacy Scripts

These scripts are kept for historical comparison and dashboard-development
debugging. They are not the canonical thesis artifact path.

Use the current synthetic ALG exporter instead:

```bash
python scripts/export_synthetic_alg_benchmark.py --rounds 24 --cycle-rounds 12 --pretrain-epochs 5 --output-dir results/synthetic_alg_benchmark
```

Legacy contents:

- `run_unified_benchmark.py`: older unified benchmark CLI.
- `run_fedavg_baseline.py`: older dashboard baseline CLI.
- `evaluate_system.py`: older live/default policy diagnostic script.
