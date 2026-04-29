# Legacy Scripts

These scripts are kept for historical comparison and dashboard-development
debugging. They are not the canonical thesis artifact path.

Use the current V2.5 exporter instead:

```bash
python scripts/export_v25_artifacts.py --rounds 24 --cycle-rounds 12 --pretrain-epochs 5 --output-dir results/v25_artifacts
```

Legacy contents:

- `run_unified_benchmark.py`: older unified benchmark CLI.
- `run_fedavg_baseline.py`: older dashboard baseline CLI.
- `evaluate_system.py`: older live/default policy diagnostic script.
