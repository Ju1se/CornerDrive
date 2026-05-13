# Paper Artifacts

`scripts/make_paper_tables.py` writes paper-facing CSV tables here. These files
are derived from `results/` outputs and can be regenerated.

Suggested workflow:

```bash
bash scripts/reproduce_all.sh main
bash scripts/reproduce_all.sh appendix
python scripts/make_paper_tables.py
```
