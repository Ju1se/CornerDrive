# Reports Index

This folder keeps reviewer-facing audit notes, calibration notes, and design
history. Treat dated reports as evidence records for the run described in that
file, not as generic always-current documentation.

## Current Reviewer-Facing Evidence

| Report | Use |
| --- | --- |
| `REVIEWER_AUDIT_FIXES_2026-05-15.md` | Main record of leakage, seed/config, and table-generation fixes after reviewer-style audit. |
| `REAL_GRADIENT_THRESHOLD_CALIBRATION_2026-05-15.md` | Calibration study explaining why L1 visibility was the main real-gradient bottleneck. |
| `REAL_GRADIENT_V3_V4_ABLATION_2026-05-16.md` | V3 vs V4 real-gradient ablation. |
| `REAL_GRADIENT_V41_L2_RARITY_SAFETY_2026-05-16.md` | Diagnostic for the V4.1 rarity safety rule change. |
| `REAL_GRADIENT_V41_RARITY_MAIN_TOL_SWEEP_2026-05-16.md` | Initial V4.1 rarity-main tolerance sweep. |
| `REAL_GRADIENT_V41_RARITY_MAIN_TOL_S20_REFINED_2026-05-16.md` | Refined V4.1 operating point selection. |
| `REAL_GRADIENT_V41_BEST_BASELINE_2026-05-16.md` | Full-seed V4.1 calibrated baseline benchmark. |
| `REAL_GRADIENT_V41_BEST_HELDOUT_2026-05-16.md` | Held-out V4.1 benchmark report. |

## Code and Integrity Audits

| Report | Status |
| --- | --- |
| `SYNTHETIC_ALG_CODE_AUDIT_AND_CLEANUP.md` | Current synthetic ALG audit and cleanup record. |
| `ACADEMIC_INTEGRITY_AUDIT_REPORT_2026-05-13.md` | Historical integrity audit. Its top note identifies which numbers are superseded by later held-out reports. |

## Legacy Notes

Older exploratory notes live under `legacy/`. They are retained for design
history only and should not be cited as current thesis evidence unless a newer
report explicitly points to them.
