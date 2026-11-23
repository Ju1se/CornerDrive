#!/usr/bin/env python3
"""Calibrate the CornerDrive operating point on development seeds and freeze it.

Input: the oracle reference trajectories of the development seeds (every
update logged with both golden-set drifts). Output: a policy JSON that the
confirmatory seeds use unchanged.

Rules (fixed before any confirmatory seed is run)
-------------------------------------------------
Drift            relative loss drift  dL / L(W)  on each golden set, so one
                 threshold serves every training phase.
Replica step     eta in {1, 0.5, 0.25}: the value with the largest minimum of
                 three AUROCs on the development updates (rarity vs honest on
                 the rare set; main-hostile fraud vs benign on the normal set;
                 corner-harm vs benign on the rare set).
L2 thresholds    from the drift of HONEST development updates only ("benign
                 audit drift", thesis Section 4.5):
                     theta_tol          99th percentile on the normal set
                     theta_corner_harm  99th percentile on the rare set
                     theta_rare         1st percentile on the rare set
                                        (kept below zero)
                     theta_util         any measurable improvement on either set
L1 thresholds    95th percentile of each routing signal over honest
                 development updates; recheck probability 0.25 (thesis Table 4.1).

No Rarity or Fraud label enters a threshold; they only pick the replica step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from dual_dataset_benchmark.analysis import (  # noqa: E402
    auroc,
    confusion,
    drifts,
    exhaustive_verdicts,
    load_units,
    rarity_discovery,
    type_of,
)
from dual_dataset_benchmark.detectors import RoutingPolicy, VerdictPolicy  # noqa: E402
from dual_dataset_benchmark.engine import CornerDrivePolicy  # noqa: E402

UTILITY_FLOOR = 1e-6
RECHECK_PROBABILITY = 0.25
BENIGN_HIGH, BENIGN_LOW, ROUTING_QUANTILE = 99.0, 1.0, 95.0


def stack(units, getter):
    return np.concatenate([getter(unit) for unit in units])


def separations(units, types, step: float) -> dict[str, float]:
    dm = stack(units, lambda u: drifts(u, "relative", step)[0])
    dc = stack(units, lambda u: drifts(u, "relative", step)[1])
    benign = np.isin(types, ("honest", "rarity"))
    return {
        "rarity_vs_honest": auroc(-dc[types == "rarity"], -dc[types == "honest"]),
        "main_hostile_vs_benign": auroc(dm[np.isin(types, ("sign_flip", "label_flip"))], dm[benign]),
        "corner_harm_vs_benign": auroc(dc[types == "corner_harm"], dc[benign]),
    }


def verdict_policy(dm: np.ndarray, dc: np.ndarray, honest: np.ndarray) -> VerdictPolicy:
    return VerdictPolicy(
        theta_tol=float(np.percentile(dm[honest], BENIGN_HIGH)),
        theta_rare=float(min(np.percentile(dc[honest], BENIGN_LOW), -UTILITY_FLOOR)),
        theta_corner_harm=float(np.percentile(dc[honest], BENIGN_HIGH)),
        theta_util=UTILITY_FLOOR,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dev-dir", required=True)
    parser.add_argument("--model", required=True, choices=("cnn", "r50head"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", default="1,0.5,0.25")
    args = parser.parse_args()

    dev_dir = Path(args.dev_dir)
    if not dev_dir.is_absolute():
        dev_dir = PROJECT_ROOT / dev_dir
    units = load_units(dev_dir, method="oracle", scenario="mixed")
    if not units:
        raise SystemExit(f"no oracle units under {dev_dir}")
    seeds = sorted(unit["meta"]["seed"] for unit in units)
    types = stack(units, type_of)
    honest = types == "honest"

    step_table = {step: separations(units, types, step)
                  for step in (float(v) for v in args.steps.split(","))}
    best_step = max(step_table, key=lambda step: min(step_table[step].values()))

    dm = stack(units, lambda u: drifts(u, "relative", best_step)[0])
    dc = stack(units, lambda u: drifts(u, "relative", best_step)[1])
    loss_verdict = verdict_policy(dm, dc, honest)
    em = stack(units, lambda u: drifts(u, "relative", space="embedding")[0])
    ec = stack(units, lambda u: drifts(u, "relative", space="embedding")[1])
    embedding_verdict = verdict_policy(em, ec, honest)

    routing = RoutingPolicy(
        tau_cos=float(np.percentile(stack(units, lambda u: u["u_cos_dev"])[honest], ROUTING_QUANTILE)),
        tau_norm_mad=float(np.percentile(stack(units, lambda u: u["u_norm_mad"])[honest], ROUTING_QUANTILE)),
        tau_sign=float(np.percentile(stack(units, lambda u: u["u_sign_dis"])[honest], ROUTING_QUANTILE)),
        p_recheck=RECHECK_PROBABILITY,
    )
    policies = {
        "loss": CornerDrivePolicy(routing, loss_verdict, audit_step=best_step,
                                  drift_mode="relative", space="loss"),
        "embedding": CornerDrivePolicy(routing, embedding_verdict, audit_step=1.0,
                                       drift_mode="relative", space="embedding"),
    }

    # How the frozen thresholds read the development updates themselves
    # (every update audited). Reported for transparency; not a result.
    dev_check = {}
    for name, policy in policies.items():
        verdicts = stack(units, lambda u: exhaustive_verdicts(
            u, policy.verdict, "relative", policy.audit_step, space=policy.space))
        dev_check[name] = {"rarity": rarity_discovery(types, verdicts),
                           "confusion": confusion(types, verdicts)}

    payload = {
        "model": args.model,
        "calibrated_on": {"folder": str(dev_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                          "seeds": seeds, "trajectory": "oracle", "updates": int(types.size),
                          "honest_updates": int(honest.sum())},
        "rules": {
            "drift": "relative (dL / L(W))",
            "replica_step": "argmax over steps of the minimum of three development AUROCs",
            "theta_tol": f"{BENIGN_HIGH:g}th percentile of honest drift on the normal set",
            "theta_corner_harm": f"{BENIGN_HIGH:g}th percentile of honest drift on the rare set",
            "theta_rare": f"{BENIGN_LOW:g}st percentile of honest drift on the rare set, below zero",
            "theta_util": "any measurable improvement on either golden set",
            "routing": f"{ROUTING_QUANTILE:g}th percentile of each L1 signal over honest updates; "
                       f"recheck probability {RECHECK_PROBABILITY}",
        },
        "replica_step_table": {f"{step:g}": values for step, values in step_table.items()},
        "policies": {name: policy.to_json() for name, policy in policies.items()},
        "development_check": dev_check,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(f"development seeds {seeds}: {types.size} updates, {int(honest.sum())} honest")
    for step, values in step_table.items():
        print(f"  step {step:<4g} " + "  ".join(f"{k} {v:.4f}" for k, v in values.items())
              + f"  -> min {min(values.values()):.4f}")
    print(f"replica step: {best_step:g}")
    for name, policy in policies.items():
        v = policy.verdict
        print(f"[{name}] theta_tol {v.theta_tol:+.4f}  theta_corner_harm {v.theta_corner_harm:+.4f}"
              f"  theta_rare {v.theta_rare:+.4f}")
        r = dev_check[name]["rarity"]
        print(f"    dev check: recognition {r['recognition']:.3f}  precision {r['precision']:.3f}"
              f"  fraud named rarity {r['fraud_named_rarity']}")
    print(f"routing: cos {routing.tau_cos:.3f}  norm-MAD {routing.tau_norm_mad:.3f}"
          f"  sign {routing.tau_sign:.3f}  recheck {routing.p_recheck}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
