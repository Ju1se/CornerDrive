#!/usr/bin/env python3
"""Tables of the CIFAR-10 dual-dataset rarity benchmark.

Reads the unit files of one stage and prints (or writes with --md) markdown
tables: method comparison, rarity discovery by the dual-dataset test,
update-level confusion, L1 routing, and a detector comparison on the shared
reference update stream. Every figure is a mean over seeds with the half-width
of its 95% t-interval, unless it is a pooled count.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from dual_dataset_benchmark.analysis import (  # noqa: E402
    TYPE_LABELS,
    TYPES,
    auroc,
    confusion,
    drifts,
    exhaustive_verdicts,
    load_units,
    mean_ci,
    rarity_discovery,
    tail_accuracy,
    type_of,
    weight_share,
)
from dual_dataset_benchmark.detectors import REASONS, VERDICTS  # noqa: E402
from dual_dataset_benchmark.engine import METHOD_LABELS, METHODS, CornerDrivePolicy  # noqa: E402

FRAUD_TYPES = ("sign_flip", "label_flip", "corner_harm")
SHORT = {"oracle": "Oracle", "fedavg": "FedAvg", "geomed": "GeoMed", "krum": "Multi-Krum",
         "fltrust": "FLTrust", "zeno": "Zeno", "zenopp": "Zeno++", "fedval": "FedVal",
         "cornerdrive": "CornerDrive (L1+L2)", "cd_exhaustive": "CornerDrive (exhaustive L2)",
         "cd_main_only": "Main-only audit", "cd_corner_only": "Corner-only audit",
         "cd_embedding": "CornerDrive (embedding space)"}


def fmt(values, digits: int = 3, pct: bool = False) -> str:
    mean, half = mean_ci(values)
    if np.isnan(mean):
        return "–"
    scale = 100.0 if pct else 1.0
    spec = f"{{:.{1 if pct else digits}f}}"
    if np.isnan(half):
        return spec.format(mean * scale)
    return f"{spec.format(mean * scale)} ± {spec.format(half * scale)}"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def by_method(units: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for unit in units:
        grouped[unit["meta"]["method"]].append(unit)
    return {m: sorted(grouped[m], key=lambda u: u["meta"]["seed"]) for m in METHODS if m in grouped}


def verdict_names(unit: dict) -> np.ndarray:
    return np.array(VERDICTS, dtype=object)[unit["u_verdict"]]


# --------------------------------------------------------------------------
def method_comparison(grouped: dict[str, list[dict]], attacked: bool) -> str:
    header = ["Method", "Normal acc", "Rare acc", "All acc", "Rarity kept", "Rarity share",
              "Honest kept"]
    if attacked:
        header += ["Sign-flip share", "Label-flip share", "Corner-harm share", "Fraud kept"]
    rows = []
    for method, units in grouped.items():
        acc = [tail_accuracy(u) for u in units]
        ws = [weight_share(u) for u in units]
        row = [SHORT[method],
               fmt([a["acc_normal"] for a in acc], pct=True),
               fmt([a["acc_rare"] for a in acc], pct=True),
               fmt([a["acc_all"] for a in acc], pct=True),
               fmt([w.get("kept_rarity", np.nan) for w in ws], pct=True),
               fmt([w.get("share_rarity", np.nan) for w in ws], 2),
               fmt([w.get("kept_honest", np.nan) for w in ws], pct=True)]
        if attacked:
            row += [fmt([w.get(f"share_{t}", np.nan) for w in ws], 2) for t in FRAUD_TYPES]
            kept = []
            for u in units:
                types = type_of(u)
                kept.append(float((u["u_weight"][np.isin(types, FRAUD_TYPES)] > 0).mean()))
            row.append(fmt(kept, pct=True))
        rows.append(row)
    return table(header, rows)


def paired_table(grouped: dict[str, list[dict]], anchor: str = "cornerdrive") -> str:
    """Seed-paired accuracy difference, anchor minus method, in points."""
    if anchor not in grouped:
        return ""
    mine = {u["meta"]["seed"]: tail_accuracy(u) for u in grouped[anchor]}
    header = ["Compared with", "Δ all acc", "Δ rare acc", "Δ normal acc"]
    rows = []
    for method, units in grouped.items():
        if method == anchor:
            continue
        theirs = {u["meta"]["seed"]: tail_accuracy(u) for u in units}
        seeds = sorted(set(mine) & set(theirs))
        if len(seeds) < 2:
            continue
        cells = []
        for key in ("acc_all", "acc_rare", "acc_normal"):
            diff = [100.0 * (mine[s][key] - theirs[s][key]) for s in seeds]
            mean, half = mean_ci(diff)
            cells.append(f"{mean:+.1f} ± {half:.1f}" + (" *" if abs(mean) > half else ""))
        rows.append([SHORT[method]] + cells)
    return table(header, rows)


def discovery_table(grouped: dict[str, list[dict]]) -> str:
    header = ["Audit", "Review rate", "Recognition", "Precision", "Said Rarity",
              "of which fraud", "of which corner-harm", "of which honest/noise"]
    rows = []
    for method in ("cornerdrive", "cd_exhaustive", "cd_embedding", "cd_corner_only", "cd_main_only"):
        if method not in grouped:
            continue
        per_seed, pooled_t, pooled_v, review = [], [], [], []
        for unit in grouped[method]:
            types, verdicts = type_of(unit), verdict_names(unit)
            per_seed.append(rarity_discovery(types, verdicts))
            pooled_t.append(types); pooled_v.append(verdicts)
            review.append(float(unit["u_routed"].mean()))
        pooled = rarity_discovery(np.concatenate(pooled_t), np.concatenate(pooled_v))
        precision = [r["precision"] if r["said_rarity"] else np.nan for r in per_seed]
        rows.append([SHORT[method], fmt(review, pct=True),
                     fmt([r["recognition"] for r in per_seed], pct=True),
                     fmt(precision, pct=True),
                     pooled["said_rarity"], pooled["fraud_named_rarity"],
                     pooled["corner_harm_named_rarity"],
                     pooled["said_rarity"] - pooled["fraud_named_rarity"]
                     - int(((np.concatenate(pooled_t) == "rarity")
                            & (np.concatenate(pooled_v) == "RARITY")).sum())])
    return table(header, rows)


def confusion_table(units: list[dict]) -> str:
    types = np.concatenate([type_of(u) for u in units])
    verdicts = np.concatenate([verdict_names(u) for u in units])
    counts = confusion(types, verdicts)
    header = ["True type", "Total", "Fraud", "Rarity", "Honest", "Noise", "Not audited"]
    rows = [[TYPE_LABELS[t], counts[t]["total"], counts[t]["FRAUD"], counts[t]["RARITY"],
             counts[t]["HONEST"], counts[t]["NOISE"], counts[t]["UNAUDITED"]] for t in TYPES
            if counts[t]["total"]]
    return table(header, rows)


def routing_table(units: list[dict]) -> str:
    types = np.concatenate([type_of(u) for u in units])
    reasons = np.concatenate([np.array(REASONS, dtype=object)[u["u_reason"]] for u in units])
    header = ["True type", "Total", "Cosine", "Norm-MAD", "Sign", "Recheck", "Bypass", "Routed %"]
    rows = []
    for t in TYPES:
        mask = types == t
        if not mask.any():
            continue
        c = {r: int((reasons[mask] == r).sum()) for r in REASONS}
        routed = mask.sum() - c["bypass"]
        rows.append([TYPE_LABELS[t], int(mask.sum()), c["cosine_screening"],
                     c["norm_mad_screening"], c["sign_screening"], c["probabilistic_recheck"],
                     c["bypass"], f"{100.0 * routed / mask.sum():.1f}"])
    return table(header, rows)


def signature_table(units: list[dict], policy: CornerDrivePolicy | None) -> str:
    step = policy.audit_step if policy else 1.0
    types = np.concatenate([type_of(u) for u in units])
    dm = np.concatenate([drifts(u, "relative", step)[0] for u in units])
    dc = np.concatenate([drifts(u, "relative", step)[1] for u in units])
    em = np.concatenate([drifts(u, "relative", space="embedding")[0] for u in units])
    ec = np.concatenate([drifts(u, "relative", space="embedding")[1] for u in units])

    def cell(values, mask):
        return (f"{np.median(values[mask]):+.3f} [{np.percentile(values[mask], 5):+.3f}, "
                f"{np.percentile(values[mask], 95):+.3f}]")

    header = ["True type", "n", "Loss drift, normal set", "Loss drift, rare set",
              "Embedding drift, normal set", "Embedding drift, rare set"]
    rows = [[TYPE_LABELS[t], int((types == t).sum()), cell(dm, types == t), cell(dc, types == t),
             cell(em, types == t), cell(ec, types == t)] for t in TYPES if (types == t).any()]
    return table(header, rows)


def _score_bank(policies: dict[str, CornerDrivePolicy]) -> tuple[dict, dict]:
    """Scores every method assigns to an update, on a common footing.

    value scores: higher = the method values the update more.
    harm scores:  higher = the method trusts the update more (less harmful).
    """
    loss, emb = policies.get("loss"), policies.get("embedding")
    step = loss.audit_step if loss else 1.0

    def gain(policy, space):
        def score(unit):
            dm, dc = drifts(unit, "relative", policy.audit_step if space == "loss" else 1.0,
                            space=space)
            safe = (dm <= policy.verdict.theta_tol) & (dc <= policy.verdict.theta_corner_harm)
            return np.where(safe, np.maximum(-dc, 0.0), -1.0)
        return score

    def harm(policy, space):
        def score(unit):
            dm, dc = drifts(unit, "relative", policy.audit_step if space == "loss" else 1.0,
                            space=space)
            return -np.maximum(dm / policy.verdict.theta_tol, dc / policy.verdict.theta_corner_harm)
        return score

    baselines = {
        "Zeno score": lambda u: u["u_score_zeno"],
        "Zeno++ score": lambda u: u["u_score_zenopp"],
        "FLTrust cosine": lambda u: u["u_score_fltrust"],
        "Multi-Krum score": lambda u: u["u_score_krum"],
        "FedVal score": lambda u: u["u_score_fedval"],
    }
    value = {"Rare-set loss drift alone (corner-only)": lambda u: -drifts(u, "relative", step)[1]}
    trust = {}
    if loss:
        value["Dual test, loss space (rare-set gain if both harm tests pass)"] = gain(loss, "loss")
        trust["Dual test, loss space (larger of the two drifts / tolerance)"] = harm(loss, "loss")
    value["Rare-set embedding drift alone"] = lambda u: -drifts(u, "relative", space="embedding")[1]
    if emb:
        value["Dual test, embedding space (rare-set gain if both harm tests pass)"] = gain(emb, "embedding")
        trust["Dual test, embedding space (larger of the two drifts / tolerance)"] = harm(emb, "embedding")
    value.update(baselines)
    trust.update(baselines)
    return value, trust


def detector_table(units: list[dict], policies: dict[str, CornerDrivePolicy]) -> str:
    """AUROC of every score on the same updates (one value per seed).

    First table: is a rare-beneficial update valued above the others?
    Second table: are benign updates (honest + rarity) trusted above fraud?
    """
    value, trust = _score_bank(policies)
    header = ["Score", "Rarity vs honest", "Rarity vs corner-harm", "Rarity vs label-flip",
              "Rarity vs sign-flip", "Rarity vs all others"]
    rows = []
    for name, getter in value.items():
        cols: list[list[float]] = [[] for _ in range(5)]
        for unit in units:
            types, s = type_of(unit).ravel(), np.asarray(getter(unit)).ravel()
            rarity = types == "rarity"
            for i, other in enumerate(("honest", "corner_harm", "label_flip", "sign_flip")):
                cols[i].append(auroc(s[rarity], s[types == other]))
            cols[4].append(auroc(s[rarity], s[~rarity]))
        rows.append([name] + [fmt(c) for c in cols])
    first = table(header, rows)

    header = ["Score", "Benign vs sign-flip", "Benign vs label-flip", "Benign vs corner-harm",
              "Benign vs all fraud", "Rarity vs honest (0.5 = no bias)"]
    rows = []
    for name, getter in trust.items():
        cols = [[] for _ in range(5)]
        for unit in units:
            types, s = type_of(unit).ravel(), np.asarray(getter(unit)).ravel()
            benign = np.isin(types, ("honest", "rarity"))
            for i, other in enumerate(FRAUD_TYPES):
                cols[i].append(auroc(s[benign], s[types == other]))
            cols[3].append(auroc(s[benign], s[np.isin(types, FRAUD_TYPES)]))
            cols[4].append(auroc(s[types == "rarity"], s[types == "honest"]))
        rows.append([name] + [fmt(c) for c in cols])
    return ("Is a rare-beneficial update valued above the others?\n\n" + first
            + "\n\nAre benign updates trusted above fraud?\n\n" + table(header, rows))


def offline_verdicts(units: list[dict], policies: dict[str, CornerDrivePolicy]) -> str:
    """The frozen dual test applied to every update of the reference stream."""
    header = ["Dual test on the reference stream", "Recognition", "Precision", "Honest named Rarity",
              "Fraud named Rarity", "Sign-flip passed", "Label-flip passed", "Corner-harm passed",
              "Honest rejected"]
    rows = []
    for name, policy in policies.items():
        per_seed = defaultdict(list)
        for unit in units:
            types = type_of(unit).ravel()
            verdicts = exhaustive_verdicts(unit, policy.verdict, policy.drift_mode,
                                           policy.audit_step, space=policy.space).ravel()
            found = rarity_discovery(types, verdicts)
            passed = np.isin(verdicts, ("RARITY", "HONEST"))
            per_seed["recognition"].append(found["recognition"])
            per_seed["precision"].append(found["precision"])
            per_seed["honest_rarity"].append(float((verdicts[types == "honest"] == "RARITY").mean()))
            per_seed["fraud_rarity"].append(
                float((verdicts[np.isin(types, FRAUD_TYPES)] == "RARITY").mean()))
            for t in FRAUD_TYPES:
                per_seed[t].append(float(passed[types == t].mean()))
            per_seed["honest_rejected"].append(float((verdicts[types == "honest"] == "FRAUD").mean()))
        rows.append([f"{name} space", *[fmt(per_seed[k], pct=True) for k in (
            "recognition", "precision", "honest_rarity", "fraud_rarity", "sign_flip",
            "label_flip", "corner_harm", "honest_rejected")]])
    return table(header, rows)


def stress_table(stages: list[tuple[str, Path]], methods: tuple[str, ...]) -> str:
    """One block of rows per stress condition (mixed scenario)."""
    header = ["Condition", "Method", "Rare acc", "All acc", "Rarity kept", "Recognition",
              "Precision", "Corner-harm share", "Fraud kept"]
    rows = []
    for label, folder in stages:
        grouped = by_method(load_units(folder, scenario="mixed"))
        for method in methods:
            if method not in grouped:
                continue
            units = grouped[method]
            acc = [tail_accuracy(u) for u in units]
            ws = [weight_share(u) for u in units]
            fraud_kept = [float((u["u_weight"][np.isin(type_of(u), FRAUD_TYPES)] > 0).mean())
                          for u in units]
            if method.startswith(("cornerdrive", "cd_")):
                found = [rarity_discovery(type_of(u), verdict_names(u)) for u in units]
                recognition = fmt([f["recognition"] for f in found], pct=True)
                precision = fmt([f["precision"] if f["said_rarity"] else np.nan for f in found],
                                pct=True)
            else:
                recognition = precision = "–"
            rows.append([label, SHORT[method],
                         fmt([a["acc_rare"] for a in acc], pct=True),
                         fmt([a["acc_all"] for a in acc], pct=True),
                         fmt([w.get("kept_rarity", np.nan) for w in ws], pct=True),
                         recognition, precision,
                         fmt([w.get("share_corner_harm", np.nan) for w in ws], 2),
                         fmt(fraud_kept, pct=True)])
    return table(header, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--policy", default="")
    parser.add_argument("--md", default="")
    parser.add_argument("--scenarios", default="mixed,no_attack")
    parser.add_argument("--csv", default="", help="write one row per unit to this CSV")
    parser.add_argument("--reference-dir", default="",
                        help="stage holding the reference trajectories (oracle, fedavg) logged "
                             "with the frozen replica step; defaults to --dir")
    parser.add_argument("--stress", nargs="*", default=[], metavar="LABEL=DIR",
                        help="extra stages to tabulate as stress conditions")
    args = parser.parse_args()

    folder = Path(args.dir)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    policies: dict[str, CornerDrivePolicy] = {}
    if args.policy:
        path = Path(args.policy)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        policies = {k: CornerDrivePolicy.from_json(v)
                    for k, v in json.loads(path.read_text())["policies"].items()}
    loss_policy = policies.get("loss")
    reference_folder = folder
    if args.reference_dir:
        reference_folder = Path(args.reference_dir)
        if not reference_folder.is_absolute():
            reference_folder = PROJECT_ROOT / reference_folder

    out: list[str] = []
    for scenario in [s for s in args.scenarios.split(",") if s]:
        units = load_units(folder, scenario=scenario)
        if not units:
            continue
        grouped = by_method(units)
        seeds = sorted({u["meta"]["seed"] for u in units})
        rounds, per_round = units[0]["u_role"].shape
        out.append(f"## Scenario: {scenario}\n")
        out.append(f"{len(seeds)} seeds ({', '.join(map(str, seeds))}), {rounds} rounds, "
                   f"{per_round} updates per round, model `{units[0]['meta'].get('model', 'cnn')}`. "
                   f"Accuracy is the mean of the last ten rounds on the held-out evaluation set.\n")
        out.append("### Method comparison\n")
        out.append(method_comparison(grouped, attacked=scenario == "mixed") + "\n")
        out.append("Share = mean aggregation weight relative to an equal share (FedAvg = 1.00). "
                   "Kept = fraction of updates with non-zero weight.\n")
        if "cornerdrive" in grouped:
            out.append("### Seed-paired accuracy difference: CornerDrive (L1+L2) minus each method\n")
            out.append(paired_table(grouped) + "\n")
            out.append("Points, mean ± half-width of the 95% t-interval over paired seeds; "
                       "* marks an interval that excludes zero.\n")
        if any(m in grouped for m in ("cornerdrive", "cd_exhaustive")):
            out.append("### Rarity discovery by the dual-dataset test (each method's own run)\n")
            out.append(discovery_table(grouped) + "\n")
        for method in ("cornerdrive", "cd_exhaustive"):
            if method in grouped:
                out.append(f"### Update-level confusion, {SHORT[method]} (pooled over seeds)\n")
                out.append(confusion_table(grouped[method]) + "\n")
        if "cornerdrive" in grouped:
            out.append("### L1 routing by type and reason, CornerDrive (L1+L2), pooled\n")
            out.append(routing_table(grouped["cornerdrive"]) + "\n")
        streams = by_method(load_units(reference_folder, scenario=scenario))
        for reference in ("oracle", "fedavg"):
            if scenario != "mixed" or reference not in streams:
                continue
            if "u_score_zeno" not in streams[reference][0]:
                continue
            out.append(f"### Golden-set signature of every update type "
                       f"({SHORT[reference]} trajectory, relative drift, median [p5, p95])\n")
            out.append(signature_table(streams[reference], loss_policy) + "\n")
            out.append(f"### Detectors on the same update stream ({SHORT[reference]} "
                       f"trajectory, AUROC per seed)\n")
            out.append(detector_table(streams[reference], policies) + "\n")
            if policies:
                out.append(f"### Frozen dual test applied to every update "
                           f"({SHORT[reference]} trajectory)\n")
                out.append(offline_verdicts(streams[reference], policies) + "\n")

    if args.stress:
        stages = [("Default", folder)]
        for item in args.stress:
            label, _, raw = item.partition("=")
            path = Path(raw)
            stages.append((label, path if path.is_absolute() else PROJECT_ROOT / path))
        out.append("## Stress tests (mixed scenario, frozen policy unchanged)\n")
        out.append(stress_table(stages, ("oracle", "krum", "zeno", "fedval", "cornerdrive",
                                         "cd_exhaustive")) + "\n")

    if args.csv:
        import csv
        rows = []
        for unit in load_units(folder):
            meta = unit["meta"]
            row = {"model": meta.get("model", "cnn"), "scenario": meta["scenario"], "method": meta["method"],
                   "seed": meta["seed"], **tail_accuracy(unit), **weight_share(unit),
                   "review_rate": float(unit["u_routed"].mean())}
            if meta["method"].startswith(("cornerdrive", "cd_")):
                found = rarity_discovery(type_of(unit), verdict_names(unit))
                row.update({f"rarity_{k}": v for k, v in found.items()})
            rows.append(row)
        csv_path = Path(args.csv)
        if not csv_path.is_absolute():
            csv_path = PROJECT_ROOT / csv_path
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({k for row in rows for k in row},
                        key=lambda k: (k not in ("model", "scenario", "method", "seed"), k))
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {csv_path}")

    text = "\n".join(out)
    if args.md:
        path = Path(args.md)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        print(text)


if __name__ == "__main__":
    main()
