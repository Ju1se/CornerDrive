#!/usr/bin/env python3
"""Draw the verdict plane of the dual-dataset test as an SVG (no plotting library).

One small panel per update type. Every dot is one client update, placed by
its relative loss drift on the golden normal set (x) and on the golden rare
set (y). The frozen thresholds cut the plane into the verdict regions, and the
panel header says which share of that type falls in each region.

    python scripts/plot_cifar10_dual_dataset_plane.py \
        --dir results/cifar10_dual_dataset/confirm_r50 \
        --policy configs/cifar10_dual_dataset_policy_r50head.json \
        --out artifacts/figures/cifar10_dual_dataset_verdict_plane.svg
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from dual_dataset_benchmark.analysis import TYPE_LABELS, TYPES, drifts, load_units, type_of  # noqa: E402
from dual_dataset_benchmark.detectors import dual_verdict  # noqa: E402
from dual_dataset_benchmark.engine import CornerDrivePolicy  # noqa: E402

LINEAR = 0.05                       # symmetric-log scale: linear below, log above
X_RANGE, Y_RANGE = (-0.3, 3.0), (-0.6, 3.0)
TICKS = (-0.3, -0.1, 0.0, 0.1, 0.3, 1.0, 3.0)
PANEL_W, PANEL_H, COLS = 268, 236, 3
GAP_X, GAP_Y, LEFT, TOP = 46, 96, 62, 108
MAX_POINTS = 350

STYLE = """
  .surface { fill: #fcfcfb; }
  .ink { fill: #0b0b0b; } .ink2 { fill: #52514e; } .muted { fill: #898781; }
  .grid { stroke: #e1e0d9; stroke-width: 1; }
  .axis { stroke: #c3c2b7; stroke-width: 1; fill: none; }
  .cut { stroke: #898781; stroke-width: 1; }
  .reject { fill: #898781; fill-opacity: 0.10; }
  .dot { fill: #2a78d6; fill-opacity: 0.55; stroke: #fcfcfb; stroke-width: 0.8; }
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .tick { font-size: 10px; font-variant-numeric: tabular-nums; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .ink { fill: #f0efec; } .ink2 { fill: #c3c2b7; } .muted { fill: #898781; }
    .grid { stroke: #2c2c2a; } .axis { stroke: #383835; } .cut { stroke: #898781; }
    .reject { fill: #898781; fill-opacity: 0.16; }
    .dot { fill: #3987e5; stroke: #1a1a19; }
  }
"""


def symlog(value: float) -> float:
    return math.copysign(math.log10(1.0 + abs(value) / LINEAR), value)


def scaler(lo: float, hi: float, start: float, length: float, flip: bool = False):
    a, b = symlog(lo), symlog(hi)

    def to_pixel(value: float) -> float:
        frac = (symlog(min(max(value, lo), hi)) - a) / (b - a)
        return start + (1.0 - frac if flip else frac) * length
    return to_pixel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trajectory", default="oracle")
    args = parser.parse_args()

    folder = Path(args.dir) if Path(args.dir).is_absolute() else PROJECT_ROOT / args.dir
    policy_path = Path(args.policy) if Path(args.policy).is_absolute() else PROJECT_ROOT / args.policy
    policy = CornerDrivePolicy.from_json(json.loads(policy_path.read_text())["policies"]["loss"])
    units = load_units(folder, method=args.trajectory, scenario="mixed")
    if not units:
        raise SystemExit(f"no {args.trajectory} units in {folder}")
    types = np.concatenate([type_of(u).ravel() for u in units])
    dm = np.concatenate([drifts(u, policy.drift_mode, policy.audit_step)[0].ravel() for u in units])
    dc = np.concatenate([drifts(u, policy.drift_mode, policy.audit_step)[1].ravel() for u in units])
    verdicts = dual_verdict(dm, dc, np.zeros(len(dm), dtype=bool), policy.verdict)
    v = policy.verdict
    rng = np.random.default_rng(0)

    rows = math.ceil(len(TYPES) / COLS)
    width = LEFT + COLS * PANEL_W + (COLS - 1) * GAP_X + 24
    height = TOP + rows * PANEL_H + (rows - 1) * GAP_Y + 62
    seeds = sorted(u["meta"]["seed"] for u in units)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="Verdict plane of the dual-dataset test, one panel per update type">',
        f"<style>{STYLE}</style>",
        f'<rect class="surface" width="{width}" height="{height}"/>',
        f'<text class="ink" x="{LEFT}" y="30" font-size="16" font-weight="600">'
        "Where each kind of update lands in the dual-dataset test</text>",
        f'<text class="ink2" x="{LEFT}" y="50" font-size="12">One dot = one client update, placed '
        f"by the relative loss drift of its replica (step {policy.audit_step:g}) on the two "
        "golden sets.</text>",
        f'<text class="ink2" x="{LEFT}" y="66" font-size="12">{len(seeds)} seeds '
        f"({seeds[0]}–{seeds[-1]}), {args.trajectory} trajectory. "
        "Grey areas are rejected as Fraud.</text>",
    ]

    for i, name in enumerate(TYPES):
        col, row = i % COLS, i // COLS
        x0 = LEFT + col * (PANEL_W + GAP_X)
        y0 = TOP + row * (PANEL_H + GAP_Y)
        sx = scaler(*X_RANGE, x0, PANEL_W)
        sy = scaler(*Y_RANGE, y0, PANEL_H, flip=True)
        mask = types == name
        total = int(mask.sum())
        share = {k: float((verdicts[mask] == k).mean()) for k in ("RARITY", "HONEST", "FRAUD")}
        parts = [f"{label} {100 * share[key]:.0f}%" for key, label in
                 (("RARITY", "Rarity"), ("HONEST", "Honest"), ("FRAUD", "Fraud")) if share[key] >= 0.005]

        out.append(f'<text class="ink" x="{x0}" y="{y0 - 26}" font-size="13" font-weight="600">'
                   f"{TYPE_LABELS[name]}</text>")
        out.append(f'<text class="ink2" x="{x0}" y="{y0 - 10}" font-size="11">'
                   f"n = {total:,} · verdict: {' · '.join(parts)}</text>")

        # rejected regions: main-task harm (right of theta_tol), corner harm (above theta_ch)
        cut_x, cut_harm, cut_rare = sx(v.theta_tol), sy(v.theta_corner_harm), sy(v.theta_rare)
        out.append(f'<rect class="reject" x="{cut_x:.1f}" y="{y0}" width="{x0 + PANEL_W - cut_x:.1f}" '
                   f'height="{PANEL_H}"/>')
        out.append(f'<rect class="reject" x="{x0}" y="{y0}" width="{cut_x - x0:.1f}" '
                   f'height="{cut_harm - y0:.1f}"/>')
        for tick in TICKS:
            if X_RANGE[0] <= tick <= X_RANGE[1]:
                out.append(f'<line class="grid" x1="{sx(tick):.1f}" y1="{y0}" x2="{sx(tick):.1f}" '
                           f'y2="{y0 + PANEL_H}"/>')
                out.append(f'<text class="muted tick" x="{sx(tick):.1f}" y="{y0 + PANEL_H + 14}" '
                           f'text-anchor="middle">{tick:+g}</text>' if tick else
                           f'<text class="muted tick" x="{sx(tick):.1f}" y="{y0 + PANEL_H + 14}" '
                           f'text-anchor="middle">0</text>')
            if Y_RANGE[0] <= tick <= Y_RANGE[1]:
                out.append(f'<line class="grid" x1="{x0}" y1="{sy(tick):.1f}" x2="{x0 + PANEL_W}" '
                           f'y2="{sy(tick):.1f}"/>')
                if col == 0:
                    label = f"{tick:+g}" if tick else "0"
                    out.append(f'<text class="muted tick" x="{x0 - 8}" y="{sy(tick) + 3.5:.1f}" '
                               f'text-anchor="end">{label}</text>')
        out.append(f'<rect class="axis" x="{x0}" y="{y0}" width="{PANEL_W}" height="{PANEL_H}"/>')
        out.append(f'<line class="cut" x1="{cut_x:.1f}" y1="{y0}" x2="{cut_x:.1f}" y2="{y0 + PANEL_H}"/>')
        out.append(f'<line class="cut" x1="{x0}" y1="{cut_harm:.1f}" x2="{cut_x:.1f}" y2="{cut_harm:.1f}"/>')
        out.append(f'<line class="cut" x1="{x0}" y1="{cut_rare:.1f}" x2="{cut_x:.1f}" y2="{cut_rare:.1f}"/>')

        picked = np.flatnonzero(mask)
        if len(picked) > MAX_POINTS:
            picked = rng.choice(picked, MAX_POINTS, replace=False)
        for j in picked:
            out.append(f'<circle class="dot" cx="{sx(dm[j]):.1f}" cy="{sy(dc[j]):.1f}" r="3">'
                       f"<title>{TYPE_LABELS[name]}: normal set {dm[j]:+.3f}, rare set {dc[j]:+.3f}, "
                       f"verdict {str(verdicts[j]).title()}</title></circle>")

        # region names, once per panel, in muted ink
        out.append(f'<text class="muted" x="{x0 + 6}" y="{y0 + 14}" font-size="10">Fraud: corner harm</text>')
        out.append(f'<text class="muted" x="{x0 + PANEL_W - 6}" y="{y0 + PANEL_H - 8}" font-size="10" '
                   f'text-anchor="end">Fraud: main harm</text>')
        out.append(f'<text class="muted" x="{x0 + 6}" y="{y0 + PANEL_H - 8}" font-size="10">Rarity</text>')
        out.append(f'<text class="muted" x="{x0 + 6}" y="{(cut_harm + cut_rare) / 2 + 3.5:.1f}" '
                   f'font-size="10">Honest</text>')

    axis_y = TOP + rows * PANEL_H + (rows - 1) * GAP_Y + 40
    centre = LEFT + (COLS * PANEL_W + (COLS - 1) * GAP_X) / 2
    out.append(f'<text class="ink2" x="{centre}" y="{axis_y}" font-size="12" text-anchor="middle">'
               "Relative loss drift on the golden normal set (ΔL_main / L_main); "
               f"tolerance θ_tol = {v.theta_tol:+.3f}</text>")
    mid_y = TOP + (rows * PANEL_H + (rows - 1) * GAP_Y) / 2
    out.append(f'<text class="ink2" x="16" y="{mid_y}" font-size="12" text-anchor="middle" '
               f'transform="rotate(-90 16 {mid_y})">Relative loss drift on the golden rare set '
               f"(ΔL_corner / L_corner); θ_rare = {v.theta_rare:+.3f}, θ_corner-harm = "
               f"{v.theta_corner_harm:+.3f}</text>")
    out.append(f'<text class="muted" x="{LEFT}" y="{axis_y + 18}" font-size="10">'
               f"Symmetric-log axes (linear inside ±{LINEAR}); values beyond the range sit on the edge. "
               f"Up to {MAX_POINTS} updates drawn per panel; the verdict shares use all of them.</text>")
    out.append("</svg>")

    target = Path(args.out) if Path(args.out).is_absolute() else PROJECT_ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {target} ({target.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
