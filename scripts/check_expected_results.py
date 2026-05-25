#!/usr/bin/env python3
"""Check generated reproduction outputs against expected metric tolerances."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPECTED = PROJECT_ROOT / "results" / "expected_results.csv"

KEY_COLUMNS = (
    "method",
    "setting",
    "config",
    "proxy",
    "proxy_type",
    "rho",
    "method_or_setting",
)

METRIC_ALIASES = {
    "macro_fraud_survival": ("fraud_survival", "fraud_survival_rate_avg_mean"),
    "macro_rarity_retention": ("rarity_retention", "rarity_retention_rate_avg_mean"),
}


@dataclass
class CheckResult:
    paper_item: str
    method_or_setting: str
    metric: str
    expected_value: float
    tolerance: float
    observed_value: float | None
    delta: float | None
    source_file: str
    status: str
    message: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate generated reproduction metrics against expected values."
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=DEFAULT_EXPECTED,
        help="Expected-results manifest CSV.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root used to resolve relative source_file paths.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print failures.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_source(project_root: Path, source_file: str) -> Path:
    path = Path(source_file)
    if path.is_absolute():
        return path
    return project_root / path


def canonical_label(value: str) -> str:
    label = str(value).strip().lower()
    if label.startswith("rho="):
        label = label.split("=", 1)[1].strip()
    try:
        number = float(label)
    except ValueError:
        return " ".join(label.split())
    return f"{number:.8g}"


def labels_match(left: str, right: str) -> bool:
    return canonical_label(left) == canonical_label(right)


def find_matching_row(rows: list[dict[str, str]], label: str) -> dict[str, str] | None:
    for row in rows:
        for column in KEY_COLUMNS:
            if column in row and labels_match(row.get(column, ""), label):
                return row
    return None


def metric_candidates(metric: str) -> tuple[str, ...]:
    aliases = METRIC_ALIASES.get(metric, ())
    return (metric, *aliases)


def parse_float(raw: Any) -> float:
    value = str(raw).strip()
    if "+/-" in value:
        value = value.split("+/-", 1)[0].strip()
    return float(value)


def get_metric(row: dict[str, str], metric: str) -> tuple[float | None, str | None]:
    for candidate in metric_candidates(metric):
        if candidate in row and str(row[candidate]).strip() != "":
            return parse_float(row[candidate]), candidate
    return None, None


def check_expected_row(project_root: Path, row: dict[str, str]) -> CheckResult:
    paper_item = row.get("paper_item", "")
    method_or_setting = row.get("method_or_setting", "")
    metric = row.get("metric", "")
    source_file = row.get("source_file", "")
    try:
        expected_value = float(row.get("expected_value", ""))
        tolerance = float(row.get("tolerance", ""))
    except ValueError as exc:
        return CheckResult(
            paper_item=paper_item,
            method_or_setting=method_or_setting,
            metric=metric,
            expected_value=math.nan,
            tolerance=math.nan,
            observed_value=None,
            delta=None,
            source_file=source_file,
            status="FAIL",
            message=f"invalid expected_value/tolerance: {exc}",
        )

    source_path = resolve_source(project_root, source_file)
    if not source_path.exists():
        return CheckResult(
            paper_item=paper_item,
            method_or_setting=method_or_setting,
            metric=metric,
            expected_value=expected_value,
            tolerance=tolerance,
            observed_value=None,
            delta=None,
            source_file=source_file,
            status="FAIL",
            message=f"missing source file: {source_file}",
        )

    try:
        source_rows = read_csv(source_path)
    except Exception as exc:  # pragma: no cover - defensive report path
        return CheckResult(
            paper_item=paper_item,
            method_or_setting=method_or_setting,
            metric=metric,
            expected_value=expected_value,
            tolerance=tolerance,
            observed_value=None,
            delta=None,
            source_file=source_file,
            status="FAIL",
            message=f"could not read source CSV: {exc}",
        )

    matched = find_matching_row(source_rows, method_or_setting)
    if matched is None:
        return CheckResult(
            paper_item=paper_item,
            method_or_setting=method_or_setting,
            metric=metric,
            expected_value=expected_value,
            tolerance=tolerance,
            observed_value=None,
            delta=None,
            source_file=source_file,
            status="FAIL",
            message=f"no row matched method_or_setting={method_or_setting!r}",
        )

    try:
        observed, matched_metric = get_metric(matched, metric)
    except ValueError as exc:
        return CheckResult(
            paper_item=paper_item,
            method_or_setting=method_or_setting,
            metric=metric,
            expected_value=expected_value,
            tolerance=tolerance,
            observed_value=None,
            delta=None,
            source_file=source_file,
            status="FAIL",
            message=f"metric value is not numeric: {exc}",
        )
    if observed is None:
        available = ", ".join(matched.keys())
        return CheckResult(
            paper_item=paper_item,
            method_or_setting=method_or_setting,
            metric=metric,
            expected_value=expected_value,
            tolerance=tolerance,
            observed_value=None,
            delta=None,
            source_file=source_file,
            status="FAIL",
            message=f"metric column not found; available columns: {available}",
        )

    delta = abs(observed - expected_value)
    status = "PASS" if delta <= tolerance else "FAIL"
    suffix = "" if matched_metric == metric else f" via {matched_metric}"
    return CheckResult(
        paper_item=paper_item,
        method_or_setting=method_or_setting,
        metric=metric,
        expected_value=expected_value,
        tolerance=tolerance,
        observed_value=observed,
        delta=delta,
        source_file=source_file,
        status=status,
        message=suffix.strip(),
    )


def write_json_report(path: Path, results: list[CheckResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": all(result.status == "PASS" for result in results),
        "checks": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_result(result: CheckResult) -> None:
    observed = "missing" if result.observed_value is None else f"{result.observed_value:.10g}"
    delta = "n/a" if result.delta is None else f"{result.delta:.10g}"
    print(
        f"{result.status} {result.paper_item} | {result.method_or_setting} | "
        f"{result.metric}: observed={observed}, expected={result.expected_value:.10g}, "
        f"tolerance={result.tolerance:.10g}, delta={delta} ({result.source_file})"
    )
    if result.message:
        print(f"  {result.message}")


def main() -> int:
    args = parse_args()
    expected_path = args.expected if args.expected.is_absolute() else args.project_root / args.expected
    if not expected_path.exists():
        raise FileNotFoundError(f"Expected-results manifest not found: {expected_path}")

    expected_rows = read_csv(expected_path)
    results = [check_expected_row(args.project_root, row) for row in expected_rows]
    if args.json_output is not None:
        write_json_report(args.json_output, results)

    for result in results:
        if not args.quiet or result.status != "PASS":
            print_result(result)

    failed = [result for result in results if result.status != "PASS"]
    if failed:
        print(f"\n{len(failed)} expected-result check(s) failed.")
        return 1
    print(f"\nAll {len(results)} expected-result check(s) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
