"""VincioBench budget gate: fails the build on performance regression.

Compares a VincioBench report against ``budgets.json`` and exits non-zero if
any budget is breached, so CI blocks merges that regress latency,
throughput, token efficiency, or quality floors.

Usage::

    python benchmarks/vinciobench.py                      # produce a report
    python benchmarks/check_budgets.py                    # gate the latest report
    python benchmarks/check_budgets.py path/to/report.json
    python benchmarks/check_budgets.py --budgets custom_budgets.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
DEFAULT_REPORT = HERE / "results" / "vinciobench_latest.json"
DEFAULT_BUDGETS = HERE / "budgets.json"


def resolve(report: dict[str, Any], dotted: str) -> Any:
    node: Any = report
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def check(value: Any, bound: dict[str, Any]) -> tuple[bool, str]:
    if value is None:
        return False, "metric missing from report"
    if "eq" in bound and value != bound["eq"]:
        return False, f"expected == {bound['eq']}, got {value}"
    if "gte" in bound and not (isinstance(value, (int, float)) and value >= bound["gte"]):
        return False, f"expected >= {bound['gte']}, got {value}"
    if "lte" in bound and not (isinstance(value, (int, float)) and value <= bound["lte"]):
        return False, f"expected <= {bound['lte']}, got {value}"
    return True, "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate a VincioBench report against budgets.")
    parser.add_argument("report", nargs="?", default=str(DEFAULT_REPORT))
    parser.add_argument("--budgets", default=str(DEFAULT_BUDGETS))
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.is_file():
        print(f"report not found: {report_path} (run benchmarks/vinciobench.py first)", file=sys.stderr)
        return 2
    report = json.loads(report_path.read_text())
    budgets = json.loads(Path(args.budgets).read_text())["budgets"]
    if not budgets:
        print("no budgets defined — nothing to check", file=sys.stderr)
        return 0

    failures = 0
    skipped = 0
    width = max(len(k) for k in budgets)
    for dotted, bound in budgets.items():
        value = resolve(report, dotted)
        family = dotted.split(".")[1] if dotted.startswith("families.") else None
        if value is None and family and family not in report.get("families", {}):
            # Family not in this report (partial run): skip, don't fail.
            print(f"SKIP  {dotted:<{width}}  family {family!r} not in report")
            skipped += 1
            continue
        ok, detail = check(value, bound)
        status = "PASS" if ok else "FAIL"
        print(f"{status}  {dotted:<{width}}  {detail}" + (f" (value={value})" if ok else ""))
        if not ok:
            failures += 1

    total = len(budgets)
    print(
        f"\n{total - failures - skipped} passed, {failures} failed, {skipped} skipped "
        f"(report: {report_path.name})"
    )
    if failures:
        print("benchmark budgets breached — build should fail", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
