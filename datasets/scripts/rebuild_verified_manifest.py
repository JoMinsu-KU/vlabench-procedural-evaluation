from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def true_value(value: str) -> bool:
    return value.strip().lower() == "true"


def case_sort_key(case_id: str) -> tuple[str, str, int]:
    category, task_name, example_name = case_id.split("/")
    return category, task_name, int(example_name.removeprefix("example"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the 2,133-case manifest from the two-replay audit.")
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    with args.audit_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 4_000:
        raise ValueError(f"expected 4,000 non-complex cases, found {len(rows)}")

    selected = sorted(
        (
            row["sample_id"]
            for row in rows
            if true_value(row["repeat_1_success"]) and true_value(row["repeat_2_success"])
        ),
        key=case_sort_key,
    )
    if len(selected) != 2_133:
        raise ValueError(f"expected 2,133 two-replay successes, found {len(selected)}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps({"input_cases": len(rows), "selected_cases": len(selected), "status": "passed"}, indent=2))


if __name__ == "__main__":
    main()
