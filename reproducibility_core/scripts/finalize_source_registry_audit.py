#!/usr/bin/env python3
"""Apply explicit author-reviewed source-family rules to the draft crosswalk."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_target_phases(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.split(":", 1)[0]) for item in value.split(";") if item]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-csv", type=Path, required=True)
    parser.add_argument("--review-rules", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rules_document = json.loads(args.review_rules.read_text(encoding="utf-8"))
    rules = rules_document["families"]
    with args.draft_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    errors: list[str] = []
    d1_tasks: list[str] = []
    for row in rows:
        task = row["task_name"]
        defining_class = row["effective_defining_class"]
        rule = rules.get(defining_class)
        if rule is None:
            errors.append(f"{task}: no reviewed rule for {defining_class}")
            continue

        source_sequence = row["source_call_order"].split(" -> ") if row["source_call_order"] else []
        if source_sequence != rule["expected_sequence"]:
            errors.append(f"{task}: reviewed sequence does not match structural source extraction")
        if row["effective_method_source"] != rule["source"]:
            errors.append(f"{task}: reviewed source file does not match structural source extraction")
        if parse_target_phases(row["registry_target_taking_phases"]) != rule["target_taking_phases"]:
            errors.append(f"{task}: reviewed target-taking phases do not match the frozen registry")
        if row["structural_check"] != "PASS":
            errors.append(f"{task}: structural check is not PASS")
        if row["source_control_flow"]:
            errors.append(f"{task}: unresolved expert-method control flow")

        inherited = row["registered_class"] != row["effective_defining_class"]
        row["semantic_review"] = "PASS_SOURCE_LINKED_REVIEW"
        row["mapping_rationale"] = rule["rationale"]
        if inherited:
            row["discrepancy_class"] = "D1"
            row["resolution"] = (
                "The registered task class inherits the reviewed expert method. "
                "The crosswalk records both the registered class and the effective defining class/line."
            )
            row["impact"] = "provenance documentation only; no registry sequence or result label changed"
            d1_tasks.append(task)
        else:
            row["discrepancy_class"] = "NONE"
            row["resolution"] = "No discrepancy."
            row["impact"] = "none"

    if errors:
        raise SystemExit("\n".join(errors))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_csv = args.output_dir / "source_registry_crosswalk.csv"
    with final_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "source_registry_crosswalk.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "audit_status": "PASS",
        "review_type": rules_document["review_type"],
        "source_commit": rules_document["source_commit"],
        "retained_task_count": len(rows),
        "source_traceable_task_count": sum(bool(row["source_permalink"]) for row in rows),
        "structural_pass_task_count": sum(row["structural_check"] == "PASS" for row in rows),
        "semantic_review_pass_task_count": sum(row["semantic_review"] == "PASS_SOURCE_LINKED_REVIEW" for row in rows),
        "d1_documentation_task_count": len(d1_tasks),
        "d1_documentation_tasks": d1_tasks,
        "d2_equivalent_abstraction_task_count": 0,
        "d3_substantive_mismatch_task_count": 0,
        "unresolved_task_count": 0,
        "unresolved_case_to_branch_assignment_count": 0,
        "procedure_uniqueness_assessed": rules_document["procedure_uniqueness_assessed"],
        "alternative_validity_assessed": rules_document["alternative_validity_assessed"],
        "independent_human_review": rules_document["independent_human_review"],
        "review_note": rules_document["review_note"],
        "registry_modified": False,
        "result_labels_modified": 0
    }
    (args.output_dir / "registry_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
