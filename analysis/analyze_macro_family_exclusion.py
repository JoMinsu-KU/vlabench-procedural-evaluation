from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


MACRO_FAMILIES = {
    "insert_flower_family": (
        "insert_bloom_flower",
        "insert_flower",
        "insert_flower_common_sense",
        "insert_flower_semantic",
    ),
    "add_condiment_family": (
        "add_condiment",
        "add_condiment_common_sense",
        "add_condiment_semantic",
        "add_condiment_spatial",
    ),
}

TRUE = "true"


def as_bool(value: str) -> bool:
    return value.strip().lower() == TRUE


def pct(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.2f}" if denominator else "NA"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]], analysis_set: str, model: str) -> dict[str, object]:
    n = len(rows)
    projectable = sum(as_bool(row["projection_supported"]) for row in rows)
    original_success = sum(as_bool(row["original_symbolic_success"]) for row in rows)
    projected_success = sum(as_bool(row["projected_full_symbolic_success"]) for row in rows)
    projected_success_full_failure = sum(as_bool(row["projected_success_full_failure"]) for row in rows)
    both = sum(
        as_bool(row["original_symbolic_success"])
        and as_bool(row["projected_full_symbolic_success"])
        for row in rows
    )
    original_only = original_success - both
    projected_only = projected_success - both
    union = both + original_only + projected_only
    return {
        "analysis_set": analysis_set,
        "model": model,
        "n": n,
        "projection_possible_n": projectable,
        "projection_coverage_pct": pct(projectable, n),
        "projected_symbolic_success_n": projected_success,
        "projected_symbolic_success_overall_pct": pct(projected_success, n),
        "projected_symbolic_success_among_projectable_pct": pct(projected_success, projectable),
        "projected_success_full_failure_n": projected_success_full_failure,
        "original_symbolic_success_n": original_success,
        "original_projected_both_n": both,
        "original_only_n": original_only,
        "projected_only_n": projected_only,
        "original_projected_jaccard_pct": pct(both, union),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute frozen-output cross-scoring after excluding source-defined macro families."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to reproducibility_core/cross_scoring under --repo-root.",
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "reproducibility_core" / "cross_scoring"
    )
    input_path = output_dir / "full_to_symbolic_projection_rows.csv"
    rows = read_rows(input_path)
    if len(rows) != 6_399:
        raise ValueError(f"Expected 6,399 frozen rows, found {len(rows)}")

    excluded_tasks = {task for tasks in MACRO_FAMILIES.values() for task in tasks}
    observed_tasks = {row["task_name"] for row in rows}
    observed_excluded_tasks = sorted(excluded_tasks & observed_tasks)
    excluded_rows = [row for row in rows if row["task_name"] in excluded_tasks]
    retained_rows = [row for row in rows if row["task_name"] not in excluded_tasks]

    usage_rows: list[dict[str, object]] = []
    for family, task_names in MACRO_FAMILIES.items():
        for task_name in task_names:
            task_rows = [row for row in rows if row["task_name"] == task_name]
            if not task_rows:
                continue
            statuses = Counter(row["projection_status"] for row in task_rows)
            macro_mapped = sum(status.startswith("projected_") and status.endswith("_macro") for status in statuses.elements())
            projected_success = sum(as_bool(row["projected_full_symbolic_success"]) for row in task_rows)
            downstream = sum(as_bool(row["projected_success_full_failure"]) for row in task_rows)
            usage_rows.append(
                {
                    "family": family,
                    "task_name": task_name,
                    "rows": len(task_rows),
                    "projectable_rows": sum(as_bool(row["projection_supported"]) for row in task_rows),
                    "macro_mapped_rows": macro_mapped,
                    "projected_symbolic_successes": projected_success,
                    "projected_success_full_failures": downstream,
                }
            )

    models = sorted({row["model"] for row in rows})
    sensitivity_rows: list[dict[str, object]] = []
    for analysis_set, selected_rows in (
        ("full_retained_manifest", rows),
        ("macro_family_excluded", retained_rows),
    ):
        for model in models:
            sensitivity_rows.append(
                summarize(
                    [row for row in selected_rows if row["model"] == model],
                    analysis_set,
                    model,
                )
            )
        sensitivity_rows.append(summarize(selected_rows, analysis_set, "Pooled"))

    usage_fields = [
        "family",
        "task_name",
        "rows",
        "projectable_rows",
        "macro_mapped_rows",
        "projected_symbolic_successes",
        "projected_success_full_failures",
    ]
    sensitivity_fields = [
        "analysis_set",
        "model",
        "n",
        "projection_possible_n",
        "projection_coverage_pct",
        "projected_symbolic_success_n",
        "projected_symbolic_success_overall_pct",
        "projected_symbolic_success_among_projectable_pct",
        "projected_success_full_failure_n",
        "original_symbolic_success_n",
        "original_projected_both_n",
        "original_only_n",
        "projected_only_n",
        "original_projected_jaccard_pct",
    ]
    write_csv(output_dir / "macro_mapping_usage_counts.csv", usage_fields, usage_rows)
    write_csv(
        output_dir / "macro_family_exclusion_sensitivity.csv",
        sensitivity_fields,
        sensitivity_rows,
    )

    spec = {
        "analysis_version": "macro_family_exclusion_v1",
        "input": "reproducibility_core/cross_scoring/full_to_symbolic_projection_rows.csv",
        "exclusion_unit": "entire task family for every model-case row",
        "families": {key: list(value) for key, value in MACRO_FAMILIES.items()},
        "observed_excluded_tasks": observed_excluded_tasks,
        "excluded_rows": len(excluded_rows),
        "retained_rows": len(retained_rows),
        "model_inference_calls": 0,
        "native_rollouts": 0,
        "success_labels_modified": 0,
        "projection_rules_modified": False,
    }
    (output_dir / "macro_family_exclusion_spec.json").write_text(
        json.dumps(spec, indent=2) + "\n", encoding="utf-8"
    )

    full_pooled = next(
        row
        for row in sensitivity_rows
        if row["analysis_set"] == "full_retained_manifest" and row["model"] == "Pooled"
    )
    excluded_pooled = next(
        row
        for row in sensitivity_rows
        if row["analysis_set"] == "macro_family_excluded" and row["model"] == "Pooled"
    )
    summary = {
        "status": "passed",
        "analysis_version": "macro_family_exclusion_v1",
        "full_manifest": full_pooled,
        "macro_family_excluded": excluded_pooled,
        "jaccard_absolute_change_pp": round(
            float(excluded_pooled["original_projected_jaccard_pct"])
            - float(full_pooled["original_projected_jaccard_pct"]),
            2,
        ),
        "downstream_cases_retained_pct": round(
            100.0
            * int(excluded_pooled["projected_success_full_failure_n"])
            / int(full_pooled["projected_success_full_failure_n"]),
            2,
        ),
        "qualitative_findings_preserved": (
            int(excluded_pooled["original_only_n"]) > 0
            and int(excluded_pooled["projected_only_n"]) > 0
            and int(excluded_pooled["projected_success_full_failure_n"]) > 0
        ),
        "provenance": (
            "This deterministic sensitivity analysis does not modify the retained manifest, "
            "model outputs, native traces, original success labels, or v1.3 projection rules."
        ),
    }
    (output_dir / "macro_family_exclusion_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
