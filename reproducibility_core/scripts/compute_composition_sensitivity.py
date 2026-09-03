#!/usr/bin/env python3
"""Compute fixed-manifest leave-one-task/category-out sensitivity metrics.

The analysis is descriptive. It does not resample or infer to unseen tasks.
Primary metrics describe A/B success-set discordance; rate contrasts are kept as
secondary diagnostics. Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_KEYS = {
    "qwen": "Qwen3-VL-8B",
    "gemma": "Gemma-3-12B-it",
    "internvl": "InternVL3-8B",
}
STRUCTURAL_FAILURES = {"extra_phase", "missing_phase", "phase_skill_mismatch"}
ATTRIBUTE_ERROR_DETAILS = {
    "'BilliardBall' object has no attribute 'get_place_point'",
    "'Fruit' object has no attribute 'get_place_point'",
    "'Toy' object has no attribute 'get_place_point'",
    "'BilliardTable' object has no attribute 'get_grasped_keypoints'",
}
PRE_NATIVE_CLASSES = {
    "output_schema_failure",
    "procedural_structure_failure",
    "target_context_failure",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"unexpected boolean value: {value!r}")


def first_attempt_success(row: dict[str, str]) -> bool:
    return as_bool(row["procedural_success"]) and int(row["native_selected_attempt"]) == 1


def adjudicated_failure_class(row: dict[str, str]) -> str:
    """Mirror analysis/reproduce_oracle_statistics.py assignment precedence."""

    success = first_attempt_success(row)
    parse_ok = as_bool(row["model_parse_ok"])
    structural = row.get("actual_failure_subtype", "") in STRUCTURAL_FAILURES
    gate_ok = as_bool(row["actual_gate1_ok"])
    native_ok = as_bool(row["actual_native_ok"])

    failure_class = "condition_failure"
    if success:
        failure_class = "success"
    if not parse_ok:
        failure_class = "output_schema_failure"
    if parse_ok and structural:
        failure_class = "procedural_structure_failure"
    if parse_ok and not structural and not gate_ok:
        failure_class = "target_context_failure"
    if gate_ok and not native_ok:
        failure_class = "native_execution_failure"
    if native_ok and not success:
        failure_class = "condition_failure"

    if (
        row.get("actual_failure_code") == "C5"
        and row.get("actual_failure_subtype") == "AttributeError"
        and row.get("actual_failure_detail", "") in ATTRIBUTE_ERROR_DETAILS
    ):
        failure_class = "target_context_failure"
    return failure_class


def indexed(rows: list[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id in output:
            raise ValueError(f"{label}: duplicate case_id {case_id}")
        output[case_id] = row
    if len(output) != 2_133:
        raise ValueError(f"{label}: expected 2,133 unique cases, found {len(output)}")
    return output


SYMBOLIC_ENDPOINTS = ("pm100", "strict", "strict_executor_filtered")
# Pooled A/B counts of the authoritative tables under the released PM=100 label.
# Strict-tier runs (artifact v1.7) record their counts in the summary instead.
EXPECTED_POOLED_COUNTS = {
    "pm100": {"both": 74, "a_only": 515, "b_only": 103, "neither": 5_707},
}


def load_strict_lookup(repo_root: Path) -> tuple[dict[tuple[str, str], bool], Path]:
    path = repo_root / "reproducibility_core" / "cross_scoring" / "full_to_symbolic_projection_rows.csv"
    lookup = {
        (row["model"], row["case_id"]): as_bool(row["original_strict_sequence_success"])
        for row in read_csv(path)
    }
    if len(lookup) != 6_399:
        raise ValueError(f"cross-scoring row table has {len(lookup)}, not 6,399 rows")
    return lookup, path


def load_model_frame(
    repo_root: Path,
    model_key: str,
    strict_lookup: dict[tuple[str, str], bool] | None = None,
) -> tuple[list[dict[str, Any]], list[Path]]:
    data_dir = repo_root / "analysis_data"
    paths = [
        data_dir / f"{model_key}_A.csv",
        data_dir / f"{model_key}_B.csv",
        data_dir / f"{model_key}_phase_oracle.csv",
        data_dir / f"{model_key}_target_oracle.csv",
    ]
    a, b, phase, target = (indexed(read_csv(path), f"{model_key}/{path.stem}") for path in paths)
    case_ids = set(a)
    for label, mapping in (("B", b), ("phase", phase), ("target", target)):
        if set(mapping) != case_ids:
            raise ValueError(f"{model_key}/{label}: case set differs from A")

    frame: list[dict[str, Any]] = []
    for case_id, a_row in a.items():
        b_row = b[case_id]
        phase_row = phase[case_id]
        target_row = target[case_id]
        for label, row in (("B", b_row), ("phase", phase_row), ("target", target_row)):
            if row["category"] != a_row["category"] or row["task_name"] != a_row["task_name"]:
                raise ValueError(f"{model_key}/{label}/{case_id}: category or task mismatch")
        frame.append(
            {
                "case_id": case_id,
                "category": a_row["category"],
                "task_name": a_row["task_name"],
                "task_cluster": f"{a_row['category']}/{a_row['task_name']}",
                "model": MODEL_KEYS[model_key],
                "a_success": (
                    as_bool(a_row["type_normalized_symbolic_success"])
                    if strict_lookup is None
                    else strict_lookup[(MODEL_KEYS[model_key], case_id)]
                ),
                "b_success": first_attempt_success(b_row),
                "phase_success": first_attempt_success(phase_row),
                "target_success": first_attempt_success(target_row),
                "b_adjudicated_class": adjudicated_failure_class(b_row),
            }
        )
    return frame, paths


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def metrics(frame: list[dict[str, Any]], scope: str, exclusion_type: str, excluded_unit: str) -> dict[str, object]:
    both = sum(row["a_success"] and row["b_success"] for row in frame)
    a_only = sum(row["a_success"] and not row["b_success"] for row in frame)
    b_only = sum(not row["a_success"] and row["b_success"] for row in frame)
    neither = sum(not row["a_success"] and not row["b_success"] for row in frame)
    union = both + a_only + b_only
    failure_rows = [row for row in frame if not row["b_success"]]
    pre_native = sum(row["b_adjudicated_class"] in PRE_NATIVE_CLASSES for row in failure_rows)
    n = len(frame)
    a_rate = mean_bool(frame, "a_success")
    b_rate = mean_bool(frame, "b_success")
    phase_rate = mean_bool(frame, "phase_success")
    target_rate = mean_bool(frame, "target_success")
    return {
        "scope": scope,
        "exclusion_type": exclusion_type,
        "excluded_unit": excluded_unit,
        "retained_rows": n,
        "retained_task_clusters": len({row["task_cluster"] for row in frame}),
        "retained_categories": len({row["category"] for row in frame}),
        "both": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "union_success": union,
        "success_set_jaccard": both / union if union else None,
        "single_interface_success_count": a_only + b_only,
        "single_interface_share_union": (a_only + b_only) / union if union else None,
        "a_only_rate_all": a_only / n,
        "b_only_rate_all": b_only / n,
        "a_only_share_union": a_only / union if union else None,
        "b_only_share_union": b_only / union if union else None,
        "success_sets_nonnested": a_only > 0 and b_only > 0,
        "a_success_rate": a_rate,
        "b_success_rate": b_rate,
        "normalized_a_minus_b": a_rate - b_rate,
        "phase_success_rate": phase_rate,
        "target_success_rate": target_rate,
        "phase_minus_full_b": phase_rate - b_rate,
        "target_minus_full_b": target_rate - b_rate,
        "full_b_failure_count": len(failure_rows),
        "pre_native_responsibility_count": pre_native,
        "pre_native_share_full_b_failures": pre_native / len(failure_rows) if failure_rows else None,
        "pre_native_share_all_full_b_outputs": pre_native / n,
    }


def metric_ranges(rows: list[dict[str, object]], exclusion_type: str) -> list[dict[str, object]]:
    primary = [
        "success_set_jaccard",
        "single_interface_share_union",
        "a_only_rate_all",
        "b_only_rate_all",
        "pre_native_share_full_b_failures",
        "pre_native_share_all_full_b_outputs",
    ]
    secondary = ["normalized_a_minus_b", "phase_minus_full_b", "target_minus_full_b"]
    subset = [row for row in rows if row["exclusion_type"] == exclusion_type]
    output: list[dict[str, object]] = []
    for scope in ["pooled", *MODEL_KEYS.values()]:
        scope_rows = [row for row in subset if row["scope"] == scope]
        for family, names in (("primary", primary), ("secondary", secondary)):
            for name in names:
                values = [(str(row["excluded_unit"]), float(row[name])) for row in scope_rows if row[name] is not None]
                low_unit, low = min(values, key=lambda item: item[1])
                high_unit, high = max(values, key=lambda item: item[1])
                output.append(
                    {
                        "scope": scope,
                        "exclusion_type": exclusion_type,
                        "metric_family": family,
                        "metric": name,
                        "minimum": low,
                        "minimum_excluded_unit": low_unit,
                        "maximum": high,
                        "maximum_excluded_unit": high_unit,
                    }
                )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--symbolic-endpoint",
        choices=SYMBOLIC_ENDPOINTS,
        default="pm100",
        help="Symbolic success label for interface A: released PM=100 (default) or strict sequence equality.",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    strict_lookup = None
    if args.symbolic_endpoint != "pm100":
        strict_lookup, strict_path = load_strict_lookup(repo_root)
    model_frames: dict[str, list[dict[str, Any]]] = {}
    source_paths: list[Path] = []
    for model_key, model_name in MODEL_KEYS.items():
        frame, paths = load_model_frame(repo_root, model_key, strict_lookup)
        model_frames[model_name] = frame
        source_paths.extend(paths)
    if strict_lookup is not None:
        source_paths.append(strict_path)
    pooled = [row for frame in model_frames.values() for row in frame]

    if len({row["task_cluster"] for row in pooled}) != 34 or len({row["category"] for row in pooled}) != 5:
        raise ValueError("expected 34 task clusters and 5 categories")
    rows: list[dict[str, object]] = []
    frames = {"pooled": pooled, **model_frames}
    for scope, frame in frames.items():
        rows.append(metrics(frame, scope, "none", "none"))
        for task_cluster in sorted({row["task_cluster"] for row in frame}):
            rows.append(metrics([row for row in frame if row["task_cluster"] != task_cluster], scope, "task", task_cluster))
        for category in sorted({row["category"] for row in frame}):
            rows.append(metrics([row for row in frame if row["category"] != category], scope, "category", category))

    baseline = [row for row in rows if row["exclusion_type"] == "none"]
    pooled_baseline = next(row for row in baseline if row["scope"] == "pooled")
    actual_counts = {key: pooled_baseline[key] for key in ("both", "a_only", "b_only", "neither")}
    expected_counts = EXPECTED_POOLED_COUNTS.get(args.symbolic_endpoint)
    if expected_counts is not None and actual_counts != expected_counts:
        raise ValueError(f"pooled A/B counts conflict with authoritative table: {actual_counts}")
    if pooled_baseline["pre_native_responsibility_count"] != 5_968:
        raise ValueError("pre-native responsibility count conflicts with authoritative taxonomy")

    loto = [row for row in rows if row["exclusion_type"] == "task"]
    loco = [row for row in rows if row["exclusion_type"] == "category"]
    ranges = metric_ranges(rows, "task") + metric_ranges(rows, "category")
    write_csv(output_dir / "baseline_composition_metrics.csv", baseline)
    write_csv(output_dir / "leave_one_task_out.csv", loto)
    write_csv(output_dir / "leave_one_category_out.csv", loco)
    write_csv(output_dir / "composition_metric_ranges.csv", ranges)

    checksums = []
    for path in sorted(set(source_paths)):
        checksums.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "sha256": sha256(path),
                "rows": len(read_csv(path)),
            }
        )
    write_csv(output_dir / "source_input_checksums.csv", checksums)

    pooled_loto = [row for row in loto if row["scope"] == "pooled"]
    pooled_loco = [row for row in loco if row["scope"] == "pooled"]
    summary = {
        "status": "passed",
        "symbolic_endpoint": args.symbolic_endpoint,
        "pooled_ab_counts": actual_counts,
        "interpretation": "fixed-manifest task/category-composition sensitivity; not inference to unseen tasks",
        "model_count": 3,
        "cases_per_model": 2_133,
        "pooled_model_case_rows": 6_399,
        "task_clusters": 34,
        "categories": 5,
        "baseline": pooled_baseline,
        "loto": {
            "pooled_exclusions": len(pooled_loto),
            "all_success_sets_nonnested": all(bool(row["success_sets_nonnested"]) for row in pooled_loto),
            "all_single_interface_majority": all(float(row["single_interface_share_union"]) > 0.5 for row in pooled_loto),
        },
        "loco": {
            "pooled_exclusions": len(pooled_loco),
            "all_success_sets_nonnested": all(bool(row["success_sets_nonnested"]) for row in pooled_loco),
            "all_single_interface_majority": all(float(row["single_interface_share_union"]) > 0.5 for row in pooled_loco),
        },
        "labels_modified": 0,
    }
    (output_dir / "composition_sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

