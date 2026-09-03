#!/usr/bin/env python3
"""Reproduce prevalence-aware agreement and control-subset sensitivities.

This script operates only on archived case-level outputs.  It performs no model
inference, native rollout, label repair, or target/action substitution.

Outputs:
  reproducibility_core/agreement_sensitivity/agreement_metrics.csv
  reproducibility_core/agreement_sensitivity/control_first_attempt_subset.csv
  reproducibility_core/agreement_sensitivity/projectability_failure_reasons.csv
  reproducibility_core/agreement_sensitivity/agreement_sensitivity_audit.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

BOOTSTRAP_SEED = 20260711
BOOTSTRAP_REPLICATES = 20_000
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def indexed(rows: Iterable[dict[str, str]], label: str) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in rows:
        case_id = row["case_id"]
        if case_id in output:
            raise ValueError(f"{label}: duplicate case_id {case_id}")
        output[case_id] = row
    return output


def contingency(rows: list[dict[str, Any]]) -> tuple[int, int, int, int]:
    both = sum(bool(row["left_success"]) and bool(row["right_success"]) for row in rows)
    left_only = sum(bool(row["left_success"]) and not bool(row["right_success"]) for row in rows)
    right_only = sum(not bool(row["left_success"]) and bool(row["right_success"]) for row in rows)
    neither = len(rows) - both - left_only - right_only
    return both, left_only, right_only, neither


def agreement_metrics(counts: tuple[int, int, int, int]) -> dict[str, float]:
    both, left_only, right_only, neither = counts
    n = sum(counts)
    union = both + left_only + right_only
    if n == 0:
        raise ValueError(f"agreement metric has an empty table: {counts}")

    raw_agreement = (both + neither) / n
    expected_agreement = (
        (both + left_only) * (both + right_only)
        + (right_only + neither) * (left_only + neither)
    ) / (n * n)
    mcc_denominator = math.sqrt(
        (both + left_only)
        * (both + right_only)
        * (neither + left_only)
        * (neither + right_only)
    )
    # A zero cell can occur in a task-cluster bootstrap replicate.  Apply the
    # Haldane--Anscombe correction only for the odds-ratio calculation in that
    # replicate; point estimates with nonzero cells remain uncorrected.
    odds_cells = [float(value) for value in counts]
    odds_ratio_corrected = any(value == 0 for value in odds_cells)
    if odds_ratio_corrected:
        odds_cells = [value + 0.5 for value in odds_cells]
    odds_ratio = (odds_cells[0] * odds_cells[3]) / (odds_cells[1] * odds_cells[2])

    return {
        "raw_agreement": raw_agreement,
        "positive_agreement": (
            2 * both / (2 * both + left_only + right_only)
            if (2 * both + left_only + right_only)
            else math.nan
        ),
        "negative_agreement": 2 * neither / (2 * neither + left_only + right_only),
        "cohen_kappa": (
            (raw_agreement - expected_agreement) / (1 - expected_agreement)
            if expected_agreement != 1
            else math.nan
        ),
        "mcc_phi": (
            (both * neither - left_only * right_only) / mcc_denominator
            if mcc_denominator
            else math.nan
        ),
        "odds_ratio": odds_ratio,
        "procedure_given_symbolic": (
            both / (both + left_only) if (both + left_only) else math.nan
        ),
        "procedure_given_symbolic_failure": right_only / (right_only + neither),
        "success_set_jaccard": both / union if union else math.nan,
        "single_interface_share": (left_only + right_only) / union if union else math.nan,
        "odds_ratio_zero_cell_correction": float(odds_ratio_corrected),
    }


def cluster_bootstrap(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    clusters = sorted({str(row["task_cluster"]) for row in rows})
    grouped_counts: dict[str, tuple[int, int, int, int]] = {}
    for cluster in clusters:
        grouped_counts[cluster] = contingency(
            [row for row in rows if str(row["task_cluster"]) == cluster]
        )

    rng = random.Random(BOOTSTRAP_SEED)
    metric_names = [
        "raw_agreement",
        "positive_agreement",
        "negative_agreement",
        "cohen_kappa",
        "mcc_phi",
        "odds_ratio",
        "procedure_given_symbolic",
        "procedure_given_symbolic_failure",
        "success_set_jaccard",
        "single_interface_share",
    ]
    estimates = {name: [] for name in metric_names}
    zero_cell_corrections = 0
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = [rng.choice(clusters) for _ in clusters]
        counts = tuple(
            sum(grouped_counts[str(cluster)][cell] for cluster in sampled)
            for cell in range(4)
        )
        metrics = agreement_metrics(counts)
        zero_cell_corrections += int(metrics["odds_ratio_zero_cell_correction"])
        for name in metric_names:
            if math.isfinite(metrics[name]):
                estimates[name].append(metrics[name])

    def percentile(values: list[float], probability: float) -> float:
        ordered = sorted(values)
        position = (len(ordered) - 1) * probability
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)

    intervals = {
        name: (
            percentile(values, 0.025),
            percentile(values, 0.975),
        )
        for name, values in estimates.items()
        if values
    }
    intervals["odds_ratio_zero_cell_correction_replicates"] = (
        float(zero_cell_corrections),
        float(zero_cell_corrections),
    )
    intervals["finite_replicates"] = {
        name: len(values) for name, values in estimates.items()
    }
    return intervals


# Symbolic endpoint tiers (artifact v1.7).  Column names refer to
# reproducibility_core/cross_scoring/full_to_symbolic_projection_rows.csv.
SYMBOLIC_ENDPOINTS = ("pm100", "strict", "strict_executor_filtered")
CROSS_ENDPOINT_COLUMNS = {
    "pm100": (
        "original_symbolic_success",
        "projected_full_symbolic_success",
        "projected_success_full_failure",
    ),
    "strict": (
        "original_strict_sequence_success",
        "projected_strict_sequence_success",
        "projected_strict_success_full_failure",
    ),
    "strict_executor_filtered": (
        "original_strict_sequence_success",
        "projected_strict_executor_filtered_success",
        "projected_strict_filtered_success_full_failure",
    ),
}


def load_original_rows(
    repo_root: Path,
    strict_lookup: dict[tuple[str, str], bool] | None = None,
) -> tuple[list[dict[str, Any]], list[Path]]:
    """Independently elicited symbolic versus full-procedure rows.

    With ``strict_lookup`` the symbolic label is the strict sequence-equality
    label carried in the cross-scoring row table instead of the archived
    PM=100 indicator.  Original symbolic outputs contain no executor phase
    names, so the executor-filtered tier shares the strict original label.
    """

    rows: list[dict[str, Any]] = []
    inputs: list[Path] = []
    for model_key, display_name in MODEL_KEYS.items():
        a_path = repo_root / "analysis_data" / f"{model_key}_A.csv"
        b_path = repo_root / "analysis_data" / f"{model_key}_B.csv"
        inputs.extend([a_path, b_path])
        a_rows = indexed(read_csv(a_path), f"{model_key}/symbolic")
        b_rows = indexed(read_csv(b_path), f"{model_key}/full")
        if set(a_rows) != set(b_rows) or len(a_rows) != 2_133:
            raise ValueError(f"{model_key}: symbolic/full case sets do not match at 2,133 rows")
        for case_id, a_row in a_rows.items():
            b_row = b_rows[case_id]
            if (a_row["category"], a_row["task_name"]) != (
                b_row["category"],
                b_row["task_name"],
            ):
                raise ValueError(f"{model_key}/{case_id}: task metadata mismatch")
            rows.append(
                {
                    "model": display_name,
                    "case_id": case_id,
                    "category": a_row["category"],
                    "task_name": a_row["task_name"],
                    "task_cluster": f"{a_row['category']}/{a_row['task_name']}",
                    "left_success": (
                        as_bool(a_row["type_normalized_symbolic_success"])
                        if strict_lookup is None
                        else strict_lookup[(display_name, case_id)]
                    ),
                    "right_success": first_attempt_success(b_row),
                    "full_row": b_row,
                }
            )
    return rows, inputs


def load_cross_rows(
    repo_root: Path, endpoint: str = "pm100"
) -> tuple[list[dict[str, Any]], Path]:
    path = (
        repo_root
        / "reproducibility_core"
        / "cross_scoring"
        / "full_to_symbolic_projection_rows.csv"
    )
    left_key, right_key, downstream_key = CROSS_ENDPOINT_COLUMNS[endpoint]
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        rows.append(
            {
                **row,
                "task_cluster": f"{row['category']}/{row['task_name']}",
                "left_success": as_bool(row[left_key]),
                "right_success": as_bool(row[right_key]),
                "projection_supported_bool": as_bool(row["projection_supported"]),
                "full_parser_accepted_bool": as_bool(row["full_parser_accepted"]),
                "full_procedure_success_bool": as_bool(row["full_procedure_success"]),
                "projected_success_full_failure_bool": as_bool(row[downstream_key]),
                "original_strict_bool": as_bool(row["original_strict_sequence_success"]),
            }
        )
    if len(rows) != 6_399:
        raise ValueError(f"cross-score table has {len(rows)}, not 6,399 rows")
    return rows, path


def agreement_output_rows(
    analysis_name: str,
    rows: list[dict[str, Any]],
    population_note: str,
    symbolic_endpoint: str = "pm100",
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scopes = [("Pooled fixed-backbone aggregate", rows)] + [
        (model, [row for row in rows if row["model"] == model])
        for model in MODEL_KEYS.values()
    ]
    for scope, scoped_rows in scopes:
        counts = contingency(scoped_rows)
        metrics = agreement_metrics(counts)
        intervals = cluster_bootstrap(scoped_rows)
        row: dict[str, Any] = {
            "analysis": analysis_name,
            "symbolic_endpoint": symbolic_endpoint,
            "scope": scope,
            "population_note": population_note,
            "rows": len(scoped_rows),
            "task_clusters": len({item["task_cluster"] for item in scoped_rows}),
            "both_success": counts[0],
            "left_only_success": counts[1],
            "right_only_success": counts[2],
            "neither_success": counts[3],
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        }
        for name, estimate in metrics.items():
            if name == "odds_ratio_zero_cell_correction":
                row[name] = bool(estimate)
                continue
            row[name] = estimate
            row[f"{name}_cluster_low"] = intervals[name][0]
            row[f"{name}_cluster_high"] = intervals[name][1]
        row["odds_ratio_corrected_bootstrap_replicates"] = int(
            intervals["odds_ratio_zero_cell_correction_replicates"][0]
        )
        for name, finite_count in intervals["finite_replicates"].items():
            row[f"{name}_finite_bootstrap_replicates"] = finite_count
        output.append(row)
    return output


def adjudicated_failure_class(row: dict[str, str]) -> str:
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


def control_subset_rows(
    repo_root: Path,
    original_by_endpoint: dict[str, list[dict[str, Any]]],
    cross_by_endpoint: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], set[str], list[Path]]:
    control_path = repo_root / "results" / "positive_control" / "gt_procedural_positive_control_report.csv"
    control_rows = read_csv(control_path)
    first_attempt_case_ids = {
        row["source_case_id"]
        for row in control_rows
        if int(row["native_selected_attempt"]) == 1
    }
    if len(first_attempt_case_ids) != 2_062:
        raise ValueError(
            f"positive-control first-attempt subset has {len(first_attempt_case_ids)}, not 2,062 cases"
        )

    output: list[dict[str, Any]] = []

    def append_metric(
        metric: str,
        model: str,
        full_n: int,
        full_value: float,
        subset_n: int,
        subset_value: float,
        unit: str,
        full_count: int | None = None,
        subset_count: int | None = None,
        symbolic_endpoint: str = "not_applicable",
    ) -> None:
        output.append(
            {
                "metric": metric,
                "symbolic_endpoint": symbolic_endpoint,
                "model": model,
                "full_manifest_n": full_n,
                "full_manifest_count": "" if full_count is None else full_count,
                "full_manifest_value": full_value,
                "control_first_attempt_subset_n": subset_n,
                "control_first_attempt_subset_count": "" if subset_count is None else subset_count,
                "control_first_attempt_subset_value": subset_value,
                "unit": unit,
            }
        )

    for endpoint in SYMBOLIC_ENDPOINTS:
        original_rows = original_by_endpoint[endpoint]
        cross_rows = cross_by_endpoint[endpoint]
        original_subset = [row for row in original_rows if row["case_id"] in first_attempt_case_ids]
        full_original_metrics = agreement_metrics(contingency(original_rows))
        subset_original_metrics = agreement_metrics(contingency(original_subset))
        append_metric(
            "original_symbolic_full_success_set_jaccard",
            "Pooled fixed-backbone aggregate",
            len(original_rows),
            full_original_metrics["success_set_jaccard"],
            len(original_subset),
            subset_original_metrics["success_set_jaccard"],
            "proportion",
            symbolic_endpoint=endpoint,
        )

        cross_subset = [row for row in cross_rows if row["case_id"] in first_attempt_case_ids]
        full_cross_metrics = agreement_metrics(contingency(cross_rows))
        subset_cross_metrics = agreement_metrics(contingency(cross_subset))
        append_metric(
            "original_symbolic_projected_full_jaccard",
            "Pooled fixed-backbone aggregate",
            len(cross_rows),
            full_cross_metrics["success_set_jaccard"],
            len(cross_subset),
            subset_cross_metrics["success_set_jaccard"],
            "proportion",
            symbolic_endpoint=endpoint,
        )
        full_projected_success = sum(row["right_success"] for row in cross_rows)
        subset_projected_success = sum(row["right_success"] for row in cross_subset)
        full_downstream = sum(row["projected_success_full_failure_bool"] for row in cross_rows)
        subset_downstream = sum(row["projected_success_full_failure_bool"] for row in cross_subset)
        append_metric(
            "projected_symbolic_success_followed_by_full_failure",
            "Pooled fixed-backbone aggregate",
            full_projected_success,
            full_downstream / full_projected_success,
            subset_projected_success,
            subset_downstream / subset_projected_success,
            "share_of_projected_symbolic_success",
            full_downstream,
            subset_downstream,
            symbolic_endpoint=endpoint,
        )

    stage_order = [
        "output_schema_failure",
        "procedural_structure_failure",
        "target_context_failure",
        "native_execution_failure",
        "condition_failure",
        "success",
    ]
    for model_key, display_name in MODEL_KEYS.items():
        full_path = repo_root / "analysis_data" / f"{model_key}_B.csv"
        full_rows = read_csv(full_path)
        subset_rows = [row for row in full_rows if row["case_id"] in first_attempt_case_ids]
        full_counts = Counter(adjudicated_failure_class(row) for row in full_rows)
        subset_counts = Counter(adjudicated_failure_class(row) for row in subset_rows)
        for stage in stage_order:
            append_metric(
                f"termination_{stage}",
                display_name,
                len(full_rows),
                full_counts[stage] / len(full_rows),
                len(subset_rows),
                subset_counts[stage] / len(subset_rows),
                "proportion_of_model_outputs",
                full_counts[stage],
                subset_counts[stage],
            )

        condition_rows: dict[str, list[dict[str, str]]] = {}
        for condition in ("B", "phase_oracle", "target_oracle"):
            path = repo_root / "analysis_data" / f"{model_key}_{condition}.csv"
            condition_rows[condition] = read_csv(path)
        rates: dict[str, tuple[float, float, int, int]] = {}
        for condition, rows in condition_rows.items():
            subset = [row for row in rows if row["case_id"] in first_attempt_case_ids]
            full_count = sum(first_attempt_success(row) for row in rows)
            subset_count = sum(first_attempt_success(row) for row in subset)
            rates[condition] = (
                full_count / len(rows),
                subset_count / len(subset),
                full_count,
                subset_count,
            )
        for scaffold in ("phase_oracle", "target_oracle"):
            append_metric(
                f"{scaffold}_minus_full_procedure",
                display_name,
                len(condition_rows["B"]),
                rates[scaffold][0] - rates["B"][0],
                len(first_attempt_case_ids),
                rates[scaffold][1] - rates["B"][1],
                "proportion_difference",
            )

    return output, first_attempt_case_ids, [control_path]


def projectability_reason_rows(
    repo_root: Path,
    cross_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Path], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    inputs: list[Path] = []
    exact_set_match_by_model: dict[str, bool] = {}
    for model_key, display_name in MODEL_KEYS.items():
        b_path = repo_root / "analysis_data" / f"{model_key}_B.csv"
        inputs.append(b_path)
        b_rows = indexed(read_csv(b_path), f"{model_key}/full")
        model_cross = [row for row in cross_rows if row["model"] == display_name]
        if len(model_cross) != 2_133:
            raise ValueError(f"{display_name}: cross-score rows are not 2,133")

        parser_rejected = 0
        parser_accepted_unsupported = 0
        parser_accepted_missing_or_ambiguous = 0
        other_unprojectable = 0
        projection_set = set()
        parser_failure_set = set()
        for row in model_cross:
            case_id = row["case_id"]
            full_parse_ok = as_bool(b_rows[case_id]["model_parse_ok"])
            if full_parse_ok != row["full_parser_accepted_bool"]:
                raise ValueError(f"{display_name}/{case_id}: parser acceptance mismatch")
            if not row["projection_supported_bool"]:
                projection_set.add(case_id)
                status = row["projection_status"]
                if not row["full_parser_accepted_bool"]:
                    parser_rejected += 1
                    parser_failure_set.add(case_id)
                elif "unsupported" in status:
                    parser_accepted_unsupported += 1
                elif "missing" in status or "ambiguous" in status:
                    parser_accepted_missing_or_ambiguous += 1
                else:
                    other_unprojectable += 1

        exact_match = projection_set == parser_failure_set
        exact_set_match_by_model[display_name] = exact_match
        if not exact_match:
            raise ValueError(f"{display_name}: unprojectable and parser-rejected sets differ")
        output.append(
            {
                "model": display_name,
                "frozen_full_outputs": len(model_cross),
                "projectable": sum(row["projection_supported_bool"] for row in model_cross),
                "unprojectable": len(projection_set),
                "parser_rejected": parser_rejected,
                "parser_accepted_unsupported_action": parser_accepted_unsupported,
                "parser_accepted_missing_or_ambiguous_macro_target": parser_accepted_missing_or_ambiguous,
                "other_unprojectable": other_unprojectable,
                "unprojectable_equals_parser_rejected_set": exact_match,
            }
        )
    return output, inputs, exact_set_match_by_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to reproducibility_core/agreement_sensitivity under repo root.",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else repo_root / "reproducibility_core" / "agreement_sensitivity"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cross_by_endpoint = {
        endpoint: load_cross_rows(repo_root, endpoint)[0] for endpoint in SYMBOLIC_ENDPOINTS
    }
    cross_path = load_cross_rows(repo_root, "pm100")[1]
    strict_lookup = {
        (row["model"], row["case_id"]): row["original_strict_bool"]
        for row in cross_by_endpoint["strict"]
    }
    original_pm100, original_inputs = load_original_rows(repo_root)
    original_strict, _ = load_original_rows(repo_root, strict_lookup)
    original_by_endpoint = {
        "pm100": original_pm100,
        "strict": original_strict,
        "strict_executor_filtered": original_strict,
    }
    original_rows = original_pm100
    cross_rows = cross_by_endpoint["pm100"]

    agreement_rows: list[dict[str, Any]] = []
    for endpoint in SYMBOLIC_ENDPOINTS:
        agreement_rows.extend(
            agreement_output_rows(
                "independently_elicited_symbolic_vs_full_procedure",
                original_by_endpoint[endpoint],
                "All 6,399 fixed-backbone model-case observations",
                endpoint,
            )
        )
        agreement_rows.extend(
            agreement_output_rows(
                "original_symbolic_vs_projected_full_all_rows",
                cross_by_endpoint[endpoint],
                "All 6,399 frozen full-procedure outputs; unprojectable outputs remain projected failures",
                endpoint,
            )
        )
        agreement_rows.extend(
            agreement_output_rows(
                "original_symbolic_vs_projected_full_projectable_only",
                [row for row in cross_by_endpoint[endpoint] if row["projection_supported_bool"]],
                "Selection-conditioned projection-possible subset",
                endpoint,
            )
        )
    write_csv(output_dir / "agreement_metrics.csv", agreement_rows)

    control_rows, first_attempt_case_ids, control_inputs = control_subset_rows(
        repo_root, original_by_endpoint, cross_by_endpoint
    )
    write_csv(output_dir / "control_first_attempt_subset.csv", control_rows)

    projectability_rows, projectability_inputs, exact_matches = projectability_reason_rows(
        repo_root, cross_rows
    )
    write_csv(output_dir / "projectability_failure_reasons.csv", projectability_rows)

    def pooled_row(analysis: str, endpoint: str) -> dict[str, Any]:
        return next(
            row
            for row in agreement_rows
            if row["analysis"] == analysis
            and row["symbolic_endpoint"] == endpoint
            and row["scope"] == "Pooled fixed-backbone aggregate"
        )

    pooled_original = pooled_row("independently_elicited_symbolic_vs_full_procedure", "pm100")
    pooled_cross = pooled_row("original_symbolic_vs_projected_full_all_rows", "pm100")
    metric_names = (
        "raw_agreement",
        "positive_agreement",
        "negative_agreement",
        "cohen_kappa",
        "mcc_phi",
        "odds_ratio",
        "success_set_jaccard",
    )
    endpoint_tier_audit = {
        endpoint: {
            analysis: {
                "counts": {
                    "both": pooled_row(analysis, endpoint)["both_success"],
                    "left_only": pooled_row(analysis, endpoint)["left_only_success"],
                    "right_only": pooled_row(analysis, endpoint)["right_only_success"],
                    "neither": pooled_row(analysis, endpoint)["neither_success"],
                },
                "metrics": {name: pooled_row(analysis, endpoint)[name] for name in metric_names},
                "success_set_jaccard_cluster_interval": [
                    pooled_row(analysis, endpoint)["success_set_jaccard_cluster_low"],
                    pooled_row(analysis, endpoint)["success_set_jaccard_cluster_high"],
                ],
                "cohen_kappa_cluster_interval": [
                    pooled_row(analysis, endpoint)["cohen_kappa_cluster_low"],
                    pooled_row(analysis, endpoint)["cohen_kappa_cluster_high"],
                ],
            }
            for analysis in (
                "independently_elicited_symbolic_vs_full_procedure",
                "original_symbolic_vs_projected_full_all_rows",
                "original_symbolic_vs_projected_full_projectable_only",
            )
        }
        for endpoint in SYMBOLIC_ENDPOINTS
    }
    audit = {
        "analysis_version": "agreement_and_control_subset_v2",
        "symbolic_endpoints": list(SYMBOLIC_ENDPOINTS),
        "symbolic_endpoint_note": (
            "pm100 = released type-normalized Precise Matching == 100 (v1.6 primary); "
            "strict = normalized whole-sequence equality; strict_executor_filtered = strict "
            "after removing executor-only phase names from the projected side only. "
            "The pooled_original_* and pooled_cross_* fields below remain the pm100 values."
        ),
        "endpoint_tiers": endpoint_tier_audit,
        "model_inference_calls": 0,
        "native_rollouts": 0,
        "success_labels_modified": 0,
        "bootstrap": {
            "unit": "task_cluster",
            "task_clusters": 34,
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile_95",
            "interpretation": "retained-task-composition sensitivity, not unseen-task population inference",
        },
        "fixed_backbone_interpretation": "descriptive aggregate with equal inclusion of the three selected backbones",
        "positive_control_first_attempt_case_count": len(first_attempt_case_ids),
        "unprojectable_equals_parser_rejected_set_by_model": exact_matches,
        "pooled_original_counts": {
            "both": pooled_original["both_success"],
            "symbolic_only": pooled_original["left_only_success"],
            "procedure_only": pooled_original["right_only_success"],
            "neither": pooled_original["neither_success"],
        },
        "pooled_original_metrics": {
            name: pooled_original[name]
            for name in (
                "raw_agreement",
                "positive_agreement",
                "negative_agreement",
                "cohen_kappa",
                "mcc_phi",
                "odds_ratio",
                "success_set_jaccard",
            )
        },
        "pooled_cross_metrics": {
            name: pooled_cross[name]
            for name in (
                "raw_agreement",
                "positive_agreement",
                "negative_agreement",
                "cohen_kappa",
                "mcc_phi",
                "odds_ratio",
                "success_set_jaccard",
            )
        },
        "input_sha256": {
            str(path.relative_to(repo_root)).replace("\\", "/"): sha256(path)
            for path in sorted(set(original_inputs + [cross_path] + control_inputs + projectability_inputs))
        },
        "output_sha256": {},
    }
    for name in (
        "agreement_metrics.csv",
        "control_first_attempt_subset.csv",
        "projectability_failure_reasons.csv",
    ):
        audit["output_sha256"][name] = sha256(output_dir / name)
    (output_dir / "agreement_sensitivity_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
