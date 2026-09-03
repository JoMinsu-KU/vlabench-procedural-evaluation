from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


MODELS = ("qwen", "gemma", "internvl")
FORBIDDEN_PATTERNS = {
    "external_ipv4": re.compile(r"\b(?!127\.0\.0\.1\b)(?:\d{1,3}\.){3}\d{1,3}\b"),
    "local_windows_user_path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE),
    "local_linux_user_path": re.compile(r"/home/(?!conda/)[^/\s]+", re.IGNORECASE),
    "local_wsl_mount_path": re.compile(r"/mnt/[a-z]/", re.IGNORECASE),
    "secret_assignment": re.compile(r"(?:api[_-]?key|authorization|bearer)\s*[:=]\s*[^\s\"']+", re.IGNORECASE),
    "excluded_legacy_study_term": re.compile(
        r"\bSILVER\b|\bTrack\s+[A-D]\b|replanning|metadata[-_ ]control",
        re.IGNORECASE,
    ),
}
TEXT_EXTENSIONS = {".py", ".json", ".jsonl", ".csv", ".md", ".txt", ".yml", ".yaml", ".toml", ".cff"}
PINNED_DEPENDENCY = re.compile(
    r"^[A-Za-z0-9_.-]+(?:\[[^\]]+\])?==[0-9]+(?:\.[0-9]+){1,4}(?:[+.-][A-Za-z0-9.]+)?$"
)


def count_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def count_jsonl(path: Path) -> tuple[int, int]:
    rows = 0
    case_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            case_ids.add(str(json.loads(line)["case_id"]))
    return rows, len(case_ids)


def read_single_jsonl(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f"{path} contains {len(rows)} rows, expected one")
    return rows[0]


def canonical_text_sha256(path: Path) -> str:
    """Hash UTF-8 text with LF endings so verification is cross-platform."""

    text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_pinned_dependency_version(text: str, match: re.Match[str]) -> bool:
    line_start = text.rfind("\n", 0, match.start()) + 1
    line_end = text.find("\n", match.end())
    if line_end == -1:
        line_end = len(text)
    return PINNED_DEPENDENCY.fullmatch(text[line_start:line_end].strip()) is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository_root", type=Path)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    errors: list[str] = []

    manifest = json.loads((root / "datasets" / "manifests" / "matched_2133_symbolic_procedural_cases.json").read_text(encoding="utf-8"))
    if len(manifest) != 2_133 or len(set(manifest)) != 2_133:
        errors.append("paired manifest is not 2,133 unique cases")
    rebuilt_manifest_path = root / "datasets" / "manifests" / "rebuilt_verified_subset_2133_cases.json"
    if rebuilt_manifest_path.exists():
        rebuilt_manifest = json.loads(rebuilt_manifest_path.read_text(encoding="utf-8"))
        if rebuilt_manifest != manifest:
            errors.append("manifest rebuilt from the two-replay audit does not match the released manifest")

    oracle_expected = {
        "qwen": {"phase_oracle": 427, "target_oracle": 488},
        "gemma": {"phase_oracle": 344, "target_oracle": 585},
        "internvl": {"phase_oracle": 314, "target_oracle": 191},
    }
    for model in MODELS:
        a_csv = root / "results" / "model_outputs" / model / "experiment_a" / "official_vlm_symbolic_scores.csv"
        b_csv = root / "results" / "model_outputs" / model / "experiment_b" / "procedural_vlm_execution_results.csv"
        if count_csv(a_csv) != 2_133:
            errors.append(f"{model}: Experiment A does not contain 2,133 rows")
        if count_csv(b_csv) != 2_133:
            errors.append(f"{model}: Experiment B does not contain 2,133 rows")
        for condition in ("phase_oracle", "target_oracle"):
            condition_root = root / "results" / "model_outputs" / model / "oracle_ablations" / condition
            result_csv = condition_root / "procedural_vlm_execution_results.csv"
            rows = read_csv(result_csv)
            if len(rows) != 2_133:
                errors.append(f"{model}/{condition}: result CSV does not contain 2,133 rows")
            elif [row["case_id"] for row in rows] != manifest:
                errors.append(f"{model}/{condition}: result CSV does not match manifest order")
            success_count = sum(
                row["procedural_success"].lower() == "true" and row["native_selected_attempt"] == "1"
                for row in rows
            )
            if success_count != oracle_expected[model][condition]:
                errors.append(
                    f"{model}/{condition}: expected {oracle_expected[model][condition]} successes, found {success_count}"
                )
            summary = json.loads((condition_root / "summary.json").read_text(encoding="utf-8"))
            if int(summary["procedural_success"]) != success_count:
                errors.append(f"{model}/{condition}: summary success count does not match CSV")
            jsonl_rows, jsonl_case_ids = count_jsonl(condition_root / "procedural_vlm_execution_rows.jsonl")
            if jsonl_rows != 2_133 or jsonl_case_ids != 2_133:
                errors.append(
                    f"{model}/{condition}: canonical JSONL is not 2,133 rows and case IDs"
                )

    oracle_main = read_csv(root / "results" / "reported_statistics" / "oracle_main_results_clustered.csv")
    if len(oracle_main) != 3:
        errors.append("oracle main table does not contain three model rows")
    else:
        for row in oracle_main:
            if float(row["phase_minus_full_ci_low"]) <= 0 or float(row["target_minus_full_ci_low"]) <= 0:
                errors.append(f"{row['model']}: a pointwise oracle-vs-full task-cluster interval includes zero")
            if not (
                float(row["target_minus_phase_ci_low"]) <= 0
                <= float(row["target_minus_phase_ci_high"])
            ):
                errors.append(f"{row['model']}: target-vs-phase task-cluster interval excludes zero")

    simultaneous = read_csv(
        root / "results" / "reported_statistics" / "oracle_primary_contrasts_simultaneous.csv"
    )
    if len(simultaneous) != 6:
        errors.append("oracle simultaneous table does not contain six contrasts")
    else:
        excluding = [row for row in simultaneous if row["simultaneous_excludes_zero"].lower() == "true"]
        if len(excluding) != 5:
            errors.append(f"expected five zero-excluding simultaneous intervals, found {len(excluding)}")
        internvl_target = [
            row
            for row in simultaneous
            if row["model"] == "InternVL3-8B" and row["contrast"] == "Target-oracle - Full B"
        ]
        if len(internvl_target) != 1 or not (
            float(internvl_target[0]["simultaneous_ci_low"]) <= 0
            <= float(internvl_target[0]["simultaneous_ci_high"])
        ):
            errors.append("InternVL Target-minus-Full-B simultaneous interval does not include zero")

    endpoint_sensitivity = read_csv(
        root
        / "results"
        / "reported_statistics"
        / "symbolic_endpoint_representation_sensitivity.csv"
    )
    expected_endpoint_counts = {
        "Qwen3-VL-8B": (7, 280, 80),
        "Gemma-3-12B-it": (17, 221, 57),
        "InternVL3-8B": (2, 88, 40),
    }
    if len(endpoint_sensitivity) != 3:
        errors.append("symbolic endpoint-sensitivity table does not contain three models")
    else:
        for row in endpoint_sensitivity:
            expected = expected_endpoint_counts.get(row["model"])
            actual = (
                int(row["raw_a_full_match_count"]),
                int(row["normalized_a_full_match_count"]),
                int(row["b_first_attempt_success_count"]),
            )
            if expected != actual:
                errors.append(f"{row['model']}: unexpected endpoint-sensitivity counts {actual}")
            if float(row["raw_a_minus_b"]) >= 0 or float(row["normalized_a_minus_b"]) <= 0:
                errors.append(f"{row['model']}: endpoint-sensitivity ordering did not reverse")

    full_b_taxonomy = read_csv(
        root / "results" / "reported_statistics" / "procedural_failure_outcomes.csv"
    )
    expected_full_b = {
        ("Qwen3-VL-8B", "Target/context incompatibility"): 307,
        ("Gemma-3-12B-it", "Target/context incompatibility"): 586,
        ("InternVL3-8B", "Target/context incompatibility"): 307,
        ("Qwen3-VL-8B", "Native execution failure"): 0,
        ("Gemma-3-12B-it", "Native execution failure"): 0,
        ("InternVL3-8B", "Native execution failure"): 0,
    }
    indexed_full_b = {(row["model"], row["outcome"]): row for row in full_b_taxonomy}
    for key, expected in expected_full_b.items():
        row = indexed_full_b.get(key)
        if row is None or int(row["count"]) != expected:
            errors.append(f"Full-B adjudicated taxonomy mismatch for {key}: expected {expected}")
        elif row.get("classification_layer") != "responsibility_adjudicated":
            errors.append(f"Full-B taxonomy lacks responsibility-adjudicated label for {key}")

    adjudication_rows = count_csv(
        root / "results" / "outcome_audits" / "oracle_attribute_error_adjudication.csv"
    )
    if adjudication_rows != 472:
        errors.append(f"oracle AttributeError adjudication contains {adjudication_rows}, not 472 rows")

    corrected_gate_path = (
        root / "results" / "outcome_audits" / "corrected_affordance_revalidation_20260721.json"
    )
    corrected_gate = json.loads(corrected_gate_path.read_text(encoding="utf-8"))
    corrected_expected = {
        "status": "passed",
        "unique_model_condition_case_keys": 19_197,
        "expected_adjudicated_c5_cases": 472,
        "corrected_gate_flagged_cases": 1_074,
        "already_c3_additional_flags": 602,
        "additional_non_c3_outcomes": 0,
        "flagged_original_successes": 0,
        "success_labels_changed": False,
    }
    for field, expected in corrected_expected.items():
        if corrected_gate.get(field) != expected:
            errors.append(
                f"corrected-gate field {field!r}: expected {expected!r}, found {corrected_gate.get(field)!r}"
            )
    corrected_rows = read_csv(
        root / "results" / "outcome_audits" / "corrected_affordance_revalidation_20260721.csv"
    )
    if len(corrected_rows) != 1_074:
        errors.append(f"corrected-gate mapping contains {len(corrected_rows)}, not 1,074 rows")
    if any(row["original_success"].lower() == "true" for row in corrected_rows):
        errors.append("corrected-gate mapping flags a successful row")

    schema = json.loads(
        (root / "shared" / "schemas" / "procedural_plan_schema.json").read_text(encoding="utf-8")
    )
    alternatives = schema["properties"]["steps"]["items"].get("oneOf", [])
    if len(alternatives) != 2:
        errors.append("procedural schema does not separate target-taking and no-target skills")

    validation_root = root / "results" / "validation_evidence"
    validation_path = validation_root / "oracle_ablation_validation_20260715.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation_expected = {
        "status": "passed",
        "case_count": 2_133,
        "task_count": 34,
        "target_slots_checked": 3_100,
        "mutation_checks": 6_357,
        "operation_sequence_read": False,
        "expert_sequence_called": False,
        "semantic_repair_applied": False,
        "errors": [],
    }
    for field, expected in validation_expected.items():
        if validation.get(field) != expected:
            errors.append(
                f"oracle validation field {field!r}: expected {expected!r}, found {validation.get(field)!r}"
            )

    full_b_no_repair = json.loads(
        (validation_root / "full_b_no_repair_validation_20260721.json").read_text(encoding="utf-8")
    )
    full_b_no_repair_expected = {
        "status": "passed",
        "case_count": 2_133,
        "task_count": 34,
        "total_mutation_checks": 10_623,
        "validation_error_count": 0,
        "semantic_repair_applied": False,
        "errors": [],
    }
    for field, expected in full_b_no_repair_expected.items():
        if full_b_no_repair.get(field) != expected:
            errors.append(
                f"Full-B no-repair field {field!r}: expected {expected!r}, found {full_b_no_repair.get(field)!r}"
            )
    full_b_mutation_table = read_csv(
        root / "results" / "reported_statistics" / "full_b_no_repair_mutation_controls.csv"
    )
    if sum(int(row["checks"]) for row in full_b_mutation_table) != 10_623:
        errors.append("Full-B no-repair table does not sum to 10,623 checks")

    protocol_path = validation_root / "VLABench_phase_target_oracle_ablation_protocol_20260715.md"
    protocol_digest = canonical_text_sha256(protocol_path)
    if protocol_digest != "147bc2c55c3912a4282fe518f0408246c91c2a77c414611244c725ae232e5a30":
        errors.append("sanitized Oracle protocol digest does not match the frozen public copy")
    protocol_text = protocol_path.read_text(encoding="utf-8")
    if "B5596BAD1C4B1C488382ED100BF8C3BF26E7C5FD026ADB3EB2CD16B1C79E364F" not in protocol_text:
        errors.append("sanitized Oracle protocol does not record the original source digest")

    required_figures = {
        "fig_dual_interface_results.png",
        "fig_oracle_responsibility_controls.png",
        "fig_reference_procedure_expansion.png",
        "fig_procedural_failure_taxonomy.png",
    }
    figure_dir = root / "analysis" / "figures"
    for filename in required_figures:
        if not (figure_dir / filename).is_file():
            errors.append(f"missing final figure: {filename}")
    for filename in (
        "fig_responsibility_exposure_results.png",
        "fig_responsibility_exposure_results.svg",
        "fig_procedural_expansion.png",
        "fig_procedural_expansion.svg",
    ):
        if (figure_dir / filename).exists():
            errors.append(f"superseded figure remains in release: {filename}")
    validation_public_sha = canonical_text_sha256(validation_path)
    if validation_public_sha != "ce1ef08e83b913ef38ef75c380e3b8c4d90eeb623b73eb3eee88b088249039b2":
        errors.append("public Oracle validation JSON digest does not match the recorded sanitized copy")

    native_smoke = json.loads(
        (validation_root / "oracle_native_path_smoke_20260715" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    for condition in ("phase_oracle", "target_oracle"):
        row = native_smoke.get(condition, {})
        for field in (
            "actual_gate1_ok",
            "native_execution_attempted",
            "actual_native_ok",
            "actual_condition_success",
            "procedural_success",
        ):
            if row.get(field) is not True:
                errors.append(f"native smoke {condition}/{field} is not true")
        if row.get("actual_failure_code") != "C0":
            errors.append(f"native smoke {condition} did not finish with C0")
    if native_smoke["phase_oracle"].get("phase_source") != "benchmark_derived_task_registry":
        errors.append("Phase-oracle native smoke has an unexpected phase source")
    if native_smoke["phase_oracle"].get("target_source") != "vlm_numbered_mask":
        errors.append("Phase-oracle native smoke has an unexpected target source")
    if native_smoke["phase_oracle"].get("ground_truth_target_used") is not False:
        errors.append("Phase-oracle native smoke unexpectedly used a ground-truth target")
    if native_smoke["target_oracle"].get("phase_source") != "vlm_output":
        errors.append("Target-oracle native smoke has an unexpected phase source")
    if native_smoke["target_oracle"].get("target_source") != "oracle_conditions_components":
        errors.append("Target-oracle native smoke has an unexpected target source")
    if native_smoke["target_oracle"].get("ground_truth_target_used") is not True:
        errors.append("Target-oracle native smoke did not use its controlled reference target")

    registry_audit_path = (
        root
        / "reproducibility_core"
        / "registry_audit"
        / "final"
        / "registry_audit_summary.json"
    )
    registry_audit = json.loads(registry_audit_path.read_text(encoding="utf-8"))
    registry_expected = {
        "audit_status": "PASS",
        "source_commit": "cf588fe60c0c7282174fe979f5913170cfe69017",
        "retained_task_count": 34,
        "source_traceable_task_count": 34,
        "structural_pass_task_count": 34,
        "semantic_review_pass_task_count": 34,
        "d2_equivalent_abstraction_task_count": 0,
        "d3_substantive_mismatch_task_count": 0,
        "unresolved_task_count": 0,
        "unresolved_case_to_branch_assignment_count": 0,
        "procedure_uniqueness_assessed": False,
        "alternative_validity_assessed": False,
        "independent_human_review": False,
        "registry_modified": False,
        "result_labels_modified": 0,
    }
    for field, expected in registry_expected.items():
        if registry_audit.get(field) != expected:
            errors.append(
                f"registry-audit field {field!r}: expected {expected!r}, "
                f"found {registry_audit.get(field)!r}"
            )
    registry_crosswalk = read_csv(
        root
        / "reproducibility_core"
        / "registry_audit"
        / "final"
        / "source_registry_crosswalk.csv"
    )
    if len(registry_crosswalk) != 34:
        errors.append(f"source-registry crosswalk contains {len(registry_crosswalk)}, not 34 rows")
    if any(row["discrepancy_class"] == "D3" for row in registry_crosswalk):
        errors.append("source-registry crosswalk contains a D3 substantive mismatch")
    if any(row["semantic_review"] != "PASS_SOURCE_LINKED_REVIEW" for row in registry_crosswalk):
        errors.append("source-registry crosswalk contains an unreviewed task")

    composition_path = (
        root
        / "reproducibility_core"
        / "robustness"
        / "composition_sensitivity_summary.json"
    )
    composition = json.loads(composition_path.read_text(encoding="utf-8"))
    baseline = composition.get("baseline", {})
    composition_expected = {
        "status": "passed",
        "model_count": 3,
        "cases_per_model": 2_133,
        "pooled_model_case_rows": 6_399,
        "task_clusters": 34,
        "categories": 5,
        "labels_modified": 0,
    }
    for field, expected in composition_expected.items():
        if composition.get(field) != expected:
            errors.append(
                f"composition-sensitivity field {field!r}: expected {expected!r}, "
                f"found {composition.get(field)!r}"
            )
    baseline_expected = {
        "both": 74,
        "a_only": 515,
        "b_only": 103,
        "neither": 5_707,
        "union_success": 692,
        "single_interface_success_count": 618,
        "full_b_failure_count": 6_222,
        "pre_native_responsibility_count": 5_968,
        "success_sets_nonnested": True,
    }
    for field, expected in baseline_expected.items():
        if baseline.get(field) != expected:
            errors.append(
                f"composition baseline field {field!r}: expected {expected!r}, "
                f"found {baseline.get(field)!r}"
            )
    if not composition.get("loto", {}).get("all_success_sets_nonnested"):
        errors.append("a leave-one-task-out omission produced nested A/B success sets")
    if not composition.get("loto", {}).get("all_single_interface_majority"):
        errors.append("a leave-one-task-out omission removed the single-interface majority")
    if not composition.get("loco", {}).get("all_success_sets_nonnested"):
        errors.append("a leave-one-category-out omission produced nested A/B success sets")
    if not composition.get("loco", {}).get("all_single_interface_majority"):
        errors.append("a leave-one-category-out omission removed the single-interface majority")

    cross_scoring_root = root / "reproducibility_core" / "cross_scoring"
    cross_scoring = json.loads(
        (cross_scoring_root / "cross_scoring_audit.json").read_text(encoding="utf-8")
    )
    cross_scoring_expected = {
        "protocol_version": "frozen_full_to_symbolic_v1",
        "model_inference_calls": 0,
        "native_rollouts": 0,
        "input_rows": 6_399,
        "no_repair_controls_passed": 5,
        "no_repair_controls_total": 5,
    }
    for field, expected in cross_scoring_expected.items():
        if cross_scoring.get(field) != expected:
            errors.append(
                f"cross-scoring field {field!r}: expected {expected!r}, "
                f"found {cross_scoring.get(field)!r}"
            )
    if any(
        value != 0
        for value in cross_scoring.get("symbolic_endpoint_reproduction_mismatches", {}).values()
    ):
        errors.append("cross-scoring did not exactly reproduce archived symbolic endpoint labels")
    if cross_scoring.get("reverse_symbolic_to_native", {}).get("performed") is not False:
        errors.append("cross-scoring unexpectedly claims a reverse symbolic-to-native score")

    cross_summary = read_csv(cross_scoring_root / "full_to_symbolic_summary.csv")
    if len(cross_summary) != 4:
        errors.append(f"cross-scoring summary contains {len(cross_summary)}, not 4 rows")
        pooled_cross = {}
    else:
        pooled_cross = next((row for row in cross_summary if row["model"] == "Pooled"), {})
        pooled_cross_expected = {
            "n": "6399",
            "full_output_projection_possible_n": "4588",
            "projected_full_symbolic_success_n": "450",
            "full_procedure_success_n": "177",
            "projected_success_full_failure_n": "273",
            "full_success_projected_failure_n": "0",
            "original_vs_projected_both_n": "203",
            "original_only_n": "386",
            "projected_only_n": "247",
            "original_vs_projected_jaccard_pct": "24.28",
        }
        for field, expected in pooled_cross_expected.items():
            if pooled_cross.get(field) != expected:
                errors.append(
                    f"cross-scoring pooled field {field!r}: expected {expected!r}, "
                    f"found {pooled_cross.get(field)!r}"
                )
    cross_rows = read_csv(cross_scoring_root / "full_to_symbolic_projection_rows.csv")
    if len(cross_rows) != 6_399:
        errors.append(f"cross-scoring row table contains {len(cross_rows)}, not 6,399 rows")
    no_repair_controls = json.loads(
        (cross_scoring_root / "projection_no_repair_controls.json").read_text(encoding="utf-8")
    )
    if len(no_repair_controls) != 5 or not all(row.get("passed") is True for row in no_repair_controls):
        errors.append("cross-scoring no-repair controls are incomplete or contain a failure")

    macro_spec = json.loads(
        (cross_scoring_root / "macro_family_exclusion_spec.json").read_text(encoding="utf-8")
    )
    macro_spec_expected = {
        "analysis_version": "macro_family_exclusion_v1",
        "excluded_rows": 339,
        "retained_rows": 6060,
        "model_inference_calls": 0,
        "native_rollouts": 0,
        "success_labels_modified": 0,
        "projection_rules_modified": False,
    }
    for field, expected in macro_spec_expected.items():
        if macro_spec.get(field) != expected:
            errors.append(
                f"macro-family exclusion field {field!r}: expected {expected!r}, "
                f"found {macro_spec.get(field)!r}"
            )
    macro_usage = read_csv(cross_scoring_root / "macro_mapping_usage_counts.csv")
    if sum(int(row["macro_mapped_rows"]) for row in macro_usage) != 220:
        errors.append("macro-mapping usage table does not contain 220 mapped rows")
    if sum(int(row["projected_symbolic_successes"]) for row in macro_usage) != 62:
        errors.append("macro-mapping usage table does not contain 62 projected successes")
    macro_sensitivity = read_csv(
        cross_scoring_root / "macro_family_exclusion_sensitivity.csv"
    )
    if len(macro_sensitivity) != 8:
        errors.append(
            f"macro-family sensitivity table contains {len(macro_sensitivity)}, not 8 rows"
        )
        macro_excluded_pooled = {}
    else:
        macro_excluded_pooled = next(
            (
                row
                for row in macro_sensitivity
                if row["analysis_set"] == "macro_family_excluded"
                and row["model"] == "Pooled"
            ),
            {},
        )
        macro_excluded_expected = {
            "n": "6060",
            "projection_possible_n": "4368",
            "projection_coverage_pct": "72.08",
            "projected_symbolic_success_n": "388",
            "projected_symbolic_success_overall_pct": "6.40",
            "projected_symbolic_success_among_projectable_pct": "8.88",
            "projected_success_full_failure_n": "223",
            "original_projected_jaccard_pct": "21.52",
        }
        for field, expected in macro_excluded_expected.items():
            if macro_excluded_pooled.get(field) != expected:
                errors.append(
                    f"macro-family excluded pooled field {field!r}: expected {expected!r}, "
                    f"found {macro_excluded_pooled.get(field)!r}"
                )
    macro_summary = json.loads(
        (cross_scoring_root / "macro_family_exclusion_summary.json").read_text(encoding="utf-8")
    )
    if macro_summary.get("qualitative_findings_preserved") is not True:
        errors.append("macro-family exclusion did not preserve the qualitative findings")

    projectable_jaccard = read_csv(
        cross_scoring_root / "projectable_subset_jaccard.csv"
    )
    if len(projectable_jaccard) != 8:
        errors.append(
            f"projectable-subset Jaccard table contains {len(projectable_jaccard)}, not 8 rows"
        )
        projectable_pooled = {}
        projectable_macro_excluded_pooled = {}
    else:
        projectable_pooled = next(
            (
                row
                for row in projectable_jaccard
                if row["analysis_set"] == "full_retained_manifest"
                and row["model"] == "Pooled"
            ),
            {},
        )
        projectable_macro_excluded_pooled = next(
            (
                row
                for row in projectable_jaccard
                if row["analysis_set"] == "macro_family_excluded"
                and row["model"] == "Pooled"
            ),
            {},
        )
        projectable_expected = {
            "all_rows_n": "6399",
            "projectable_rows_n": "4588",
            "original_success_projectable_n": "474",
            "projected_success_n": "450",
            "both_success_n": "203",
            "original_only_n": "271",
            "projected_only_n": "247",
            "success_union_n": "721",
            "projectable_subset_jaccard_pct": "28.16",
            "original_success_unprojectable_n": "115",
        }
        for field, expected in projectable_expected.items():
            if projectable_pooled.get(field) != expected:
                errors.append(
                    f"projectable pooled field {field!r}: expected {expected!r}, "
                    f"found {projectable_pooled.get(field)!r}"
                )
        projectable_macro_expected = {
            "all_rows_n": "6060",
            "projectable_rows_n": "4368",
            "original_success_projectable_n": "423",
            "projected_success_n": "388",
            "both_success_n": "161",
            "original_only_n": "262",
            "projected_only_n": "227",
            "success_union_n": "650",
            "projectable_subset_jaccard_pct": "24.77",
            "original_success_unprojectable_n": "98",
        }
        for field, expected in projectable_macro_expected.items():
            if projectable_macro_excluded_pooled.get(field) != expected:
                errors.append(
                    f"projectable macro-excluded field {field!r}: expected {expected!r}, "
                    f"found {projectable_macro_excluded_pooled.get(field)!r}"
                )
    projectable_summary = json.loads(
        (cross_scoring_root / "projectable_subset_jaccard_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if projectable_summary.get("qualitative_findings_preserved") is not True:
        errors.append("projectable-subset Jaccard did not preserve the qualitative findings")
    if any(
        projectable_summary.get(field) != expected
        for field, expected in {
            "model_inference_calls": 0,
            "native_rollouts": 0,
            "success_labels_modified": 0,
        }.items()
    ):
        errors.append("projectable-subset Jaccard reports an unexpected data-changing operation")

    evidence_layers = read_csv(
        root
        / "reproducibility_core"
        / "result_tables"
        / "corrected_gate_evidence_layers.csv"
    )
    if len(evidence_layers) != 8:
        errors.append(f"corrected-gate evidence-layer table contains {len(evidence_layers)}, not 8 rows")
    evidence_counts = {(row["evidence_layer"], row["scope"]): int(row["count"]) for row in evidence_layers}
    evidence_expected = {
        ("observed_runtime", "Full B raw C5 AttributeError"): 5,
        ("observed_runtime", "Reference-phase scaffold raw C5 AttributeError"): 467,
        ("responsibility_adjudication", "All gate-related C5 rows"): 472,
        ("offline_corrected_gate", "All frozen model-condition-case outputs"): 19_197,
        ("offline_corrected_gate", "All adjudicated gate-related C5 rows recovered"): 472,
        ("offline_corrected_gate", "Existing C3 target/context rows additionally flagged"): 602,
        ("offline_corrected_gate", "All flagged outputs"): 1_074,
        ("offline_corrected_gate", "Successful outputs flagged"): 0,
    }
    if evidence_counts != evidence_expected:
        errors.append("corrected-gate evidence-layer counts do not match the frozen audit")
    if any(int(row["success_label_change"]) != 0 for row in evidence_layers):
        errors.append("corrected-gate evidence-layer table reports a success-label change")

    live_smoke_root = validation_root / "internvl_oracle_live_inference_smoke_20260715"
    try:
        live_phase = read_single_jsonl(
            live_smoke_root / "phase_oracle" / "procedural_vlm_execution_rows.jsonl"
        )
        live_target = read_single_jsonl(
            live_smoke_root / "target_oracle" / "procedural_vlm_execution_rows.jsonl"
        )
    except ValueError as exc:
        errors.append(str(exc))
        live_phase = {}
        live_target = {}
    if not (
        live_phase.get("ablation_mode") == "phase_oracle"
        and live_phase.get("model_parse_ok") is True
        and live_phase.get("strict_parse_ok") is True
        and live_phase.get("semantic_repair_applied") is False
        and live_phase.get("inference_only") is True
    ):
        errors.append("InternVL Phase-oracle live-parser smoke does not match the recorded contract")
    if not (
        live_target.get("ablation_mode") == "target_oracle"
        and live_target.get("model_parse_ok") is True
        and live_target.get("strict_parse_ok") is True
        and live_target.get("oracle_target_read") is False
        and live_target.get("actual_failure_code") == "C3"
        and live_target.get("actual_failure_subtype") == "missing_phase"
        and live_target.get("native_execution_attempted") is False
    ):
        errors.append("InternVL Target-oracle access-order smoke does not match the recorded contract")

    source_checksum_lines = [
        line
        for line in (validation_root / "SOURCE_SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(source_checksum_lines) != 40:
        errors.append(f"recovered-source checksum manifest contains {len(source_checksum_lines)}, not 40 rows")
    expected_source_validation = (
        "6a8a95247c8118851d20bb230ab2143b247f15b6572453d66e21105f6e8c49a4  "
        "oracle_ablation_validation_20260715.json"
    )
    if expected_source_validation not in source_checksum_lines:
        errors.append("recovered-source checksum manifest lacks the original validation JSON digest")

    agreement_root = root / "reproducibility_core" / "agreement_sensitivity"
    agreement_audit = json.loads(
        (agreement_root / "agreement_sensitivity_audit.json").read_text(encoding="utf-8")
    )
    agreement_expected = {
        "analysis_version": "agreement_and_control_subset_v2",
        "model_inference_calls": 0,
        "native_rollouts": 0,
        "success_labels_modified": 0,
        "positive_control_first_attempt_case_count": 2_062,
    }
    for field, expected in agreement_expected.items():
        if agreement_audit.get(field) != expected:
            errors.append(
                f"agreement sensitivity field {field!r}: expected {expected!r}, "
                f"found {agreement_audit.get(field)!r}"
            )
    original_counts_expected = {
        "both": 74,
        "symbolic_only": 515,
        "procedure_only": 103,
        "neither": 5_707,
    }
    if agreement_audit.get("pooled_original_counts") != original_counts_expected:
        errors.append("agreement sensitivity original pooled counts do not match frozen labels")
    original_metric_expected = {
        "raw_agreement": 0.9034224097515237,
        "positive_agreement": 0.19321148825065274,
        "negative_agreement": 0.9486369680851063,
        "cohen_kappa": 0.15736753823888902,
        "mcc_phi": 0.19021738392987753,
        "odds_ratio": 7.961504383070977,
        "success_set_jaccard": 0.1069364161849711,
    }
    for field, expected in original_metric_expected.items():
        actual = agreement_audit.get("pooled_original_metrics", {}).get(field)
        if actual is None or abs(float(actual) - expected) > 1e-12:
            errors.append(
                f"agreement sensitivity pooled original metric {field!r}: "
                f"expected {expected!r}, found {actual!r}"
            )
    if not all(
        agreement_audit.get("unprojectable_equals_parser_rejected_set_by_model", {}).values()
    ):
        errors.append("agreement sensitivity does not confirm parser/unprojectable set identity")

    agreement_metrics = read_csv(agreement_root / "agreement_metrics.csv")
    if len(agreement_metrics) != 36:
        errors.append(f"agreement sensitivity table contains {len(agreement_metrics)}, not 36 rows")
    control_subset = read_csv(agreement_root / "control_first_attempt_subset.csv")
    if len(control_subset) != 33:
        errors.append(f"control first-attempt subset table contains {len(control_subset)}, not 33 rows")
    projectability_reasons = read_csv(agreement_root / "projectability_failure_reasons.csv")
    if len(projectability_reasons) != 3:
        errors.append(
            f"projectability reason table contains {len(projectability_reasons)}, not 3 rows"
        )
    if sum(int(row["unprojectable"]) for row in projectability_reasons) != 1_811:
        errors.append("projectability reason table does not total 1,811 unprojectable outputs")
    if any(
        int(row["unprojectable"]) != int(row["parser_rejected"])
        or int(row["parser_accepted_unsupported_action"]) != 0
        or int(row["parser_accepted_missing_or_ambiguous_macro_target"]) != 0
        or int(row["other_unprojectable"]) != 0
        for row in projectability_reasons
    ):
        errors.append("projectability reason decomposition differs from parser-rejection audit")

    # ---- artifact v1.7: symbolic endpoint tiers ----------------------------------
    tiers_audit = cross_scoring.get("symbolic_endpoint_tiers", {})
    tiers_expected = {
        "pm100": (589, 450, 273, 177, "24.28", "10.69"),
        "strict": (564, 260, 83, 177, "18.56", "11.09"),
        "strict_executor_filtered": (564, 405, 228, 177, "27.33", "11.09"),
    }
    tier_fields = (
        "original_symbolic_success_n",
        "projected_symbolic_success_n",
        "projected_success_full_failure_n",
        "full_success_with_projected_success_n",
        "original_vs_projected_jaccard_pct",
        "original_vs_full_jaccard_pct",
    )
    for tier, expected_values in tiers_expected.items():
        actual = tiers_audit.get("pooled", {}).get(tier, {})
        for field, expected in zip(tier_fields, expected_values):
            if actual.get(field) != expected:
                errors.append(
                    f"endpoint tier {tier}/{field}: expected {expected!r}, found {actual.get(field)!r}"
                )
    if tiers_audit.get("executor_only_phase_vocabulary") != ["moveto", "wait", "open_gripper", "move_offset"]:
        errors.append("endpoint tiers: executor-only phase vocabulary differs from the frozen list")
    strict_reasons = tiers_audit.get("projected_pm100_pass_strict_fail_reasons", {})
    if (
        sum(strict_reasons.values()) != 190
        or strict_reasons.get("executor_only_phase_residue:moveto") != 135
        or strict_reasons.get("surplus_evaluator_vocabulary_action") != 54
    ):
        errors.append("endpoint tiers: PM=100-pass/strict-fail reason decomposition differs from the frozen audit")
    if tiers_audit.get("original_pm100_pass_strict_fail_reasons", {}).get("surplus_evaluator_vocabulary_action") != 25:
        errors.append("endpoint tiers: original PM=100-pass/strict-fail count is not 25")
    tier_table = read_csv(cross_scoring_root / "symbolic_endpoint_tiers.csv")
    if len(tier_table) != 12:
        errors.append(f"symbolic endpoint tier table contains {len(tier_table)}, not 12 rows")
    required_tier_columns = {
        "original_pm100_success",
        "original_strict_sequence_success",
        "projected_pm100_success",
        "projected_strict_sequence_success",
        "projected_strict_executor_filtered_success",
        "projected_pm100_pass_strict_fail_reason",
    }
    if cross_rows and not required_tier_columns <= set(cross_rows[0]):
        errors.append("cross-scoring row table lacks the v1.7 endpoint tier columns")
    else:
        strict_projected = sum(row["projected_strict_sequence_success"].lower() == "true" for row in cross_rows)
        filtered_projected = sum(
            row["projected_strict_executor_filtered_success"].lower() == "true" for row in cross_rows
        )
        pm100_projected = sum(row["projected_pm100_success"].lower() == "true" for row in cross_rows)
        strict_not_pm100 = sum(
            row["projected_strict_sequence_success"].lower() == "true"
            and row["projected_pm100_success"].lower() != "true"
            for row in cross_rows
        )
        filtered_not_pm100 = sum(
            row["projected_strict_executor_filtered_success"].lower() == "true"
            and row["projected_pm100_success"].lower() != "true"
            for row in cross_rows
        )
        if (pm100_projected, strict_projected, filtered_projected) != (450, 260, 405):
            errors.append("cross-scoring rows do not reproduce the 450/260/405 projected tier counts")
        if strict_not_pm100 != 0:
            errors.append("strict projected successes are not nested in the PM=100 successes")
        if filtered_not_pm100 != 123:
            errors.append(
                f"executor-filtered strict successes outside PM=100: expected 123, found {filtered_not_pm100}"
            )
    tier_sensitivity = read_csv(cross_scoring_root / "endpoint_tier_sensitivities.csv")
    if len(tier_sensitivity) != 24:
        errors.append(f"endpoint tier sensitivity table contains {len(tier_sensitivity)}, not 24 rows")
    tier_sensitivity_summary = json.loads(
        (cross_scoring_root / "endpoint_tier_sensitivities_summary.json").read_text(encoding="utf-8")
    )
    tier_sensitivity_pooled = tier_sensitivity_summary.get("pooled", {})
    tier_sensitivity_expected = {
        ("full_retained_manifest/strict", "projected_symbolic_success_n"): 260,
        ("full_retained_manifest/strict", "projectable_subset_jaccard_pct"): "22.13",
        ("full_retained_manifest/strict_executor_filtered", "projectable_subset_jaccard_pct"): "32.05",
        ("macro_family_excluded/strict", "projected_success_full_failure_n"): 76,
        ("macro_family_excluded/strict_executor_filtered", "projected_success_full_failure_n"): 221,
    }
    for (key, field), expected in tier_sensitivity_expected.items():
        if tier_sensitivity_pooled.get(key, {}).get(field) != expected:
            errors.append(f"endpoint tier sensitivity {key}/{field}: expected {expected!r}")
    if tier_sensitivity_summary.get("qualitative_findings_preserved") is not True:
        errors.append("endpoint tier sensitivities did not preserve the qualitative findings")
    agreement_tiers = agreement_audit.get("endpoint_tiers", {})
    agreement_tier_expected = {
        ("strict", "independently_elicited_symbolic_vs_full_procedure"): (74, 490, 103, 5_732),
        ("strict", "original_symbolic_vs_projected_full_all_rows"): (129, 435, 131, 5_704),
        ("strict_executor_filtered", "original_symbolic_vs_projected_full_all_rows"): (208, 356, 197, 5_638),
    }
    for (tier, analysis), expected_counts in agreement_tier_expected.items():
        counts = agreement_tiers.get(tier, {}).get(analysis, {}).get("counts", {})
        actual_counts = tuple(counts.get(key) for key in ("both", "left_only", "right_only", "neither"))
        if actual_counts != expected_counts:
            errors.append(f"agreement endpoint tier {tier}/{analysis}: counts {actual_counts} differ from frozen values")
    for tier_dir, expected_counts in (
        ("robustness_strict", {"both": 74, "a_only": 490, "b_only": 103, "neither": 5_732}),
        ("robustness_strict_executor_filtered", {"both": 74, "a_only": 490, "b_only": 103, "neither": 5_732}),
    ):
        summary_path = root / "reproducibility_core" / tier_dir / "composition_sensitivity_summary.json"
        if not summary_path.exists():
            errors.append(f"{tier_dir}: composition-sensitivity summary is missing")
            continue
        tier_composition = json.loads(summary_path.read_text(encoding="utf-8"))
        if tier_composition.get("pooled_ab_counts") != expected_counts:
            errors.append(f"{tier_dir}: pooled A/B counts differ from the frozen strict labels")
        for exclusion in ("loto", "loco"):
            block = tier_composition.get(exclusion, {})
            if not (block.get("all_success_sets_nonnested") and block.get("all_single_interface_majority")):
                errors.append(f"{tier_dir}: a {exclusion} omission broke the non-nested/majority invariants")
    tier_rates = read_csv(root / "results" / "reported_statistics" / "symbolic_endpoint_tier_rates.csv")
    tier_rate_expected = {"Qwen3-VL-8B": (280, 258, 80), "Gemma-3-12B-it": (221, 219, 57), "InternVL3-8B": (88, 87, 40)}
    if len(tier_rates) != 3:
        errors.append("symbolic endpoint tier-rate table does not contain three model rows")
    for row in tier_rates:
        actual = (int(row["pm100_count"]), int(row["strict_count"]), int(row["procedure_count"]))
        if tier_rate_expected.get(row["model"]) != actual:
            errors.append(f"{row['model']}: endpoint tier-rate counts {actual} differ from frozen values")
    for relative in (
        "tests/test_symbolic_endpoints.py",
        "analysis/analyze_endpoint_tier_sensitivities.py",
        "analysis/generate_endpoint_tier_assets.py",
        "figures/fig_frozen_output_tiers.png",
        "figures/fig_ab_agreement_tiers.png",
        "figures/fig_interface_success_rates_tiers.png",
    ):
        if not (root / relative).exists():
            errors.append(f"artifact v1.7 file missing: {relative}")

    findings: list[dict[str, object]] = []
    for path in root.rglob("*"):
        if ".git" in path.parts or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if path.relative_to(root).as_posix() in {
            "tests/verify_release.py",
            "PACKAGE_VERIFICATION.json",
            "SHA256SUMS.txt",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        for name, pattern in FORBIDDEN_PATTERNS.items():
            for match in pattern.finditer(text):
                if name == "external_ipv4" and is_pinned_dependency_version(text, match):
                    continue
                findings.append({"type": name, "path": path.relative_to(root).as_posix(), "match": match.group(0)[:120]})
    if findings:
        errors.append(f"anonymity scan found {len(findings)} forbidden strings")

    checksum_rows = []
    for path in sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
        and p.name not in {"SHA256SUMS.txt", "PACKAGE_VERIFICATION.json"}
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_rows.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")

    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "anonymity_findings": findings,
        "manifest_cases": len(manifest),
        "model_result_pairs": 3,
        "oracle_model_condition_runs": 6,
        "oracle_adjudication_rows": adjudication_rows,
        "oracle_validation_cases": validation.get("case_count"),
        "oracle_validation_mutation_checks": validation.get("mutation_checks"),
        "full_b_no_repair_mutation_checks": full_b_no_repair.get("total_mutation_checks"),
        "oracle_simultaneous_zero_excluding_contrasts": sum(
            row["simultaneous_excludes_zero"].lower() == "true" for row in simultaneous
        ),
        "corrected_gate_mapping_rows": len(corrected_rows),
        "corrected_gate_adjudicated_c5_recovered": corrected_gate.get(
            "expected_adjudicated_c5_cases"
        ),
        "registry_audit_tasks": registry_audit.get("retained_task_count"),
        "registry_audit_substantive_mismatches": registry_audit.get(
            "d3_substantive_mismatch_task_count"
        ),
        "registry_audit_unresolved_tasks": registry_audit.get("unresolved_task_count"),
        "composition_loto_omissions": composition.get("loto", {}).get("pooled_exclusions"),
        "composition_loco_omissions": composition.get("loco", {}).get("pooled_exclusions"),
        "composition_labels_modified": composition.get("labels_modified"),
        "cross_scoring_input_rows": cross_scoring.get("input_rows"),
        "cross_scoring_projectable_rows": pooled_cross.get("full_output_projection_possible_n"),
        "cross_scoring_projected_symbolic_successes": pooled_cross.get(
            "projected_full_symbolic_success_n"
        ),
        "cross_scoring_projected_success_full_failure": pooled_cross.get(
            "projected_success_full_failure_n"
        ),
        "cross_scoring_no_repair_controls_passed": cross_scoring.get(
            "no_repair_controls_passed"
        ),
        "cross_scoring_model_inference_calls": cross_scoring.get("model_inference_calls"),
        "cross_scoring_native_rollouts": cross_scoring.get("native_rollouts"),
        "macro_family_exclusion_rows": macro_spec.get("excluded_rows"),
        "macro_family_exclusion_retained_rows": macro_spec.get("retained_rows"),
        "macro_family_exclusion_projected_successes": macro_excluded_pooled.get(
            "projected_symbolic_success_n"
        ),
        "macro_family_exclusion_projected_success_full_failure": macro_excluded_pooled.get(
            "projected_success_full_failure_n"
        ),
        "macro_family_exclusion_jaccard_pct": macro_excluded_pooled.get(
            "original_projected_jaccard_pct"
        ),
        "projectable_subset_jaccard_pct": projectable_pooled.get(
            "projectable_subset_jaccard_pct"
        ),
        "projectable_subset_original_only": projectable_pooled.get("original_only_n"),
        "projectable_subset_projected_only": projectable_pooled.get("projected_only_n"),
        "projectable_macro_excluded_jaccard_pct": projectable_macro_excluded_pooled.get(
            "projectable_subset_jaccard_pct"
        ),
        "agreement_sensitivity_rows": len(agreement_metrics),
        "agreement_original_raw_agreement": agreement_audit.get(
            "pooled_original_metrics", {}
        ).get("raw_agreement"),
        "agreement_original_kappa": agreement_audit.get("pooled_original_metrics", {}).get(
            "cohen_kappa"
        ),
        "agreement_original_mcc_phi": agreement_audit.get(
            "pooled_original_metrics", {}
        ).get("mcc_phi"),
        "agreement_original_odds_ratio": agreement_audit.get(
            "pooled_original_metrics", {}
        ).get("odds_ratio"),
        "positive_control_first_attempt_subset_cases": agreement_audit.get(
            "positive_control_first_attempt_case_count"
        ),
        "projectability_parser_rejected_rows": sum(
            int(row["parser_rejected"]) for row in projectability_reasons
        ),
        "recovered_validation_source_files": len(source_checksum_lines),
        "checksummed_files": len(checksum_rows),
        "symbolic_endpoint_tiers_pooled": {
            tier: {
                "original_symbolic_success_n": tiers_audit.get("pooled", {}).get(tier, {}).get("original_symbolic_success_n"),
                "projected_symbolic_success_n": tiers_audit.get("pooled", {}).get(tier, {}).get("projected_symbolic_success_n"),
                "projected_success_full_failure_n": tiers_audit.get("pooled", {}).get(tier, {}).get("projected_success_full_failure_n"),
            }
            for tier in tiers_expected
        },
        "executor_filtered_strict_successes_outside_pm100": 123 if not errors else None,
    }
    (root / "PACKAGE_VERIFICATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
