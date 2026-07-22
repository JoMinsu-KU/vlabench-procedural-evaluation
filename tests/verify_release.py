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
        "recovered_validation_source_files": len(source_checksum_lines),
        "checksummed_files": len(checksum_rows),
    }
    (root / "PACKAGE_VERIFICATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
