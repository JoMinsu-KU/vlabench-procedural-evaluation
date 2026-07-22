from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODEL_RUNS = {
    "Qwen3-VL-8B": {
        "full": "results/model_outputs/qwen/experiment_b/procedural_vlm_execution_rows.jsonl",
        "phase_oracle": "results/model_outputs/qwen/oracle_ablations/phase_oracle/procedural_vlm_execution_rows.jsonl",
        "target_oracle": "results/model_outputs/qwen/oracle_ablations/target_oracle/procedural_vlm_execution_rows.jsonl",
    },
    "Gemma-3-12B-it": {
        "full": "results/model_outputs/gemma/experiment_b/procedural_vlm_execution_rows.jsonl",
        "phase_oracle": "results/model_outputs/gemma/oracle_ablations/phase_oracle/procedural_vlm_execution_rows.jsonl",
        "target_oracle": "results/model_outputs/gemma/oracle_ablations/target_oracle/procedural_vlm_execution_rows.jsonl",
    },
    "InternVL3-8B": {
        "full": "results/model_outputs/internvl/experiment_b/procedural_vlm_execution_rows.jsonl",
        "phase_oracle": "results/model_outputs/internvl/oracle_ablations/phase_oracle/procedural_vlm_execution_rows.jsonl",
        "target_oracle": "results/model_outputs/internvl/oracle_ablations/target_oracle/procedural_vlm_execution_rows.jsonl",
    },
}

EXPECTED_ATTRIBUTE_ERRORS = {
    "'BilliardBall' object has no attribute 'get_place_point'",
    "'Fruit' object has no attribute 'get_place_point'",
    "'Toy' object has no attribute 'get_place_point'",
    "'BilliardTable' object has no attribute 'get_grasped_keypoints'",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repository_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument(
        "--vlabench-public-root",
        type=Path,
        required=True,
        help="Path to the external VLABench public-view case tree containing env_config.json files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repository_root / "results" / "outcome_audits" / "corrected_affordance_revalidation_20260721.json",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=repository_root / "results" / "outcome_audits" / "corrected_affordance_revalidation_20260721.csv",
    )
    return parser.parse_args()


def load_unique_rows(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["case_id"])] = row
    return rows


def target_class(env_config: dict, target: int) -> str:
    components = env_config.get("task", {}).get("components", [])
    if not isinstance(components, list) or target < 0 or target >= len(components):
        return ""
    component = components[target]
    return str(component.get("class", "")) if isinstance(component, dict) else ""


def corrected_gate_violation(steps: list[dict], env_config: dict) -> dict | None:
    for index, step in enumerate(steps, start=1):
        if "target" not in step:
            continue
        skill = str(step.get("skill", ""))
        selected_class = target_class(env_config, int(step["target"]))
        if skill == "pick" and selected_class == "BilliardTable":
            return {
                "step": index,
                "skill": skill,
                "target_class": selected_class,
                "missing_native_method": "get_grasped_keypoints",
            }
        if skill == "place" and selected_class in {"BilliardBall", "Fruit", "Toy"}:
            return {
                "step": index,
                "skill": skill,
                "target_class": selected_class,
                "missing_native_method": "get_place_point",
            }
    return None


def main() -> None:
    args = parse_args()
    revalidated: list[dict] = []
    expected_c5: set[tuple[str, str, str]] = set()
    observed_keys: set[tuple[str, str, str]] = set()
    total_rows = 0
    duplicate_source_rows = 0

    for model, conditions in MODEL_RUNS.items():
        for condition, relative_path in conditions.items():
            path = args.repository_root / relative_path
            raw_line_count = sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
            rows = load_unique_rows(path)
            duplicate_source_rows += raw_line_count - len(rows)
            if len(rows) != 2_133:
                raise ValueError(f"{model}/{condition}: expected 2,133 unique cases, found {len(rows)}")
            total_rows += len(rows)
            for case_id, row in rows.items():
                key = (model, condition, case_id)
                observed_keys.add(key)
                detail = str(row.get("actual_failure_detail", ""))
                if row.get("actual_failure_code") == "C5" and detail in EXPECTED_ATTRIBUTE_ERRORS:
                    expected_c5.add(key)

                parsed_plan = row.get("parsed_plan") or {}
                steps = parsed_plan.get("steps") if isinstance(parsed_plan, dict) else None
                if not isinstance(steps, list) or not steps:
                    continue
                category, task_name, example_name = case_id.split("/")[:3]
                env_path = args.vlabench_public_root / category / task_name / example_name / "env_config.json"
                env_config = json.loads(env_path.read_text(encoding="utf-8"))
                violation = corrected_gate_violation(steps, env_config)
                if violation is None:
                    continue
                revalidated.append(
                    {
                        "model": model,
                        "condition": condition,
                        "case_id": case_id,
                        "original_failure_code": row.get("actual_failure_code", ""),
                        "original_failure_subtype": row.get("actual_failure_subtype", ""),
                        "original_failure_detail": detail,
                        "original_success": bool(row.get("procedural_success", False)),
                        "corrected_gate_failure_code": "C3",
                        "corrected_gate_failure_subtype": "incompatible_target_affordance",
                        **violation,
                    }
                )

    flagged_keys = {(row["model"], row["condition"], row["case_id"]) for row in revalidated}
    missing_from_corrected_gate = sorted(expected_c5 - flagged_keys)
    additional_corrected_gate_flags = sorted(flagged_keys - expected_c5)
    flagged_successes = [row for row in revalidated if row["original_success"]]
    additional_rows = [
        row
        for row in revalidated
        if (row["model"], row["condition"], row["case_id"]) not in expected_c5
    ]
    additional_already_c3 = [row for row in additional_rows if row["original_failure_code"] == "C3"]
    additional_other_outcomes = [row for row in additional_rows if row["original_failure_code"] != "C3"]

    report = {
        "status": "passed"
        if len(expected_c5) == 472
        and expected_c5.issubset(flagged_keys)
        and len(additional_already_c3) == len(additional_rows)
        and not additional_other_outcomes
        and not flagged_successes
        and len(observed_keys) == 19_197
        else "failed",
        "model_condition_rows": total_rows,
        "unique_model_condition_case_keys": len(observed_keys),
        "source_duplicate_rows_ignored_by_case_id": duplicate_source_rows,
        "expected_adjudicated_c5_cases": len(expected_c5),
        "corrected_gate_flagged_cases": len(flagged_keys),
        "all_adjudicated_c5_recovered": expected_c5.issubset(flagged_keys),
        "already_c3_additional_flags": len(additional_already_c3),
        "additional_non_c3_outcomes": len(additional_other_outcomes),
        "missing_from_corrected_gate": missing_from_corrected_gate,
        "additional_corrected_gate_flag_count": len(additional_corrected_gate_flags),
        "flagged_original_successes": len(flagged_successes),
        "success_labels_changed": False,
        "rules": {
            "pick": {"BilliardTable": "missing get_grasped_keypoints"},
            "place": {
                "BilliardBall": "missing get_place_point",
                "Fruit": "missing get_place_point",
                "Toy": "missing get_place_point",
            },
        },
        "scope": (
            "Offline revalidation of frozen Full-B, Phase-oracle, and Target-oracle outputs. "
            "No model inference or native rollout was performed, and raw observed failure codes were not overwritten."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(revalidated[0].keys()) if revalidated else []
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(revalidated)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("corrected affordance revalidation did not reproduce the adjudication map")


if __name__ == "__main__":
    main()
