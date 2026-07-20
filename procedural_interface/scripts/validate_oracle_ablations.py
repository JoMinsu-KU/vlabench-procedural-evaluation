"""Validate both oracle ablations over the real 2,133-case manifest without VLM calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from procedural_interface.adapter.task_phase_templates import TASK_TEMPLATES
from procedural_interface.evaluation.oracle_ablations import (
    PHASE_ORACLE_MODE,
    TARGET_ORACLE_MODE,
    build_reference_steps,
    resolve_phase_oracle_output,
    resolve_target_oracle_output,
)
from procedural_interface.scripts.run_procedural_vlm_eval_api import build_condition_prompt
from vlabench_official_eval.dataset.case_loader import load_manifest, select_case_ids, split_case_id


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse paths for the real-data oracle validation."""

    # Keep defaults aligned with the experiment runner.
    parser = argparse.ArgumentParser(description=__doc__)
    # Use the fixed 2,133-case paired manifest.
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "datasets" / "manifests" / "matched_2133_symbolic_procedural_cases.json")
    # Use the same public-view dataset root as Experiment B.
    parser.add_argument("--public-root", type=Path, required=True)
    # Save an auditable validation report outside model-result directories.
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "validation" / "oracle_ablation_validation.json")
    # Return the parsed values.
    return parser.parse_args()


def forbidden_oracle_loader() -> dict[str, Any]:
    """Fail if an invalid sequence attempts to read target-oracle metadata."""

    # Mutation controls must be rejected before any condition/component access.
    raise AssertionError("oracle target loader called before sequence validation")


def main() -> int:
    """Run coverage, ownership, mutation, and prompt-isolation checks."""

    # Read CLI paths.
    args = parse_args()
    # Load the same manifest parser used by Experiment B.
    manifest = load_manifest(args.manifest)
    # Select every case in deterministic manifest order.
    case_ids = select_case_ids(manifest=manifest, max_cases=0, case_list_file=None)
    # Collect explicit errors instead of stopping at the first task.
    errors: list[dict[str, Any]] = []
    # Track aggregate coverage for the final report.
    task_counts: dict[str, int] = {}
    # Count target slots tested through the phase-oracle parser.
    target_slots = 0
    # Count mutation checks that demonstrate no repair.
    mutation_checks = 0
    # Validate every retained real case.
    for case_id in case_ids:
        # Split the canonical case id.
        category, task_name, example_name, _ = split_case_id(case_id)
        # Build the real env_config path.
        env_config_path = args.public_root / category / task_name / example_name / "env_config.json"
        try:
            # Every retained task must have one frozen procedural registry entry.
            if task_name not in TASK_TEMPLATES:
                raise KeyError(f"task template missing: {task_name}")
            # Load actual case metadata used by the existing positive-control path.
            env_config = json.loads(env_config_path.read_text(encoding="utf-8"))
            # Build the reference procedure and numeric target evidence.
            reference_steps, reference_targets = build_reference_steps(task_name, env_config)
            # Track real-case and task coverage.
            task_counts[task_name] = task_counts.get(task_name, 0) + 1
            # Extract only target values in frozen target-bearing phase order.
            target_values = [step["target"] for step in reference_steps if "target" in step]
            target_slots += len(target_values)
            # Phase oracle must reconstruct the same phase skeleton using supplied targets only.
            phase_result = resolve_phase_oracle_output(json.dumps({"targets": target_values}), task_name)
            if not phase_result.get("ok") or phase_result.get("steps") != reference_steps:
                raise AssertionError("phase-oracle resolution did not preserve reference phase/target alignment")
            # Phase oracle must mark no case-specific oracle target read.
            if phase_result.get("oracle_target_read"):
                raise AssertionError("phase oracle unexpectedly read oracle targets")
            # Target oracle receives only model-equivalent skill/order output.
            model_skills = [step["skill"] for step in reference_steps]
            target_result = resolve_target_oracle_output(json.dumps({"skills": model_skills}), task_name, env_config)
            if not target_result.get("ok") or target_result.get("steps") != reference_steps:
                raise AssertionError("target-oracle resolution changed a valid skill sequence")
            # A valid target-oracle row must record the intentional oracle read.
            if not target_result.get("oracle_target_read"):
                raise AssertionError("target oracle did not record its oracle target read")
            # Remove one target slot; phase oracle must fail rather than infer it.
            missing_target_result = resolve_phase_oracle_output(json.dumps({"targets": target_values[:-1]}), task_name)
            mutation_checks += 1
            if missing_target_result.get("ok"):
                raise AssertionError("phase oracle repaired a missing target slot")
            # Remove one skill; target oracle must reject before reading oracle targets.
            missing_skill_result = resolve_target_oracle_output(json.dumps({"skills": model_skills[:-1]}), task_name, forbidden_oracle_loader)
            mutation_checks += 1
            if missing_skill_result.get("ok") or missing_skill_result.get("oracle_target_read"):
                raise AssertionError("target oracle repaired a missing skill or read targets before sequence validation")
            # Swap the first two skills when possible; no reordering is allowed.
            if len(model_skills) > 1:
                swapped = list(model_skills)
                swapped[0], swapped[1] = swapped[1], swapped[0]
                swapped_result = resolve_target_oracle_output(json.dumps({"skills": swapped}), task_name, forbidden_oracle_loader)
                mutation_checks += 1
                if swapped_result.get("ok") or swapped_result.get("oracle_target_read"):
                    raise AssertionError("target oracle repaired a reordered skill sequence")
            # Verify audit-only names cover every numeric reference target.
            if set(reference_targets) != set(target_values):
                raise AssertionError("reference target audit map does not match execution targets")
        except Exception as exc:  # noqa: BLE001 - report every real-case validation error.
            errors.append({"case_id": case_id, "error_type": type(exc).__name__, "error": str(exc)})
    # Validate prompt ownership once per retained task.
    for task_name in sorted(task_counts):
        # Use the exact Experiment B skill-description header.
        full_prompt = build_condition_prompt(instruction="", task_name=task_name, ablation_mode="full")
        skillbook_header = full_prompt.split("\n---\n", maxsplit=1)[0]
        # Phase prompt must expose the phase skeleton and target-only response contract.
        phase_prompt = build_condition_prompt(instruction="", task_name=task_name, ablation_mode=PHASE_ORACLE_MODE)
        if skillbook_header not in phase_prompt or '"targets"' not in phase_prompt or '"skills"' in phase_prompt.split("### Phase-oracle condition", 1)[-1]:
            errors.append({"task_name": task_name, "error_type": "PromptOwnershipError", "error": "invalid phase-oracle prompt contract"})
        # Target prompt must request skills only and must not serialize case-specific targets.
        target_prompt = build_condition_prompt(instruction="", task_name=task_name, ablation_mode=TARGET_ORACLE_MODE)
        if skillbook_header not in target_prompt or '"skills"' not in target_prompt or "Reference phase skeleton" in target_prompt:
            errors.append({"task_name": task_name, "error_type": "PromptOwnershipError", "error": "invalid target-oracle prompt contract"})
    # Build the machine-readable report.
    report = {
        "status": "passed" if not errors and len(case_ids) == 2133 else "failed",
        "manifest": str(args.manifest),
        "public_root": str(args.public_root),
        "case_count": len(case_ids),
        "task_count": len(task_counts),
        "target_slots_checked": target_slots,
        "mutation_checks": mutation_checks,
        "phase_oracle_target_source": "model_numbered_mask",
        "phase_oracle_phase_source": "benchmark_derived_task_registry",
        "target_oracle_target_source": "conditions_and_components_after_sequence_validation",
        "target_oracle_phase_source": "model_output",
        "operation_sequence_read": False,
        "expert_sequence_called": False,
        "semantic_repair_applied": False,
        "errors": errors,
    }
    # Create the report directory.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Persist the complete validation outcome.
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # Print the same report for terminal review.
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    # Fail CI/terminal execution when any integrity check fails.
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
