"""Procedural-adapter smoke/evaluation runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from procedural_interface.core.io import ensure_dir, write_json
from procedural_interface.evaluation.taxonomy import (
    EXECUTABLE_CONDITION_FAILURE,
    INFRASTRUCTURE_FAILURE,
    NATIVE_EXECUTION_EXCEPTION,
    TARGET_BINDING_FAILURE,
    TASK_SUCCESS,
    status_record,
)
from procedural_interface.dataset.public_view_loader import SampleRef
from procedural_interface.execution.native import run_native_execution
from procedural_interface.adapter.phase_binding import bind_plan_to_phases
from procedural_interface.adapter.task_phase_templates import TASK_TEMPLATES
from procedural_interface.interface.parser import parse_plan_text


# Use one explicit version string for fixture reproducibility.
FIXTURE_VERSION = "20260618_phase_aware_no_role"
# Use one explicit dataset snapshot label for every generated fixture.
DATASET_SNAPSHOT = "vlabench_vlm4500_public_20260618"
# Use stable filenames so the smoke command can be copied from the document.
POSITIVE_FIXTURE_FILENAME = "gate3_positive_controls.json"
# Use stable filenames so the smoke command can be copied from the document.
NEGATIVE_FIXTURE_FILENAME = "gate3_negative_controls.json"
# Retry native execution only to absorb stochastic MuJoCo/RRT/IK failures.
MAX_NATIVE_RETRY_ATTEMPTS = 10
# Keep one policy label in every report row for transparent interpretation.
NATIVE_RETRY_POLICY = "same_plan_same_target_condition_failure_only"


def _native_seed_for_fixture(fixture: dict[str, Any]) -> int | None:
    """Return an explicit native-execution seed only when the fixture freezes one."""

    # Respect an explicitly frozen native seed when a smoke fixture needs one.
    if "native_seed" in fixture:
        # Cast to int so JSON numeric values and string-form audit values behave identically.
        return int(fixture["native_seed"])
    # Match official GT raw execution by not forcing a synthetic legacy seed by default.
    return None


def _reset_wait_step_for_task(task_name: str) -> int:
    """Return deterministic VLABench reset wait for the active task family."""

    # Official GT raw executions for the adapter-parity ledger used reset_wait_step=1.
    # Keep VLM skill/order/target dynamic while matching the native reset policy.
    return 1


# Positive fixtures cover the required task families from the specification.
POSITIVE_SPECS: list[dict[str, Any]] = [
    {"fixture_id": "select_pick_place_positive_001", "family": "select pick/place", "source_case_id": "M&T/select_fruit/example0", "steps": [{"skill": "pick", "target": 2}, {"skill": "place", "target": 1}], "condition_targets": ["apple", "plate_seen"]},
    {"fixture_id": "select_pick_lift_positive_001", "family": "select pick/lift", "source_case_id": "M&T/select_poker/example0", "steps": [{"skill": "pick", "target": 2}, {"skill": "lift"}], "condition_targets": ["2_of_clubs"]},
    {"fixture_id": "select_pick_pull_positive_001", "family": "select pick/pull", "source_case_id": "M&T/select_book/example0", "steps": [{"skill": "pick", "target": 4}, {"skill": "pull"}], "condition_targets": ["the_life_of_samuel_johnson"]},
    {"fixture_id": "insert_flower_positive_001", "family": "insert flower executor-level sequence", "source_case_id": "M&T/insert_flower/example1", "steps": [{"skill": "pick", "target": 2}, {"skill": "lift"}, {"skill": "moveto", "target": 1}, {"skill": "lift"}], "condition_targets": ["tulip", "vase_seen"]},
    {"fixture_id": "add_condiment_pour_positive_001", "family": "add condiment pour", "source_case_id": "M&T/add_condiment/example47", "steps": [{"skill": "pick", "target": 4}, {"skill": "lift"}, {"skill": "moveto", "target": 2}, {"skill": "pour"}], "condition_targets": ["salt", "pan_seen"]},
    {"fixture_id": "physicslaw_press_positive_001", "family": "PhysicsLaw press", "source_case_id": "PhysicsLaw/density_qa/example10", "steps": [{"skill": "press", "target": 1}], "condition_targets": ["button0"]},
    {"fixture_id": "book_rearrange_positive_001", "family": "book rearrange push/pull", "source_case_id": "Complex/book_rearrange/example0", "steps": [{"skill": "pick", "target": 3}, {"skill": "moveto", "target": 1}, {"skill": "push", "target": 3}, {"skill": "pull"}, {"skill": "pick", "target": 2}, {"skill": "moveto", "target": 1}, {"skill": "push", "target": 2}, {"skill": "pull"}, {"skill": "pick", "target": 4}, {"skill": "moveto", "target": 1}, {"skill": "push", "target": 4}, {"skill": "pull"}], "condition_targets": ["century17_don_quixote_book", "century19_boule_de_suif_book", "century21_the_three_body_problem_book", "shelf_seen"]},
    {"fixture_id": "hammer_nail_positive_001", "family": "hammer nail long sequence", "source_case_id": "Complex/hammer_nail_and_hang_picture/example0", "steps": [{"skill": "pick", "target": 4}, {"skill": "lift"}, {"skill": "pull"}, {"skill": "moveto", "target": 3}, {"skill": "push", "target": 3}, {"skill": "place", "target": 0}, {"skill": "lift"}, {"skill": "pick", "target": 2}, {"skill": "lift"}, {"skill": "moveto", "target": 3}, {"skill": "push", "target": 3}], "condition_targets": ["hammer", "nail", "the night_painting"]},
    {"fixture_id": "take_chemistry_experiment_positive_001", "family": "take chemistry experiment long sequence", "source_case_id": "Complex/take_chemistry_experiment/example2", "steps": [{"skill": "pick", "target": 3}, {"skill": "lift"}, {"skill": "moveto", "target": 1}, {"skill": "pour"}, {"skill": "wait"}, {"skill": "pull"}, {"skill": "pour"}, {"skill": "moveto", "target": 2}, {"skill": "move_offset"}, {"skill": "open_gripper"}, {"skill": "pick", "target": 5}, {"skill": "lift"}, {"skill": "moveto", "target": 1}, {"skill": "pour"}, {"skill": "wait"}, {"skill": "pull"}, {"skill": "pour"}, {"skill": "moveto", "target": 2}, {"skill": "move_offset"}, {"skill": "open_gripper"}], "condition_targets": ["Na2CO3", "HCl", "flask", "chemistry_tube_stand"]},
]


# Negative fixtures keep the same phase length as each task template.
NEGATIVE_SPECS: list[dict[str, Any]] = [
    {"base_fixture_id": "select_pick_place", "family": "select pick/place", "source_case_id": "M&T/select_fruit/example0", "negative_type": "wrong_target_executable", "steps": [{"skill": "pick", "target": 3}, {"skill": "place", "target": 1}]},
    {"base_fixture_id": "select_pick_place", "family": "select pick/place", "source_case_id": "M&T/select_fruit/example0", "negative_type": "incompatible_target", "steps": [{"skill": "pick", "target": 0}, {"skill": "place", "target": 1}]},
    {"base_fixture_id": "select_pick_lift", "family": "select pick/lift", "source_case_id": "M&T/select_poker/example0", "negative_type": "wrong_target_executable", "steps": [{"skill": "pick", "target": 4}, {"skill": "lift"}]},
    {"base_fixture_id": "select_pick_lift", "family": "select pick/lift", "source_case_id": "M&T/select_poker/example0", "negative_type": "incompatible_target", "steps": [{"skill": "pick", "target": 1}, {"skill": "lift"}]},
    {"base_fixture_id": "select_pick_pull", "family": "select pick/pull", "source_case_id": "M&T/select_book/example0", "negative_type": "wrong_target_executable", "steps": [{"skill": "pick", "target": 2}, {"skill": "pull"}]},
    {"base_fixture_id": "select_pick_pull", "family": "select pick/pull", "source_case_id": "M&T/select_book/example0", "negative_type": "incompatible_target", "steps": [{"skill": "pick", "target": 1}, {"skill": "pull"}]},
    {"base_fixture_id": "insert_flower", "family": "insert flower executor-level sequence", "source_case_id": "M&T/insert_flower/example1", "negative_type": "wrong_target_executable", "steps": [{"skill": "pick", "target": 3}, {"skill": "lift"}, {"skill": "moveto", "target": 1}, {"skill": "lift"}]},
    {"base_fixture_id": "insert_flower", "family": "insert flower executor-level sequence", "source_case_id": "M&T/insert_flower/example1", "negative_type": "incompatible_target", "steps": [{"skill": "pick", "target": 0}, {"skill": "lift"}, {"skill": "moveto", "target": 1}, {"skill": "lift"}]},
    {"base_fixture_id": "add_condiment_pour", "family": "add condiment pour", "source_case_id": "M&T/add_condiment/example47", "negative_type": "wrong_target_executable", "steps": [{"skill": "pick", "target": 6}, {"skill": "lift"}, {"skill": "moveto", "target": 2}, {"skill": "pour"}]},
    {"base_fixture_id": "add_condiment_pour", "family": "add condiment pour", "source_case_id": "M&T/add_condiment/example47", "negative_type": "incompatible_target", "steps": [{"skill": "pick", "target": 0}, {"skill": "lift"}, {"skill": "moveto", "target": 2}, {"skill": "pour"}]},
    {"base_fixture_id": "physicslaw_press", "family": "PhysicsLaw press", "source_case_id": "PhysicsLaw/density_qa/example10", "negative_type": "wrong_target_executable", "steps": [{"skill": "press", "target": 3}]},
    {"base_fixture_id": "physicslaw_press", "family": "PhysicsLaw press", "source_case_id": "PhysicsLaw/density_qa/example10", "negative_type": "incompatible_target", "steps": [{"skill": "press", "target": 2}]},
    {"base_fixture_id": "book_rearrange", "family": "book rearrange push/pull", "source_case_id": "Complex/book_rearrange/example0", "negative_type": "wrong_target_executable", "steps": [{"skill": "pick", "target": 2}, {"skill": "moveto", "target": 1}, {"skill": "push", "target": 2}, {"skill": "pull"}, {"skill": "pick", "target": 3}, {"skill": "moveto", "target": 1}, {"skill": "push", "target": 3}, {"skill": "pull"}, {"skill": "pick", "target": 4}, {"skill": "moveto", "target": 1}, {"skill": "push", "target": 4}, {"skill": "pull"}]},
    {"base_fixture_id": "book_rearrange", "family": "book rearrange push/pull", "source_case_id": "Complex/book_rearrange/example0", "negative_type": "incompatible_target", "steps": [{"skill": "pick", "target": 0}, {"skill": "moveto", "target": 1}, {"skill": "push", "target": 3}, {"skill": "pull"}, {"skill": "pick", "target": 2}, {"skill": "moveto", "target": 1}, {"skill": "push", "target": 2}, {"skill": "pull"}, {"skill": "pick", "target": 4}, {"skill": "moveto", "target": 1}, {"skill": "push", "target": 4}, {"skill": "pull"}]},
    {"base_fixture_id": "hammer_nail", "family": "hammer nail long sequence", "source_case_id": "Complex/hammer_nail_and_hang_picture/example0", "negative_type": "wrong_target_executable", "steps": [{"skill": "pick", "target": 4}, {"skill": "lift"}, {"skill": "pull"}, {"skill": "moveto", "target": 1}, {"skill": "push", "target": 1}, {"skill": "place", "target": 0}, {"skill": "lift"}, {"skill": "pick", "target": 2}, {"skill": "lift"}, {"skill": "moveto", "target": 1}, {"skill": "push", "target": 1}]},
    {"base_fixture_id": "hammer_nail", "family": "hammer nail long sequence", "source_case_id": "Complex/hammer_nail_and_hang_picture/example0", "negative_type": "incompatible_target", "steps": [{"skill": "pick", "target": 0}, {"skill": "lift"}, {"skill": "pull"}, {"skill": "moveto", "target": 3}, {"skill": "push", "target": 3}, {"skill": "place", "target": 0}, {"skill": "lift"}, {"skill": "pick", "target": 1}, {"skill": "lift"}, {"skill": "moveto", "target": 3}, {"skill": "push", "target": 3}]},
    {"base_fixture_id": "take_chemistry_experiment", "family": "take chemistry experiment long sequence", "source_case_id": "Complex/take_chemistry_experiment/example2", "negative_type": "wrong_target_executable", "steps": [{"skill": "pick", "target": 7}, {"skill": "lift"}, {"skill": "moveto", "target": 1}, {"skill": "pour"}, {"skill": "wait"}, {"skill": "pull"}, {"skill": "pour"}, {"skill": "moveto", "target": 2}, {"skill": "move_offset"}, {"skill": "open_gripper"}, {"skill": "pick", "target": 5}, {"skill": "lift"}, {"skill": "moveto", "target": 1}, {"skill": "pour"}, {"skill": "wait"}, {"skill": "pull"}, {"skill": "pour"}, {"skill": "moveto", "target": 2}, {"skill": "move_offset"}, {"skill": "open_gripper"}]},
    {"base_fixture_id": "take_chemistry_experiment", "family": "take chemistry experiment long sequence", "source_case_id": "Complex/take_chemistry_experiment/example2", "negative_type": "incompatible_target", "steps": [{"skill": "pick", "target": 0}, {"skill": "lift"}, {"skill": "moveto", "target": 1}, {"skill": "pour"}, {"skill": "wait"}, {"skill": "pull"}, {"skill": "pour"}, {"skill": "moveto", "target": 2}, {"skill": "move_offset"}, {"skill": "open_gripper"}, {"skill": "pick", "target": 5}, {"skill": "lift"}, {"skill": "moveto", "target": 1}, {"skill": "pour"}, {"skill": "wait"}, {"skill": "pull"}, {"skill": "pour"}, {"skill": "moveto", "target": 2}, {"skill": "move_offset"}, {"skill": "open_gripper"}]},
]


def _sample_from_case_id(dataset_root: Path, case_id: str) -> SampleRef:
    """Build a SampleRef from a slash-form public case id."""

    # Split category, task, and example id from the fixture source id.
    category, task_name, example_id = case_id.split("/")
    # Build the public-view example directory path.
    root = dataset_root / category / task_name / example_id
    # Return the current SampleRef dataclass shape.
    return SampleRef(category=category, task_name=task_name, example_id=example_id, root=root)


def _component_lookup(env_config: dict) -> dict[int, dict[str, Any]]:
    """Return numeric component lookup from an env_config."""

    # Read components from allowed internal scene config.
    components = env_config.get("task", {}).get("components", [])
    # Enumerate the visible component list exactly as numeric target binding does.
    return {number: component for number, component in enumerate(components)}


def _fixture_from_spec(dataset_root: Path, spec: dict[str, Any], mode: str, index: int) -> dict[str, Any]:
    """Create one frozen fixture from an offline fixture spec."""

    # Load the public-view sample for offline fixture validation.
    sample = _sample_from_case_id(dataset_root, spec["source_case_id"])
    # Read env_config during fixture generation only.
    env_config = sample.load_env_config()
    # Build numeric target lookup from components.
    components = _component_lookup(env_config)
    # Serialize conditions only for offline provenance validation.
    condition_blob = json.dumps(env_config.get("task", {}).get("conditions", {}), ensure_ascii=False)
    # Collect component names for human-readable fixture metadata.
    component_names = [component.get("name", "") for component in components.values()]
    # Check that declared condition targets are present in components or conditions.
    missing_targets = [name for name in spec.get("condition_targets", []) if name not in component_names and name not in condition_blob]
    # Refuse to build stale fixtures against a changed dataset snapshot.
    if missing_targets:
        # Raise an explicit generation error.
        raise ValueError(f"{spec.get('fixture_id', spec.get('base_fixture_id'))} missing target evidence: {missing_targets}")
    # Gather numeric target ids used by the frozen steps.
    used_targets = sorted({step["target"] for step in spec["steps"] if "target" in step})
    # Verify every target id is in range.
    for target in used_targets:
        # Out-of-range frozen fixtures indicate a spec bug, not a model failure.
        if target not in components:
            # Stop fixture generation with a clear error.
            raise ValueError(f"{spec.get('fixture_id', spec.get('base_fixture_id'))} target {target} out of range")
    # Build human-readable target metadata for audit only.
    human_readable_targets = {str(target): components[target].get("name", "") for target in used_targets}
    # Build a stable fixture id for positives and negatives.
    fixture_id = spec.get("fixture_id") or f"{spec['base_fixture_id']}_{spec['negative_type']}_{index:03d}"
    # Determine whether native execution should run.
    native_should_run = mode == "positive_control" or spec.get("negative_type") == "wrong_target_executable"
    # Determine expected final status for negative controls.
    expected_failure_code = "" if mode == "positive_control" else (EXECUTABLE_CONDITION_FAILURE.code if native_should_run else TARGET_BINDING_FAILURE.code)
    # Determine expected subtype for incompatible controls.
    expected_failure_subtype = "" if native_should_run else "incompatible_target_affordance"
    # Return the frozen fixture record.
    return {
        "fixture_id": fixture_id,
        "fixture_version": FIXTURE_VERSION,
        "fixture_mode": mode,
        "family": spec["family"],
        "fixture_generated_from_gt": True,
        "fixture_generation_phase": "offline_only",
        "source_fields_used": ["conditions", "components"],
        "frozen_after_generation": True,
        "dataset_snapshot": DATASET_SNAPSHOT,
        "source_case_id": spec["source_case_id"],
        "target_source": "frozen_numeric_fixture",
        "counts_as_model_evaluation": False,
        "expected_gate1_ok": native_should_run,
        "expected_native_ok": native_should_run,
        "expected_condition_success": mode == "positive_control",
        "negative_type": spec.get("negative_type", ""),
        "expected_failure_code": expected_failure_code,
        "expected_failure_subtype": expected_failure_subtype,
        "native_execution_should_run": native_should_run,
        "steps": spec["steps"],
        "human_readable_targets": human_readable_targets,
    }


def build_gate3_fixtures(dataset_root: str | Path, fixture_root: str | Path) -> dict[str, Any]:
    """Build frozen positive-control fixture manifests."""

    # Normalize dataset root from CLI input.
    dataset_root_path = Path(dataset_root)
    # Create fixture output root.
    fixture_root_path = ensure_dir(fixture_root)
    # Build positive fixtures from offline specs.
    positive = [_fixture_from_spec(dataset_root_path, spec, "positive_control", index) for index, spec in enumerate(POSITIVE_SPECS, start=1)]
    # Build negative fixtures from offline specs.
    negative = [_fixture_from_spec(dataset_root_path, spec, "negative_control", index) for index, spec in enumerate(NEGATIVE_SPECS, start=1)]
    # Write positive fixture manifest.
    positive_path = write_json(fixture_root_path / POSITIVE_FIXTURE_FILENAME, positive)
    # Write negative fixture manifest.
    negative_path = write_json(fixture_root_path / NEGATIVE_FIXTURE_FILENAME, negative)
    # Build fixture generation summary.
    summary = {
        "status": "built",
        "fixture_version": FIXTURE_VERSION,
        "dataset_root": str(dataset_root_path),
        "fixture_root": str(fixture_root_path),
        "positive_fixtures": len(positive),
        "negative_fixtures": len(negative),
        "positive_path": str(positive_path),
        "negative_path": str(negative_path),
        "gt_usage_scope": "offline_fixture_generation_only",
        "counts_as_model_evaluation": False,
    }
    # Write summary provenance.
    write_json(fixture_root_path / "gate3_fixture_generation_summary.json", summary)
    # Return summary to CLI.
    return summary


def _plan_text_from_fixture(fixture: dict[str, Any]) -> str:
    """Serialize only model-facing steps from a frozen fixture."""

    # Keep fixture metadata out of parser input.
    return json.dumps({"steps": fixture["steps"]}, ensure_ascii=False)


def _class_for_target(env_config: dict, target_number: int) -> str:
    """Read component class for a numeric target."""

    # Read components from allowed internal scene config.
    components = env_config.get("task", {}).get("components", [])
    # Return an empty class when the component is malformed.
    return str(components[target_number].get("class", "")) if 0 <= target_number < len(components) else ""


def _check_affordance(parsed_steps: list[dict[str, Any]], env_config: dict) -> dict[str, Any]:
    """Reject target/skill combinations that cannot be executed by affordance."""

    # Track whether the robot is holding an object for no-target skills.
    holding_object = False
    # Iterate parsed steps in model order.
    for index, step in enumerate(parsed_steps, start=1):
        # Read the skill.
        skill = step["skill"]
        # Read target class when a target exists.
        target_class = _class_for_target(env_config, int(step["target"])) if "target" in step else ""
        # Pick requires a graspable object, not a fixture/container.
        if skill == "pick" and target_class in {"Table", "Plate", "Shelf", "Counter", "Stove", "Button", "NameTag", "TubeStand", "FlatContainer", "Vase", "CommonContainer", "CardHolder"}:
            # Return a C3 incompatible target failure.
            return {"ok": False, "status": status_record(TARGET_BINDING_FAILURE, "incompatible_target_affordance", f"step={index}, skill=pick, class={target_class}")}
        # Press requires a Button component.
        if skill == "press" and target_class != "Button":
            # Return a C3 incompatible target failure.
            return {"ok": False, "status": status_record(TARGET_BINDING_FAILURE, "incompatible_target_affordance", f"step={index}, skill=press, class={target_class}")}
        # Mark the gripper as holding after an executable pick.
        if skill == "pick":
            # Record held-object state for later lift/pour/open_gripper checks.
            holding_object = True
        # Lift is allowed even after place because several official templates use it as retreat.
        # Pour requires a current held object.
        if skill == "pour" and not holding_object:
            # Return a C3 state precondition failure.
            return {"ok": False, "status": status_record(TARGET_BINDING_FAILURE, "invalid_gripper_context", f"step={index}, skill=pour")}
        # Open gripper requires a current held object.
        if skill == "open_gripper" and not holding_object:
            # Return a C3 state precondition failure.
            return {"ok": False, "status": status_record(TARGET_BINDING_FAILURE, "invalid_gripper_context", f"step={index}, skill=open_gripper")}
        # Open gripper releases the held object if it runs.
        if skill in {"open_gripper"}:
            # Clear held-object state after release.
            holding_object = False
    # Return success when no incompatible target or state was found.
    return {"ok": True, "status": {"code": "", "label": "", "subtype": "", "detail": ""}}


def _bind_fixture_plan(sample: SampleRef, fixture: dict[str, Any]) -> dict[str, Any]:
    """Parse, affordance-check, and phase-bind a frozen fixture plan."""

    # Load env_config for numeric target binding.
    env_config = sample.load_env_config()
    # Parse the frozen plan through the real no-role parser.
    parsed = parse_plan_text(_plan_text_from_fixture(fixture))
    # Stop if parser rejected the fixture.
    if not parsed.get("ok"):
        # Return parser status directly.
        return {"ok": False, "status": parsed["status"], "parsed": parsed, "phase_trace": [], "executable_steps": []}
    # Check target affordance before native execution.
    affordance = _check_affordance(parsed["steps"], env_config)
    # Stop incompatible fixtures before phase binding/native execution.
    if not affordance.get("ok"):
        # Return the deterministic C3 failure.
        return {"ok": False, "status": affordance["status"], "parsed": parsed, "phase_trace": [], "executable_steps": []}
    # Read the task template for this sample.
    template = TASK_TEMPLATES[sample.task_name]
    # Bind parsed steps to deterministic task phases.
    # Read explicit target provenance from the fixture; full B keeps VLM defaults.
    target_source = str(fixture.get("target_source", "vlm_output"))
    # Target-oracle fixtures are the only model-evaluation fixtures that use GT targets.
    ground_truth_target_used = bool(fixture.get("ground_truth_target_used", False))
    # Bind steps without changing numbers while preserving source provenance.
    phase_result = bind_plan_to_phases(
        parsed["steps"],
        env_config,
        template,
        target_source=target_source,
        ground_truth_target_used=ground_truth_target_used,
    )
    # Stop if phase binding rejected the fixture.
    if not phase_result.get("ok"):
        # Return phase binding status directly.
        return {"ok": False, "status": phase_result["status"], "parsed": parsed, "phase_trace": phase_result.get("phase_trace", []), "executable_steps": []}
    # Convert phase trace rows into native worker executable steps.
    executable_steps = []
    # Iterate accepted phase rows in order.
    for trace_row in phase_result["phase_trace"]:
        # Start each executable step with index, skill, and phase provenance.
        executable_step = {
            "index": trace_row["vlm_step_index"] - 1,
            "skill": trace_row["vlm_skill"],
            "matched_phase_template": trace_row["matched_phase_template"],
            "phase_cursor_source": trace_row["phase_cursor_source"],
            "sequence_reordered": False,
            "missing_phase_inserted": False,
            "gt_target_used": bool(trace_row.get("gt_target_used", False)),
            "operation_sequence_used": False,
            "expert_sequence_called": False,
        }
        # Add target binding only when the phase had a target.
        if trace_row.get("target_binding"):
            # Preserve model-selected target number and bound entity info.
            executable_step["target"] = {
                "number": trace_row["target_binding"]["target_number"],
                "entity_name": trace_row["target_binding"]["entity_name"],
                "entity_class": trace_row["target_binding"]["entity_class"],
                "target_source": trace_row["target_binding"]["target_source"],
            }
        # Append the executable step.
        executable_steps.append(executable_step)
    # Return accepted executable plan.
    return {"ok": True, "status": {"code": "", "label": "", "subtype": "", "detail": ""}, "parsed": parsed, "phase_trace": phase_result["phase_trace"], "executable_steps": executable_steps}


def _status_from_native(gate_result: dict[str, Any], native_result: dict[str, Any] | None) -> dict[str, str]:
    """Map binding/native results into procedural-evaluation status codes."""

    # Return parser/binding/precondition status when plan-validation-side checks failed.
    if not gate_result.get("ok"):
        # Preserve the exact C2/C3/C4 subtype.
        return gate_result["status"]
    # Missing native result is infrastructure failure.
    if native_result is None:
        # Return explicit INFRA status.
        return status_record(INFRASTRUCTURE_FAILURE, "missing_native_result", "native result missing")
    # Worker infrastructure failures are INFRA.
    if native_result.get("infrastructure_error"):
        # Return explicit INFRA status.
        return status_record(INFRASTRUCTURE_FAILURE, "native_worker_infrastructure", native_result.get("detail", ""))
    # Native exceptions are C5.
    if not native_result.get("ok"):
        # Preserve timeout as an explicit subtype because the worker did not raise a native exception.
        if native_result.get("native_timeout"):
            # Keep the timeout budget in detail so the report explains the failure mode.
            return status_record(NATIVE_EXECUTION_EXCEPTION, "native_timeout", f"timeout_sec={native_result.get('timeout_sec', '')}")
        # Preserve exception type as subtype when available.
        return status_record(NATIVE_EXECUTION_EXCEPTION, native_result.get("exception_type", "native_exception"), native_result.get("exception", ""))
    # Native condition success is task success.
    if native_result.get("condition_success"):
        # Return C0 success.
        return status_record(TASK_SUCCESS, "", "condition_success=true")
    # Completed native execution with failed condition is C1.
    return status_record(EXECUTABLE_CONDITION_FAILURE, "", "condition_success=false")


def _fixture_passed(fixture: dict[str, Any], gate_result: dict[str, Any], native_attempted: bool, native_result: dict[str, Any] | None, status: dict[str, str]) -> bool:
    """Evaluate one fixture against its expected Gate3 behavior."""

    # Resolve native success from the optional native result.
    native_ok = bool(native_result.get("ok")) if native_result else False
    # Resolve condition success from the optional native result.
    condition_success = bool(native_result.get("condition_success")) if native_result else False
    # Positive controls must pass plan validation, native execution, and condition success.
    if fixture["fixture_mode"] == "positive_control":
        # Return exact expectation match.
        return bool(gate_result.get("ok")) and native_ok and condition_success
    # Wrong-target executable controls must run natively and fail condition.
    if fixture.get("negative_type") == "wrong_target_executable":
        # Return true only when the expected C1 path occurs.
        return bool(gate_result.get("ok")) and native_attempted and native_ok and not condition_success and status["code"] == EXECUTABLE_CONDITION_FAILURE.code
    # Incompatible controls must stop before native execution with C3.
    if fixture.get("negative_type") == "incompatible_target":
        # Return true only when C3 happens before native execution.
        return (not gate_result.get("ok")) and (not native_attempted) and status["code"] == TARGET_BINDING_FAILURE.code and status["subtype"] == "incompatible_target_affordance"
    # Unknown fixture modes must never pass silently.
    return False


def _should_retry_native_execution(fixture: dict[str, Any], native_result: dict[str, Any] | None) -> bool:
    """Return true when a retry can only test native planner stochasticity."""

    # Never retry when native did not produce a structured result.
    if native_result is None:
        # Missing output is an infrastructure signal, not a condition retry case.
        return False
    # Retry only positive controls because wrong-target negatives are expected to fail condition.
    if fixture.get("fixture_mode") != "positive_control":
        # Negative controls must preserve their first observed failure.
        return False
    # Retry only when the native worker itself completed successfully.
    if not native_result.get("ok"):
        # Timeouts and native exceptions must remain visible as C5.
        return False
    # Stop retrying as soon as the native condition succeeds.
    if native_result.get("condition_success"):
        # The fixture already reached C0.
        return False
    # A completed native rollout with condition false is the stochastic C1 boundary.
    return True


def _summarize_native_attempt(attempt_index: int, native_result: dict[str, Any]) -> dict[str, Any]:
    """Build a compact per-attempt audit record."""

    # Return only stable, report-safe fields from the native worker output.
    return {
        "attempt": attempt_index,
        "ok": bool(native_result.get("ok")),
        "execution_completed": bool(native_result.get("execution_completed")),
        "condition_success": bool(native_result.get("condition_success")),
        "exception_type": native_result.get("exception_type", ""),
        "trace_length": len(native_result.get("trace", [])) if isinstance(native_result.get("trace"), list) else 0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write variable-width report rows to CSV."""

    # Create the report directory.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Collect fieldnames in first-seen order.
    fieldnames: list[str] = []
    # Inspect every row for optional keys.
    for row in rows:
        # Preserve row key order.
        for key in row:
            # Add unseen keys once.
            if key not in fieldnames:
                # Append the new field.
                fieldnames.append(key)
    # Open with utf-8-sig for Windows spreadsheet compatibility.
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        # Create a DictWriter with stable field order.
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Write the header.
        writer.writeheader()
        # Write all report rows.
        writer.writerows(rows)


def _run_one_fixture(dataset_root: Path, fixture: dict[str, Any], trace_root: Path, timeout_sec: int) -> dict[str, Any]:
    """Run one frozen fixture through parser, phase binding, and optional native execution."""

    # Build the sample reference from fixture metadata.
    sample = _sample_from_case_id(dataset_root, fixture["source_case_id"])
    # Create a dedicated trace directory for this fixture.
    fixture_trace_root = ensure_dir(trace_root / fixture["fixture_id"])
    # Save the exact frozen fixture used for this run.
    write_json(fixture_trace_root / "fixture.json", fixture)
    # Parse, affordance-check, and phase-bind the frozen steps.
    gate_result = _bind_fixture_plan(sample, fixture)
    # Save the plan-validation-side result for provenance.
    write_json(fixture_trace_root / "gate1_phase_result.json", gate_result)
    # Native execution starts as not attempted.
    native_attempted = False
    # Native result starts absent.
    native_result: dict[str, Any] | None = None
    # Track every same-plan native attempt for auditability.
    native_attempt_summaries: list[dict[str, Any]] = []
    # Track which attempt produced the selected native result.
    selected_native_attempt = 0
    # Run native execution only when parser/affordance/phase binding accepted the fixture.
    if gate_result.get("ok"):
        # Mark native as attempted.
        native_attempted = True
        # Build native command from executable steps only.
        native_command = {
            "task_name": sample.task_name,
            "env_config_path": str(sample.env_config_path),
            "executable_steps": gate_result["executable_steps"],
            "reset_wait_step": _reset_wait_step_for_task(sample.task_name),
            "native_seed": _native_seed_for_fixture(fixture),
            "visual_audit_dir": str(fixture_trace_root / "visual_audit"),
        }
        # Save the controller-level native command.
        write_json(fixture_trace_root / "native_command.json", native_command)
        # Execute the same native command more than once only for stochastic C1 outcomes.
        for attempt_index in range(1, MAX_NATIVE_RETRY_ATTEMPTS + 1):
            # Store each attempt under its own directory so no evidence is overwritten.
            attempt_root = fixture_trace_root / f"execution_attempt_{attempt_index:02d}"
            # Execute native worker and capture the structured result.
            attempt_result = run_native_execution(native_command, attempt_root, timeout_sec=timeout_sec)
            # Save a stable copy of this attempt at fixture root.
            write_json(fixture_trace_root / f"native_result_attempt_{attempt_index:02d}.json", attempt_result)
            # Append a compact audit row for the final report.
            native_attempt_summaries.append(_summarize_native_attempt(attempt_index, attempt_result))
            # Select the latest attempt as the active native result.
            native_result = attempt_result
            # Record which attempt the final row represents.
            selected_native_attempt = attempt_index
            # Stop immediately when C0 is reached.
            if not _should_retry_native_execution(fixture, attempt_result):
                # Keep successful, exception, timeout, and expected-negative outcomes as observed.
                break
        # Save a stable copy at fixture root.
        write_json(fixture_trace_root / "native_result.json", native_result)
    # Convert the outcome into a procedural-evaluation status.
    status = _status_from_native(gate_result, native_result)
    # Evaluate pass/fail against fixture expectation.
    passed = _fixture_passed(fixture, gate_result, native_attempted, native_result, status)
    # Build the report row.
    row = {
        "fixture_id": fixture["fixture_id"],
        "family": fixture["family"],
        "fixture_mode": fixture["fixture_mode"],
        "negative_type": fixture.get("negative_type", ""),
        "source_case_id": fixture["source_case_id"],
        "fixture_generated_from_gt": fixture["fixture_generated_from_gt"],
        "gt_used_during_execution": False,
        "adapter_read_conditions": False,
        "lowerer_read_conditions": False,
        "operation_sequence_read": False,
        "expert_sequence_called": False,
        "target_source": fixture["target_source"],
        "counts_as_model_evaluation": fixture["counts_as_model_evaluation"],
        "native_execution_attempted": native_attempted,
        "native_execution_should_run": fixture["native_execution_should_run"],
        "native_retry_policy": NATIVE_RETRY_POLICY if native_attempted else "",
        "native_max_attempts": MAX_NATIVE_RETRY_ATTEMPTS if native_attempted else 0,
        "native_attempt_count": len(native_attempt_summaries),
        "native_selected_attempt": selected_native_attempt,
        "native_attempt_summaries": json.dumps(native_attempt_summaries, ensure_ascii=False),
        "expected_gate1_ok": fixture["expected_gate1_ok"],
        "expected_native_ok": fixture["expected_native_ok"],
        "expected_condition_success": fixture["expected_condition_success"],
        "actual_gate1_ok": bool(gate_result.get("ok")),
        "actual_native_ok": bool(native_result.get("ok")) if native_result else False,
        "actual_condition_success": bool(native_result.get("condition_success")) if native_result else False,
        "actual_failure_code": status["code"],
        "actual_failure_subtype": status["subtype"],
        "actual_failure_detail": status["detail"],
        "exception_type": native_result.get("exception_type", "") if native_result else "",
        "episode_config_added_fields": json.dumps(native_result.get("episode_config_audit", {}).get("added_fields", {}), ensure_ascii=False) if native_result else "",
        "before_contact_sheet": native_result.get("visual_audit", {}).get("before", {}).get("contact_sheet", "") if native_result else "",
        "after_contact_sheet": native_result.get("visual_audit", {}).get("after", {}).get("contact_sheet", "") if native_result else "",
        "passed": passed,
        "trace_dir": str(fixture_trace_root),
    }
    # Save the per-fixture row as JSON too.
    write_json(fixture_trace_root / "result_summary.json", row)
    # Return the report row.
    return row


def run_gate3_smoke(dataset_root: str | Path, positive_fixtures: str | Path, negative_fixtures: str | Path, output_root: str | Path, timeout_sec: int) -> dict[str, Any]:
    """Run procedural-adapter smoke on frozen positive and negative fixtures."""

    # Normalize dataset root from CLI.
    dataset_root_path = Path(dataset_root)
    # Create output root.
    output_root_path = ensure_dir(output_root)
    # Read frozen positive manifest.
    positive = json.loads(Path(positive_fixtures).read_text(encoding="utf-8"))
    # Read frozen negative manifest.
    negative = json.loads(Path(negative_fixtures).read_text(encoding="utf-8"))
    # Create trace root for per-fixture artifacts.
    trace_root = ensure_dir(output_root_path / "smoke_execution_traces")
    # Execute all positive fixtures.
    positive_rows = [_run_one_fixture(dataset_root_path, fixture, trace_root, timeout_sec) for fixture in positive]
    # Execute all negative fixtures.
    negative_rows = [_run_one_fixture(dataset_root_path, fixture, trace_root, timeout_sec) for fixture in negative]
    # Write positive report.
    _write_csv(output_root_path / "task_family_smoke_report.csv", positive_rows)
    # Write negative report.
    _write_csv(output_root_path / "negative_control_report.csv", negative_rows)
    # Combine rows for summary.
    all_rows = positive_rows + negative_rows
    # Identify failed fixtures.
    failed_fixture_ids = [row["fixture_id"] for row in all_rows if not row["passed"]]
    # Count native exception outcomes.
    native_exception_count = sum(1 for row in all_rows if row["actual_failure_code"] == NATIVE_EXECUTION_EXCEPTION.code)
    # Count infrastructure failures.
    infrastructure_failure_count = sum(1 for row in all_rows if row["actual_failure_code"] == INFRASTRUCTURE_FAILURE.code)
    # Verify execution-time GT guard fields are all false.
    gt_guard_passed = all(not row["gt_used_during_execution"] and not row["adapter_read_conditions"] and not row["lowerer_read_conditions"] and not row["operation_sequence_read"] and not row["expert_sequence_called"] for row in all_rows)
    # Build summary.
    summary = {
        "gate": "Gate3 phase-aware task-family smoke",
        "status": "passed" if not failed_fixture_ids and gt_guard_passed else "failed",
        "dataset_root": str(dataset_root_path),
        "positive_fixture_count": len(positive_rows),
        "negative_fixture_count": len(negative_rows),
        "total_fixture_count": len(all_rows),
        "passed_fixture_count": sum(1 for row in all_rows if row["passed"]),
        "failed_fixture_count": len(failed_fixture_ids),
        "failed_fixture_ids": failed_fixture_ids,
        "native_exception_count": native_exception_count,
        "infrastructure_failure_count": infrastructure_failure_count,
        "gt_guard_passed": gt_guard_passed,
        "counts_as_model_evaluation": False,
        "outputs": {
            "task_family_smoke_report": str(output_root_path / "task_family_smoke_report.csv"),
            "negative_control_report": str(output_root_path / "negative_control_report.csv"),
            "gate3_summary": str(output_root_path / "gate3_summary.json"),
            "smoke_execution_traces": str(trace_root),
        },
    }
    # Write summary JSON.
    write_json(output_root_path / "gate3_summary.json", summary)
    # Return summary to CLI.
    return summary
