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


MAX_NATIVE_RETRY_ATTEMPTS = 10
NATIVE_RETRY_POLICY = "same_plan_same_target_condition_failure_only"


def _native_seed_for_fixture(fixture: dict[str, Any]) -> int | None:
    """Return an explicit native-execution seed only when the fixture defines one."""

    if "native_seed" in fixture:
        return int(fixture["native_seed"])
    return None


def _reset_wait_step_for_task(task_name: str) -> int:
    """Return the reset wait used by the canonical execution protocol."""

    return 1


def _sample_from_case_id(dataset_root: Path, case_id: str) -> SampleRef:
    """Build a sample reference from a slash-form public case identifier."""

    category, task_name, example_id = case_id.split("/")
    root = dataset_root / category / task_name / example_id
    return SampleRef(category=category, task_name=task_name, example_id=example_id, root=root)

def _plan_text_from_fixture(fixture: dict[str, Any]) -> str:
    """Serialize only model-facing steps from a frozen fixture."""

    return json.dumps({"steps": fixture["steps"]}, ensure_ascii=False)


def _class_for_target(env_config: dict, target_number: int) -> str:
    """Read component class for a numeric target."""

    components = env_config.get("task", {}).get("components", [])
    return str(components[target_number].get("class", "")) if 0 <= target_number < len(components) else ""


def _check_affordance(parsed_steps: list[dict[str, Any]], env_config: dict) -> dict[str, Any]:
    """Reject target/skill combinations that cannot be executed by affordance."""

    holding_object = False
    for index, step in enumerate(parsed_steps, start=1):
        skill = step["skill"]
        target_class = _class_for_target(env_config, int(step["target"])) if "target" in step else ""
        if skill == "pick" and target_class in {"Table", "Plate", "Shelf", "Counter", "Stove", "Button", "NameTag", "TubeStand", "FlatContainer", "Vase", "CommonContainer", "CardHolder"}:
            return {"ok": False, "status": status_record(TARGET_BINDING_FAILURE, "incompatible_target_affordance", f"step={index}, skill=pick, class={target_class}")}
        if skill == "press" and target_class != "Button":
            return {"ok": False, "status": status_record(TARGET_BINDING_FAILURE, "incompatible_target_affordance", f"step={index}, skill=press, class={target_class}")}
        if skill == "pick":
            holding_object = True
        if skill == "pour" and not holding_object:
            return {"ok": False, "status": status_record(TARGET_BINDING_FAILURE, "invalid_gripper_context", f"step={index}, skill=pour")}
        if skill == "open_gripper" and not holding_object:
            return {"ok": False, "status": status_record(TARGET_BINDING_FAILURE, "invalid_gripper_context", f"step={index}, skill=open_gripper")}
        if skill in {"open_gripper"}:
            holding_object = False
    return {"ok": True, "status": {"code": "", "label": "", "subtype": "", "detail": ""}}


def _bind_fixture_plan(sample: SampleRef, fixture: dict[str, Any]) -> dict[str, Any]:
    """Parse, affordance-check, and phase-bind a frozen fixture plan."""

    env_config = sample.load_env_config()
    parsed = parse_plan_text(_plan_text_from_fixture(fixture))
    if not parsed.get("ok"):
        return {"ok": False, "status": parsed["status"], "parsed": parsed, "phase_trace": [], "executable_steps": []}
    affordance = _check_affordance(parsed["steps"], env_config)
    if not affordance.get("ok"):
        return {"ok": False, "status": affordance["status"], "parsed": parsed, "phase_trace": [], "executable_steps": []}
    template = TASK_TEMPLATES[sample.task_name]
    phase_result = bind_plan_to_phases(parsed["steps"], env_config, template)
    if not phase_result.get("ok"):
        return {"ok": False, "status": phase_result["status"], "parsed": parsed, "phase_trace": phase_result.get("phase_trace", []), "executable_steps": []}
    executable_steps = []
    for trace_row in phase_result["phase_trace"]:
        executable_step = {
            "index": trace_row["vlm_step_index"] - 1,
            "skill": trace_row["vlm_skill"],
            "matched_phase_template": trace_row["matched_phase_template"],
            "phase_cursor_source": trace_row["phase_cursor_source"],
            "sequence_reordered": False,
            "missing_phase_inserted": False,
            "gt_target_used": False,
            "operation_sequence_used": False,
            "expert_sequence_called": False,
        }
        if trace_row.get("target_binding"):
            executable_step["target"] = {
                "number": trace_row["target_binding"]["target_number"],
                "entity_name": trace_row["target_binding"]["entity_name"],
                "entity_class": trace_row["target_binding"]["entity_class"],
                "target_source": trace_row["target_binding"]["target_source"],
            }
        executable_steps.append(executable_step)
    return {"ok": True, "status": {"code": "", "label": "", "subtype": "", "detail": ""}, "parsed": parsed, "phase_trace": phase_result["phase_trace"], "executable_steps": executable_steps}


def _status_from_native(gate_result: dict[str, Any], native_result: dict[str, Any] | None) -> dict[str, str]:
    """Map binding/native results into procedural-evaluation status codes."""

    if not gate_result.get("ok"):
        return gate_result["status"]
    if native_result is None:
        return status_record(INFRASTRUCTURE_FAILURE, "missing_native_result", "native result missing")
    if native_result.get("infrastructure_error"):
        return status_record(INFRASTRUCTURE_FAILURE, "native_worker_infrastructure", native_result.get("detail", ""))
    if not native_result.get("ok"):
        if native_result.get("native_timeout"):
            return status_record(NATIVE_EXECUTION_EXCEPTION, "native_timeout", f"timeout_sec={native_result.get('timeout_sec', '')}")
        return status_record(NATIVE_EXECUTION_EXCEPTION, native_result.get("exception_type", "native_exception"), native_result.get("exception", ""))
    if native_result.get("condition_success"):
        return status_record(TASK_SUCCESS, "", "condition_success=true")
    return status_record(EXECUTABLE_CONDITION_FAILURE, "", "condition_success=false")


def _fixture_passed(fixture: dict[str, Any], gate_result: dict[str, Any], native_attempted: bool, native_result: dict[str, Any] | None, status: dict[str, str]) -> bool:
    """Evaluate one fixture against its expected validation and execution behavior."""

    native_ok = bool(native_result.get("ok")) if native_result else False
    condition_success = bool(native_result.get("condition_success")) if native_result else False
    if fixture["fixture_mode"] == "positive_control":
        return bool(gate_result.get("ok")) and native_ok and condition_success
    if fixture.get("negative_type") == "wrong_target_executable":
        return bool(gate_result.get("ok")) and native_attempted and native_ok and not condition_success and status["code"] == EXECUTABLE_CONDITION_FAILURE.code
    if fixture.get("negative_type") == "incompatible_target":
        return (not gate_result.get("ok")) and (not native_attempted) and status["code"] == TARGET_BINDING_FAILURE.code and status["subtype"] == "incompatible_target_affordance"
    return False


def _should_retry_native_execution(fixture: dict[str, Any], native_result: dict[str, Any] | None) -> bool:
    """Return true when a retry can only test native planner stochasticity."""

    if native_result is None:
        return False
    if fixture.get("fixture_mode") != "positive_control":
        return False
    if not native_result.get("ok"):
        return False
    if native_result.get("condition_success"):
        return False
    return True


def _summarize_native_attempt(attempt_index: int, native_result: dict[str, Any]) -> dict[str, Any]:
    """Build a compact per-attempt audit record."""

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

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_one_fixture(dataset_root: Path, fixture: dict[str, Any], trace_root: Path, timeout_sec: int) -> dict[str, Any]:
    """Run one frozen fixture through parser, phase binding, and optional native execution."""

    sample = _sample_from_case_id(dataset_root, fixture["source_case_id"])
    fixture_trace_root = ensure_dir(trace_root / fixture["fixture_id"])
    write_json(fixture_trace_root / "fixture.json", fixture)
    gate_result = _bind_fixture_plan(sample, fixture)
    write_json(fixture_trace_root / "gate1_phase_result.json", gate_result)
    native_attempted = False
    native_result: dict[str, Any] | None = None
    native_attempt_summaries: list[dict[str, Any]] = []
    selected_native_attempt = 0
    if gate_result.get("ok"):
        native_attempted = True
        native_command = {
            "task_name": sample.task_name,
            "env_config_path": str(sample.env_config_path),
            "executable_steps": gate_result["executable_steps"],
            "reset_wait_step": _reset_wait_step_for_task(sample.task_name),
            "native_seed": _native_seed_for_fixture(fixture),
            "visual_audit_dir": str(fixture_trace_root / "visual_audit"),
        }
        write_json(fixture_trace_root / "native_command.json", native_command)
        for attempt_index in range(1, MAX_NATIVE_RETRY_ATTEMPTS + 1):
            attempt_root = fixture_trace_root / f"execution_attempt_{attempt_index:02d}"
            attempt_result = run_native_execution(native_command, attempt_root, timeout_sec=timeout_sec)
            write_json(fixture_trace_root / f"native_result_attempt_{attempt_index:02d}.json", attempt_result)
            native_attempt_summaries.append(_summarize_native_attempt(attempt_index, attempt_result))
            native_result = attempt_result
            selected_native_attempt = attempt_index
            if not _should_retry_native_execution(fixture, attempt_result):
                break
        write_json(fixture_trace_root / "native_result.json", native_result)
    status = _status_from_native(gate_result, native_result)
    passed = _fixture_passed(fixture, gate_result, native_attempted, native_result, status)
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
    write_json(fixture_trace_root / "result_summary.json", row)
    return row


