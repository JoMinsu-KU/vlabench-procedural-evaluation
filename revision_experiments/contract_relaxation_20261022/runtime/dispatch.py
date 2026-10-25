"""Separate relaxed dispatcher; original parser, binder and native worker stay intact."""

import json
import sys

from common import PROJECT, project_steps

sys.path.insert(0, str(PROJECT))
from procedural_interface.adapter.binding import bind_numeric_target
from procedural_interface.adapter.skillbook import skill_requires_target
from procedural_interface.evaluation.runner import _check_affordance
from procedural_interface.interface.parser import parse_plan_text


def lower(task_name, original_steps, env_config):
    parsed = parse_plan_text(json.dumps({"steps": original_steps}))
    if not parsed.get("ok") or parsed.get("steps") != original_steps:
        return {"category": "retained_validation_rejection", "detail": parsed.get("status"), "steps": []}
    affordance = _check_affordance(original_steps, env_config)
    if not affordance.get("ok"):
        return {"category": "retained_validation_rejection", "detail": affordance.get("status"), "steps": []}
    skills = [step["skill"] for step in original_steps]
    # Flower lift has two distinct meanings selected by canonical phase metadata.
    # Guessing a phase would change the interpretation of the archived plan.
    flower = "insert_flower" in task_name or "insert_bloom_flower" in task_name
    if flower and "lift" in skills:
        return {"category": "unsupported_dispatch", "detail": "flower_lift_requires_unobserved_phase_role", "steps": []}
    if "move_offset" in skills:
        return {"category": "unsupported_dispatch", "detail": "move_offset_has_no_native_template_in_noncomplex_tasks", "steps": []}
    if task_name in {"book_rearrange", "take_chemistry_experiment", "hammer_nail_and_hang_picture", "cook_dishes", "texas_holdem"}:
        return {"category": "unsupported_dispatch", "detail": "complex_task_outside_matched_subset", "steps": []}
    executable = []
    for index, step in enumerate(original_steps):
        lowered = {"index": index, "skill": step["skill"],
                   "matched_phase_template": f"relaxed_diagnostic.step_{index + 1}.{step['skill']}",
                   "phase_cursor_source": "archived_model_order_no_reference_phase_assignment",
                   "sequence_reordered": False, "missing_phase_inserted": False,
                   "gt_target_used": False, "operation_sequence_used": False, "expert_sequence_called": False}
        if skill_requires_target(step["skill"]):
            bound = bind_numeric_target(env_config, step["target"])
            if not bound.get("ok"):
                return {"category": "retained_validation_rejection", "detail": bound.get("status"), "steps": []}
            lowered["target"] = {"number": step["target"], "entity_name": bound["entity_name"],
                                 "entity_class": bound["entity_class"], "target_source": "archived_vlm_output"}
        executable.append(lowered)
    if project_steps(executable) != original_steps:
        raise RuntimeError("Dispatcher changed a supplied skill, order or numeric target")
    return {"category": "runnable", "detail": "only_reference_phase_equality_bypassed", "steps": executable}


def audit(result, expected):
    trace = result.get("trace", [])
    observed = project_steps(trace)
    wanted = project_steps(expected)
    identity = observed == wanted
    prefix = observed == wanted[:len(observed)]
    stage_exception = bool(result.get("stage_exception_policy", {}).get("stage_exception_caught"))
    complete = (bool(result.get("execution_completed")) and identity and not stage_exception
                and not result.get("native_timeout") and not result.get("infrastructure_error"))
    flags = [step.get("stage_success") for step in trace]
    all_stages = bool(expected) and complete and len(flags) == len(expected) and all(flag is True for flag in flags)
    goal = bool(result.get("condition_success"))
    return {"raw_final_goal": goal, "complete_without_exception": complete,
            "full_trace_identity": identity, "trace_is_original_prefix": prefix,
            "observed_steps": observed, "expected_steps": wanted,
            "stage_success_flags": flags, "all_stages_success": all_stages,
            "stage_false_count": sum(flag is False for flag in flags),
            "goal_after_complete_execution": complete and goal,
            "goal_and_all_stages_success": all_stages and goal,
            "goal_with_stage_failure": complete and goal and any(flag is False for flag in flags),
            "stage_exception": stage_exception,
            "native_timeout": bool(result.get("native_timeout")),
            "infrastructure_error": bool(result.get("infrastructure_error")),
            "pre_action_load_failure": bool(result.get("native_exception") and "trace" not in result),
            "native_exception": result.get("exception", ""),
            "stage_exception_detail": result.get("stage_exception_policy", {}).get("exception", "")}
