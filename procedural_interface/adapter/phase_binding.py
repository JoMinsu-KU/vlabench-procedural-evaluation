"""Phase binding for no-role, target-preserving the procedural evaluation project plans."""

from __future__ import annotations

from procedural_interface.evaluation.taxonomy import TARGET_BINDING_FAILURE, status_record
from procedural_interface.adapter.binding import bind_numeric_target
from procedural_interface.adapter.task_phase_templates import TaskTemplate


def bind_plan_to_phases(plan_steps: list[dict], env_config: dict, template: TaskTemplate) -> dict:
    """Bind VLM steps to deterministic task phases without reordering."""

    if len(plan_steps) < len(template.phases):
        return {
            "ok": False,
            "status": status_record(TARGET_BINDING_FAILURE, "missing_phase", f"expected {len(template.phases)} steps, got {len(plan_steps)}"),
            "phase_trace": [],
        }
    if len(plan_steps) > len(template.phases):
        return {
            "ok": False,
            "status": status_record(TARGET_BINDING_FAILURE, "extra_phase", f"expected {len(template.phases)} steps, got {len(plan_steps)}"),
            "phase_trace": [],
        }
    phase_trace: list[dict] = []
    for zero_index, (step, phase) in enumerate(zip(plan_steps, template.phases)):
        vlm_step_index = zero_index + 1
        vlm_skill = step["skill"]
        skill_matches_phase = vlm_skill == phase.expected_skill
        if not skill_matches_phase:
            return {
                "ok": False,
                "status": status_record(TARGET_BINDING_FAILURE, "phase_skill_mismatch", f"phase {phase.phase_index} expected {phase.expected_skill}, got {vlm_skill}"),
                "phase_trace": phase_trace,
            }
        target_binding = None
        if phase.requires_target:
            target_binding = bind_numeric_target(env_config, int(step["target"]))
            if not target_binding["ok"]:
                return {"ok": False, "status": target_binding["status"], "phase_trace": phase_trace}
        phase_trace.append(
            {
                "vlm_step_index": vlm_step_index,
                "phase_cursor_before": phase.phase_index,
                "phase_cursor_source": "deterministic_task_phase_cursor",
                "vlm_skill": vlm_skill,
                "vlm_target": step.get("target", ""),
                "expected_skill_for_phase": phase.expected_skill,
                "skill_matches_phase": True,
                "matched_phase_template": phase.template_name,
                "template_parameter_source": phase.parameter_source,
                "target_binding": target_binding,
                "sequence_reordered": False,
                "missing_phase_inserted": False,
                "gt_target_used": False,
                "ground_truth_target_used": False,
                "operation_sequence_used": False,
                "expert_sequence_called": False,
            }
        )
    return {"ok": True, "status": status_record(TARGET_BINDING_FAILURE, "", ""), "phase_trace": phase_trace}
