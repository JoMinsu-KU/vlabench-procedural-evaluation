"""Phase binding for no-role, target-preserving legacy plans."""

from __future__ import annotations

from procedural_interface.evaluation.taxonomy import TARGET_BINDING_FAILURE, status_record
from procedural_interface.adapter.binding import bind_numeric_target
from procedural_interface.adapter.task_phase_templates import TaskTemplate


def bind_plan_to_phases(
    plan_steps: list[dict],
    env_config: dict,
    template: TaskTemplate,
    *,
    target_source: str = "vlm_output",
    ground_truth_target_used: bool = False,
) -> dict:
    """Bind VLM steps to deterministic task phases without reordering."""

    # Reject missing phases; the adapter must not insert them later.
    if len(plan_steps) < len(template.phases):
        return {
            "ok": False,
            "status": status_record(TARGET_BINDING_FAILURE, "missing_phase", f"expected {len(template.phases)} steps, got {len(plan_steps)}"),
            "phase_trace": [],
        }
    # Reject extra phases; the adapter must not skip official phases to align later steps.
    if len(plan_steps) > len(template.phases):
        return {
            "ok": False,
            "status": status_record(TARGET_BINDING_FAILURE, "extra_phase", f"expected {len(template.phases)} steps, got {len(plan_steps)}"),
            "phase_trace": [],
        }
    # Keep one trace row per model step.
    phase_trace: list[dict] = []
    # Iterate model steps and official-derived phases in lockstep.
    for zero_index, (step, phase) in enumerate(zip(plan_steps, template.phases)):
        # Store one-based VLM step index for readable audit.
        vlm_step_index = zero_index + 1
        # Read the model-selected skill exactly.
        vlm_skill = step["skill"]
        # Check skill/phase match without rewriting the model step.
        skill_matches_phase = vlm_skill == phase.expected_skill
        # Reject mismatch immediately because missing phase insertion is forbidden.
        if not skill_matches_phase:
            return {
                "ok": False,
                "status": status_record(TARGET_BINDING_FAILURE, "phase_skill_mismatch", f"phase {phase.phase_index} expected {phase.expected_skill}, got {vlm_skill}"),
                "phase_trace": phase_trace,
            }
        # Bind target only when the phase expects one.
        target_binding = None
        # Oracle-target provenance applies only to target-bearing phases.
        phase_ground_truth_target_used = bool(ground_truth_target_used and phase.requires_target)
        # Target-taking phases preserve the model-selected number.
        if phase.requires_target:
            # Read target after parser has guaranteed numeric type.
            target_binding = bind_numeric_target(env_config, int(step["target"]))
            # Propagate target binding failure without fallback.
            if not target_binding["ok"]:
                return {"ok": False, "status": target_binding["status"], "phase_trace": phase_trace}
            # Replace provenance only; the already-bound numeric target is unchanged.
            target_binding["target_source"] = target_source
            # Mark target-oracle use explicitly for ablation audit rows.
            target_binding["ground_truth_target_used"] = phase_ground_truth_target_used
        # Append complete provenance for this accepted phase.
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
                "gt_target_used": phase_ground_truth_target_used,
                "ground_truth_target_used": phase_ground_truth_target_used,
                "operation_sequence_used": False,
                "expert_sequence_called": False,
            }
        )
    # Return accepted phase-lowered plan trace.
    return {"ok": True, "status": status_record(TARGET_BINDING_FAILURE, "", ""), "phase_trace": phase_trace}
