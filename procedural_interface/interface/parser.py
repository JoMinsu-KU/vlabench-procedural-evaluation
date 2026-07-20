"""No-role JSON plan parser for VLABench procedural evaluation."""

from __future__ import annotations

import json
from typing import Any

from procedural_interface.evaluation.taxonomy import PARSER_SCHEMA_FAILURE, UNSUPPORTED_SKILL_FAILURE, status_record
from procedural_interface.adapter.skillbook import FORBIDDEN_STEP_KEYS, skill_names, skill_requires_target


def parse_plan_text(plan_text: str) -> dict[str, Any]:
    """Parse and validate a model-emitted skill plan."""

    # Try strict JSON first; syntax repair is intentionally not implemented here.
    try:
        # Parse the raw model text as JSON.
        payload = json.loads(plan_text)
    except json.JSONDecodeError as exc:
        # Return a C2 failure for non-JSON output.
        return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "invalid_json", str(exc))}
    # Require a JSON object because the schema has one top-level key.
    if not isinstance(payload, dict):
        return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "top_level_not_object", "top-level JSON must be an object")}
    # Reject unknown top-level keys to keep the output contract narrow.
    if set(payload.keys()) != {"steps"}:
        return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "invalid_top_level_keys", f"keys={sorted(payload.keys())}")}
    # Read the ordered steps list.
    steps = payload.get("steps")
    # Require a non-empty list of steps.
    if not isinstance(steps, list) or not steps:
        return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "steps_not_nonempty_list", "steps must be a non-empty list")}
    # Validate each step independently before phase binding.
    normalized_steps: list[dict[str, Any]] = []
    # Iterate with one-based index for readable error messages.
    for index, step in enumerate(steps, start=1):
        # Each step must be an object.
        if not isinstance(step, dict):
            return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "step_not_object", f"step={index}")}
        # Detect role/phase/template leakage before accepting the step.
        forbidden = sorted(set(step.keys()) & FORBIDDEN_STEP_KEYS)
        # The main experiment does not allow role or phase hints from the model.
        if forbidden:
            return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "forbidden_step_keys", f"step={index}, keys={forbidden}")}
        # Require skill field.
        skill = step.get("skill")
        # Skill must be a string.
        if not isinstance(skill, str):
            return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "missing_or_invalid_skill", f"step={index}")}
        # Unsupported skill is C4 rather than generic JSON failure.
        if skill not in skill_names():
            return {"ok": False, "status": status_record(UNSUPPORTED_SKILL_FAILURE, "unsupported_skill", f"step={index}, skill={skill}")}
        # Determine whether this skill requires a target.
        needs_target = skill_requires_target(skill)
        # Target-taking skills must include target.
        if needs_target and "target" not in step:
            return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "missing_target", f"step={index}, skill={skill}")}
        # No-target skills must not include target in the main schema.
        if not needs_target and "target" in step:
            return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "unexpected_target", f"step={index}, skill={skill}")}
        # Target-taking skills must use numeric identifiers.
        if needs_target and not isinstance(step.get("target"), int):
            return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "non_numeric_target", f"step={index}, target={step.get('target')}")}
        # Preserve only schema-approved fields.
        normalized = {"skill": skill}
        # Preserve model-selected target when present.
        if needs_target:
            normalized["target"] = int(step["target"])
        # Append the normalized step without adding role or phase.
        normalized_steps.append(normalized)
    # Return normalized plan for phase binding.
    return {"ok": True, "steps": normalized_steps, "status": {"code": "", "label": "", "subtype": "", "detail": ""}}
