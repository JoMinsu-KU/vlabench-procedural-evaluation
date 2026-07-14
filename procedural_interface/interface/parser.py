"""No-role JSON plan parser for VLABench procedural evaluation."""

from __future__ import annotations

import json
from typing import Any

from procedural_interface.evaluation.taxonomy import PARSER_SCHEMA_FAILURE, UNSUPPORTED_SKILL_FAILURE, status_record
from procedural_interface.adapter.skillbook import FORBIDDEN_STEP_KEYS, skill_names, skill_requires_target


def parse_plan_text(plan_text: str) -> dict[str, Any]:
    """Parse and validate a model-emitted skill plan."""

    try:
        payload = json.loads(plan_text)
    except json.JSONDecodeError as exc:
        return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "invalid_json", str(exc))}
    if not isinstance(payload, dict):
        return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "top_level_not_object", "top-level JSON must be an object")}
    if set(payload.keys()) != {"steps"}:
        return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "invalid_top_level_keys", f"keys={sorted(payload.keys())}")}
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "steps_not_nonempty_list", "steps must be a non-empty list")}
    normalized_steps: list[dict[str, Any]] = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "step_not_object", f"step={index}")}
        forbidden = sorted(set(step.keys()) & FORBIDDEN_STEP_KEYS)
        if forbidden:
            return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "forbidden_step_keys", f"step={index}, keys={forbidden}")}
        skill = step.get("skill")
        if not isinstance(skill, str):
            return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "missing_or_invalid_skill", f"step={index}")}
        if skill not in skill_names():
            return {"ok": False, "status": status_record(UNSUPPORTED_SKILL_FAILURE, "unsupported_skill", f"step={index}, skill={skill}")}
        needs_target = skill_requires_target(skill)
        if needs_target and "target" not in step:
            return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "missing_target", f"step={index}, skill={skill}")}
        if not needs_target and "target" in step:
            return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "unexpected_target", f"step={index}, skill={skill}")}
        if needs_target and not isinstance(step.get("target"), int):
            return {"ok": False, "status": status_record(PARSER_SCHEMA_FAILURE, "non_numeric_target", f"step={index}, target={step.get('target')}")}
        normalized = {"skill": skill}
        if needs_target:
            normalized["target"] = int(step["target"])
        normalized_steps.append(normalized)
    return {"ok": True, "steps": normalized_steps, "status": {"code": "", "label": "", "subtype": "", "detail": ""}}
