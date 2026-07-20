"""Oracle ablation contracts for the procedural VLM evaluation.

The phase-oracle condition fixes only the benchmark-derived phase order and
asks the model to select numeric targets.  The target-oracle condition keeps
phase/order generation with the model and inserts case-specific reference
targets only after the emitted skill sequence satisfies the same procedural
contract used by Experiment B.
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from procedural_interface.adapter.skillbook import skill_names
from procedural_interface.adapter.task_phase_templates import TASK_TEMPLATES, TaskTemplate
from procedural_interface.evaluation.taxonomy import (
    PARSER_SCHEMA_FAILURE,
    TARGET_BINDING_FAILURE,
    UNSUPPORTED_SKILL_FAILURE,
    status_record,
)


FULL_MODE = "full"
PHASE_ORACLE_MODE = "phase_oracle"
TARGET_ORACLE_MODE = "target_oracle"
ABLATION_MODES = (FULL_MODE, PHASE_ORACLE_MODE, TARGET_ORACLE_MODE)


def _component_index(env_config: dict[str, Any], name: str) -> int:
    """Resolve one exact component name to its numbered-mask index."""

    # The numbered mask and component list share this zero-based ordering.
    components = env_config.get("task", {}).get("components", [])
    # Search exact names only; fuzzy matching would silently repair oracle data.
    for index, component in enumerate(components):
        # Return the public numeric identifier for the exact component.
        if isinstance(component, dict) and component.get("name") == name:
            return index
    # A missing reference target is a dataset/registry error, not model failure.
    raise ValueError(f"reference component not found: {name}")


def _first_condition_entity(task: dict[str, Any]) -> str:
    """Read the reference manipulated entity from benchmark conditions."""

    # These list-valued conditions cover the retained non-complex task families.
    for condition_name in ("contain", "not_contain", "is_grasped", "lift"):
        # Read the condition entity list without changing its order.
        entities = task.get("conditions", {}).get(condition_name, {}).get("entities")
        # The first entity is the benchmark reference manipulated object.
        if isinstance(entities, list) and entities and isinstance(entities[0], str):
            return entities[0]
    # Pour/above conditions encode the manipulated entity as a scalar field.
    for condition_name in ("pour", "above"):
        # Read the scalar entity exactly.
        target_entity = task.get("conditions", {}).get(condition_name, {}).get("target_entity")
        # Return only a valid string target.
        if isinstance(target_entity, str):
            return target_entity
    # Refuse to guess when the benchmark condition does not expose an entity.
    raise ValueError("cannot resolve reference manipulated entity from conditions")


def _condition_container(task: dict[str, Any]) -> str:
    """Read the reference destination/container from benchmark conditions."""

    # Containment tasks expose the destination under container.
    for condition_name in ("contain", "not_contain"):
        # Read the destination name exactly.
        container = task.get("conditions", {}).get(condition_name, {}).get("container")
        # Return only a valid string destination.
        if isinstance(container, str):
            return container
    # Above-style tasks expose the destination under platform.
    platform = task.get("conditions", {}).get("above", {}).get("platform")
    # Return a valid platform string.
    if isinstance(platform, str):
        return platform
    # Missing destination evidence invalidates the oracle fixture.
    raise ValueError("cannot resolve reference destination from conditions")


def _press_button(task: dict[str, Any]) -> str:
    """Read the reference answer button from PhysicsLaw conditions."""

    # PhysicsLaw QA tasks store the correct answer button explicitly.
    button = task.get("conditions", {}).get("press_button", {}).get("target_button")
    # Return only a valid button name.
    if isinstance(button, str):
        return button
    # Refuse to infer an answer from instruction text or component order.
    raise ValueError("cannot resolve reference press button from conditions")


def build_reference_steps(task_name: str, env_config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[int, str]]:
    """Build the frozen reference procedure used only by target-oracle evaluation."""

    # The retained 2,133-case manifest must have an explicit task registry entry.
    template = TASK_TEMPLATES[task_name]
    # Read the benchmark task object once.
    task = env_config.get("task", {})
    # Resolve reference names only when the task contract needs them.
    entity_name = _first_condition_entity(task) if any(phase.expected_skill in {"pick", "push"} for phase in template.phases) else None
    # Resolve a destination only for target-taking destination phases.
    container_name = _condition_container(task) if any(phase.expected_skill in {"place", "moveto"} for phase in template.phases) else None
    # Resolve an answer button only for press phases.
    button_name = _press_button(task) if any(phase.expected_skill == "press" for phase in template.phases) else None
    # Keep the full executable step sequence in reference phase order.
    steps: list[dict[str, Any]] = []
    # Keep human-readable target provenance separate from execution fields.
    targets: dict[int, str] = {}
    # Materialize one step per frozen benchmark-derived phase.
    for phase in template.phases:
        # Copy the registered skill without deriving a new action.
        skill = phase.expected_skill
        # Pick and push operate on the reference manipulated entity.
        if skill in {"pick", "push"}:
            if entity_name is None:
                raise ValueError(f"missing reference entity for {task_name}:{skill}")
            target_number = _component_index(env_config, entity_name)
            steps.append({"skill": skill, "target": target_number})
            targets[target_number] = entity_name
        # Place and moveto operate on the reference destination.
        elif skill in {"place", "moveto"}:
            if container_name is None:
                raise ValueError(f"missing reference destination for {task_name}:{skill}")
            target_number = _component_index(env_config, container_name)
            steps.append({"skill": skill, "target": target_number})
            targets[target_number] = container_name
        # Press operates on the benchmark reference answer button.
        elif skill == "press":
            if button_name is None:
                raise ValueError(f"missing reference button for {task_name}")
            target_number = _component_index(env_config, button_name)
            steps.append({"skill": skill, "target": target_number})
            targets[target_number] = button_name
        # Targetless phases retain only their skill names.
        else:
            steps.append({"skill": skill})
    # Return numeric execution data and audit-only names separately.
    return steps, targets


def serialize_phase_skeleton(task_name: str) -> list[dict[str, Any]]:
    """Serialize the phase-oracle skeleton without target identities."""

    # Read the frozen task template.
    template = TASK_TEMPLATES[task_name]
    # Expose phase index, skill, and target requirement only.
    return [
        {
            "phase_index": phase.phase_index,
            "skill": phase.expected_skill,
            "requires_target": phase.requires_target,
        }
        for phase in template.phases
    ]


def build_oracle_prompt(*, mode: str, task_name: str, skillbook_header: str) -> str:
    """Build an explicit oracle prompt while retaining the B skill descriptions."""

    # Reject accidental mode typos before any model request is sent.
    if mode not in {PHASE_ORACLE_MODE, TARGET_ORACLE_MODE}:
        raise ValueError(f"oracle prompt requested for unsupported mode: {mode}")
    # The common header is copied from Experiment B through its first separator.
    common_input = """---

You will receive the following input:

1. **Image input**: Two images
   - The first image shows four different perspectives of objects (without labels).
   - The second image contains the same four perspectives of objects, but each view is labeled with a number (representing each object's identifier).

2. **Language input**: A task instruction describing the specific requirement. All directional references in the language are relative to the robot arm as the central origin.

Return one JSON object only. Do not use Markdown code fences or explanatory text.
"""
    # Phase oracle reveals only the frozen phase/order structure.
    if mode == PHASE_ORACLE_MODE:
        skeleton = json.dumps(serialize_phase_skeleton(task_name), ensure_ascii=False)
        target_count = sum(1 for phase in TASK_TEMPLATES[task_name].phases if phase.requires_target)
        contract = f"""

### Phase-oracle condition

The benchmark-derived reference phase order is fixed below. Do not predict, remove, add, or reorder skills. Select only the numbered-image targets for the target-taking phases, in their listed order.

Reference phase skeleton:
{skeleton}

Exactly {target_count} target values are required. Output format:
{{"targets": [Target Number 1, Target Number 2]}}

The first value corresponds to the first phase whose `requires_target` value is true, the second value to the next such phase, and so on. Do not output skill names or phase metadata.
"""
    # Target oracle keeps the complete skill/order decision with the model.
    else:
        contract = """

### Target-oracle condition

Reference target identities are supplied by the evaluator after parsing and are not part of your output. Generate only the complete ordered procedural skill sequence required for the instruction. Do not output targets, phase numbers, roles, or geometric parameters.

As in the Full-B prompt, select the complete sequence from the following supported sub-skill patterns:
- ["pick", "place"]
- ["pick", "lift"]
- ["pick", "pull"]
- ["pick", "lift", "place"]
- ["pick", "lift", "moveto", "lift"]
- ["pick", "lift", "moveto", "pour"]
- ["pick", "lift", "moveto", "push"]
- ["pick", "lift", "moveto", "push", "pull"]
- ["pick", "lift", "moveto", "pour", "wait", "pull", "pour", "moveto", "move_offset", "open_gripper"]
- ["press"]

If the object must be picked up and displayed, use ["pick", "lift"]. If it must be placed directly into a specified location, use ["pick", "place"]. If it must be inserted into a target location, use ["pick", "lift", "moveto", "lift"].

Output format:
{"skills": ["Skill Name 1", "Skill Name 2"]}

Every item must be one of the 11 available skill names. The evaluator will not add, delete, reorder, or replace any model-emitted skill.
"""
    # Concatenate the unchanged skill descriptions with the oracle-specific contract.
    return skillbook_header.rstrip() + "\n\n" + common_input.rstrip() + contract


def _load_object(json_text: str) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    """Load one strict JSON object without syntax repair."""

    # Parse the extracted object candidate exactly.
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        return None, status_record(PARSER_SCHEMA_FAILURE, "invalid_json", str(exc))
    # Oracle schemas require one top-level object.
    if not isinstance(payload, dict):
        return None, status_record(PARSER_SCHEMA_FAILURE, "top_level_not_object", "top-level JSON must be an object")
    # Return the object for condition-specific validation.
    return payload, None


def resolve_phase_oracle_output(json_text: str, task_name: str) -> dict[str, Any]:
    """Resolve model-selected targets into the fixed reference phase skeleton."""

    # Load the strict JSON object first.
    payload, error = _load_object(json_text)
    # Preserve parser failure without constructing steps.
    if error is not None:
        return {"ok": False, "model_parse_ok": False, "strict_parse_ok": False, "type_normalized_parse_ok": False, "status": error}
    assert payload is not None
    # Reject extra keys that could smuggle skill/order decisions into this condition.
    if set(payload.keys()) != {"targets"}:
        status = status_record(PARSER_SCHEMA_FAILURE, "invalid_top_level_keys", f"keys={sorted(payload.keys())}")
        return {"ok": False, "model_parse_ok": False, "strict_parse_ok": False, "type_normalized_parse_ok": False, "status": status}
    # Require an ordered target list.
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list):
        status = status_record(PARSER_SCHEMA_FAILURE, "targets_not_list", "targets must be a list")
        return {"ok": False, "model_parse_ok": False, "strict_parse_ok": False, "type_normalized_parse_ok": False, "status": status}
    # Count only phases whose targets remain a model responsibility.
    template = TASK_TEMPLATES[task_name]
    target_phases = [phase for phase in template.phases if phase.requires_target]
    # Missing or extra target slots are contract failures; no slot is inferred.
    if len(raw_targets) != len(target_phases):
        status = status_record(PARSER_SCHEMA_FAILURE, "target_slot_count_mismatch", f"expected {len(target_phases)} targets, got {len(raw_targets)}")
        return {"ok": False, "model_parse_ok": False, "strict_parse_ok": False, "type_normalized_parse_ok": False, "status": status, "raw_targets": raw_targets}
    # Strict mode accepts only JSON integers and excludes booleans.
    strict_ok = all(isinstance(value, int) and not isinstance(value, bool) for value in raw_targets)
    # Apply the same narrow integer-string policy used by Experiment B.
    normalized_targets: list[Any] = []
    cast_applied = False
    for value in raw_targets:
        if isinstance(value, str) and re.fullmatch(r"\d+", value.strip()):
            normalized_targets.append(int(value.strip()))
            cast_applied = True
        else:
            normalized_targets.append(value)
    # Reject values that remain non-integer after deterministic casting.
    normalized_ok = all(isinstance(value, int) and not isinstance(value, bool) for value in normalized_targets)
    if not normalized_ok:
        status = status_record(PARSER_SCHEMA_FAILURE, "non_numeric_target", f"targets={raw_targets}")
        return {"ok": False, "model_parse_ok": False, "strict_parse_ok": strict_ok, "type_normalized_parse_ok": False, "status": status, "raw_targets": raw_targets, "normalized_targets": normalized_targets}
    # Materialize the fixed phase skeleton using only model-selected targets.
    steps: list[dict[str, Any]] = []
    target_cursor = 0
    for phase in template.phases:
        step: dict[str, Any] = {"skill": phase.expected_skill}
        if phase.requires_target:
            step["target"] = int(normalized_targets[target_cursor])
            target_cursor += 1
        steps.append(step)
    # Return complete provenance for execution and later audit.
    return {
        "ok": True,
        "model_parse_ok": True,
        "strict_parse_ok": strict_ok,
        "type_normalized_parse_ok": True,
        "target_numeric_cast_applied": cast_applied,
        "semantic_repair_applied": False,
        "raw_targets": raw_targets,
        "normalized_targets": normalized_targets,
        "steps": steps,
        "status": {"code": "", "label": "", "subtype": "", "detail": ""},
    }


def _sequence_status(model_skills: list[str], template: TaskTemplate) -> dict[str, str] | None:
    """Check model phase/order without inserting, deleting, or reordering skills."""

    # Missing phases remain a model sequence failure.
    if len(model_skills) < len(template.phases):
        return status_record(TARGET_BINDING_FAILURE, "missing_phase", f"expected {len(template.phases)} skills, got {len(model_skills)}")
    # Extra phases remain a model sequence failure.
    if len(model_skills) > len(template.phases):
        return status_record(TARGET_BINDING_FAILURE, "extra_phase", f"expected {len(template.phases)} skills, got {len(model_skills)}")
    # Compare each skill in the model-emitted order.
    for index, (model_skill, phase) in enumerate(zip(model_skills, template.phases), start=1):
        if model_skill != phase.expected_skill:
            return status_record(TARGET_BINDING_FAILURE, "phase_skill_mismatch", f"phase {index} expected {phase.expected_skill}, got {model_skill}")
    # None means the model sequence satisfies the reference contract.
    return None


def resolve_target_oracle_output(
    json_text: str,
    task_name: str,
    env_config_source: dict[str, Any] | Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Validate model skill/order and attach frozen case-specific reference targets."""

    # Load the strict JSON object first.
    payload, error = _load_object(json_text)
    # Preserve parser failure without consulting oracle targets.
    if error is not None:
        return {"ok": False, "model_parse_ok": False, "strict_parse_ok": False, "type_normalized_parse_ok": False, "oracle_target_read": False, "status": error}
    assert payload is not None
    # Reject target fields or any other top-level information in model output.
    if set(payload.keys()) != {"skills"}:
        status = status_record(PARSER_SCHEMA_FAILURE, "invalid_top_level_keys", f"keys={sorted(payload.keys())}")
        return {"ok": False, "model_parse_ok": False, "strict_parse_ok": False, "type_normalized_parse_ok": False, "oracle_target_read": False, "status": status}
    # Require an ordered skill-name list.
    model_skills = payload.get("skills")
    if not isinstance(model_skills, list) or not model_skills:
        status = status_record(PARSER_SCHEMA_FAILURE, "skills_not_nonempty_list", "skills must be a non-empty list")
        return {"ok": False, "model_parse_ok": False, "strict_parse_ok": False, "type_normalized_parse_ok": False, "oracle_target_read": False, "status": status}
    # Validate string type before checking the vocabulary.
    if any(not isinstance(skill, str) for skill in model_skills):
        status = status_record(PARSER_SCHEMA_FAILURE, "non_string_skill", f"skills={model_skills}")
        return {"ok": False, "model_parse_ok": False, "strict_parse_ok": False, "type_normalized_parse_ok": False, "oracle_target_read": False, "status": status}
    # Reject unsupported skills as a model vocabulary failure.
    unsupported = [skill for skill in model_skills if skill not in skill_names()]
    if unsupported:
        status = status_record(UNSUPPORTED_SKILL_FAILURE, "unsupported_skill", f"skills={unsupported}")
        return {"ok": False, "model_parse_ok": False, "strict_parse_ok": False, "type_normalized_parse_ok": False, "oracle_target_read": False, "status": status}
    # Validate sequence/order before reading case-specific oracle targets.
    sequence_error = _sequence_status(model_skills, TASK_TEMPLATES[task_name])
    if sequence_error is not None:
        return {
            "ok": False,
            "model_parse_ok": True,
            "strict_parse_ok": True,
            "type_normalized_parse_ok": True,
            "oracle_target_read": False,
            "model_skills": model_skills,
            "status": sequence_error,
        }
    # Only a contract-valid model sequence may trigger the env_config/condition read.
    env_config = env_config_source() if callable(env_config_source) else env_config_source
    # Attach case-specific reference targets after the sequence gate.
    steps, human_readable_targets = build_reference_steps(task_name, env_config)
    # Assert the attached skills remain byte-for-byte equal to model order.
    if [step["skill"] for step in steps] != model_skills:
        raise AssertionError("target-oracle attachment changed the model skill sequence")
    # Return resolved steps with explicit oracle provenance.
    return {
        "ok": True,
        "model_parse_ok": True,
        "strict_parse_ok": True,
        "type_normalized_parse_ok": True,
        "target_numeric_cast_applied": False,
        "semantic_repair_applied": False,
        "oracle_target_read": True,
        "model_skills": model_skills,
        "steps": steps,
        "human_readable_targets": human_readable_targets,
        "status": {"code": "", "label": "", "subtype": "", "detail": ""},
    }
