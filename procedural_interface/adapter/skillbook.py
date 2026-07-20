"""Model-facing legacy Skill Book."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillSpec:
    """One VLM-facing skill definition."""

    # Skill name that the model must emit.
    name: str
    # Whether the JSON step must include a numeric target.
    requires_target: bool
    # Short model-facing meaning.
    meaning: str


# Keep this list explicit so prompt, validator, and audits share one source.
MODEL_FACING_SKILLS: tuple[SkillSpec, ...] = (
    SkillSpec("pick", True, "grasp the selected visible object"),
    SkillSpec("lift", False, "lift or move the currently held object/end effector as required by the current phase"),
    SkillSpec("moveto", True, "move toward the selected visible object/container/anchor"),
    SkillSpec("place", True, "place the held object at the selected support/container"),
    SkillSpec("pull", False, "perform the current phase pull motion"),
    SkillSpec("pour", False, "pour the currently held pourable object"),
    SkillSpec("press", True, "press the selected button/device/answer option"),
    SkillSpec("push", True, "perform the current phase push motion toward the selected target"),
    SkillSpec("wait", False, "wait for the task-defined duration"),
    SkillSpec("move_offset", False, "perform the task-defined fixed offset motion"),
    SkillSpec("open_gripper", False, "open the gripper in the current state"),
)


# Forbid role/phase fields in model output to keep the contract no-role.
FORBIDDEN_STEP_KEYS = {
    "role",
    "phase",
    "phase_id",
    "template",
    "pose",
    "offset",
    "quaternion",
    "xml_path",
    "component_id",
    "object_id",
}


def skill_names() -> set[str]:
    """Return allowed skill names."""

    # Compute from the canonical tuple to avoid duplicate maintenance.
    return {skill.name for skill in MODEL_FACING_SKILLS}


def skill_requires_target(skill_name: str) -> bool:
    """Return whether a skill requires a target."""

    # Search the small canonical list directly for clarity.
    for skill in MODEL_FACING_SKILLS:
        # Return the requirement when the skill matches.
        if skill.name == skill_name:
            return skill.requires_target
    # Unknown skills are handled by SkillBookValidator, so default false here.
    return False


def skillbook_as_dict() -> list[dict]:
    """Serialize the model-facing Skill Book."""

    # Produce stable dictionaries for prompt builders and Gate0 artifacts.
    return [
        {"skill": skill.name, "requires_target": skill.requires_target, "meaning": skill.meaning}
        for skill in MODEL_FACING_SKILLS
    ]
