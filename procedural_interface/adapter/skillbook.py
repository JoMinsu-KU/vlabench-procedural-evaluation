"""Model-facing the procedural evaluation project Skill Book."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillSpec:
    """One VLM-facing skill definition."""

    name: str
    requires_target: bool
    meaning: str


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

    return {skill.name for skill in MODEL_FACING_SKILLS}


def skill_requires_target(skill_name: str) -> bool:
    """Return whether a skill requires a target."""

    for skill in MODEL_FACING_SKILLS:
        if skill.name == skill_name:
            return skill.requires_target
    return False


def skillbook_as_dict() -> list[dict]:
    """Serialize the model-facing Skill Book."""

    return [
        {"skill": skill.name, "requires_target": skill.requires_target, "meaning": skill.meaning}
        for skill in MODEL_FACING_SKILLS
    ]
