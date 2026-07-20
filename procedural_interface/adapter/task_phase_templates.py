"""Task-specific phase-aware template registry.

This module contains the Gate0/Gate1 phase inventory derived from official
VLABench executor scripts.  It stores skill order and template identity only;
numeric motion parameters are attached by the execution bridge in the next
implementation gate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseTemplate:
    """One expected phase in a task-specific execution template."""

    # One-based phase index for human-readable audit logs.
    phase_index: int
    # Skill name expected from the VLM at this phase.
    expected_skill: str
    # Whether this phase expects a numeric target in model output.
    requires_target: bool
    # Role-like template name used internally only; the model never emits it.
    template_name: str
    # Official VLABench class/method source used to derive this phase.
    parameter_source: str


@dataclass(frozen=True)
class TaskTemplate:
    """Complete phase template sequence for one public task."""

    # Public VLABench task directory name.
    task_name: str
    # Registered official class name.
    official_class: str
    # Official module path under hierarchical_tasks.
    official_module: str
    # Ordered phase templates.
    phases: tuple[PhaseTemplate, ...]


def _phases(task_name: str, official_class: str, official_module: str, skills: list[str]) -> TaskTemplate:
    """Build phase templates from an official skill sequence."""

    # The source string ties every phase to the official script without calling it.
    source = f"{official_module}:{official_class}.get_expert_skill_sequence"
    # Convert the sequence into immutable phase records.
    phases = tuple(
        PhaseTemplate(
            phase_index=index,
            expected_skill=skill,
            requires_target=skill in {"pick", "moveto", "place", "press", "push"},
            template_name=f"{task_name}.phase_{index}.{skill}",
            parameter_source=source,
        )
        for index, skill in enumerate(skills, start=1)
    )
    # Return a task template with official provenance.
    return TaskTemplate(task_name, official_class, official_module, phases)


# Public 4500 task templates resolved from official VLABench registered classes.
TASK_TEMPLATES: dict[str, TaskTemplate] = {
    # Add condiment variants share AddCondimentTask behavior.
    "add_condiment": _phases("add_condiment", "AddCondimentTask", "primitive/add_condiment_series.py", ["pick", "lift", "moveto", "pour"]),
    "add_condiment_common_sense": _phases("add_condiment_common_sense", "AddCondimentCommonSenseTask", "primitive/add_condiment_series.py", ["pick", "lift", "moveto", "pour"]),
    "add_condiment_semantic": _phases("add_condiment_semantic", "AddCondimentSemanticTask", "primitive/add_condiment_series.py", ["pick", "lift", "moveto", "pour"]),
    "add_condiment_spatial": _phases("add_condiment_spatial", "AddCondimentSpatialTask", "primitive/add_condiment_series.py", ["pick", "lift", "moveto", "pour"]),
    # Insert flower variants share the executor-level pick/lift/moveto/lift sequence.
    "insert_flower": _phases("insert_flower", "InsertFlowerTask", "primitive/insert_flower_series.py", ["pick", "lift", "moveto", "lift"]),
    "insert_flower_common_sense": _phases("insert_flower_common_sense", "InsertFlowerCommonSenseTask", "primitive/insert_flower_series.py", ["pick", "lift", "moveto", "lift"]),
    "insert_flower_semantic": _phases("insert_flower_semantic", "InsertFlowerSemanticTask", "primitive/insert_flower_series.py", ["pick", "lift", "moveto", "lift"]),
    "insert_bloom_flower": _phases("insert_bloom_flower", "InsertBloomFlowerTask", "primitive/insert_flower_series.py", ["pick", "lift", "moveto", "lift"]),
    # Select family templates.
    "select_billiards": _phases("select_billiards", "SelectBilliardsTask", "primitive/select_billiards_series.py", ["pick", "place"]),
    "select_billiards_common_sense": _phases("select_billiards_common_sense", "SelectBilliardsCommonSenseTask", "primitive/select_billiards_series.py", ["pick", "place"]),
    "select_billiards_semantic": _phases("select_billiards_semantic", "SelectBilliardsSemanticTask", "primitive/select_billiards_series.py", ["pick", "place"]),
    "select_billiards_spatial": _phases("select_billiards_spatial", "SelectBilliardsSpatialTask", "primitive/select_billiards_series.py", ["pick", "place"]),
    "select_book": _phases("select_book", "SelectBookTask", "primitive/select_book_series.py", ["pick", "pull"]),
    "select_book_semantic": _phases("select_book_semantic", "SelectBookSemanticTask", "primitive/select_book_series.py", ["pick", "pull"]),
    "select_book_spatial": _phases("select_book_spatial", "SelectBookSpatialTask", "primitive/select_book_series.py", ["pick", "pull"]),
    "select_chemistry_tube": _phases("select_chemistry_tube", "SelectChemistryTubeTask", "primitive/select_chemistry_tube_series.py", ["pick", "lift"]),
    "select_chemistry_tube_common_sense": _phases("select_chemistry_tube_common_sense", "SelectChemistryTubeCommonSenseTask", "primitive/select_chemistry_tube_series.py", ["pick", "lift"]),
    "select_chemistry_tube_semantic": _phases("select_chemistry_tube_semantic", "SelectChemistryTubeSemanticTask", "primitive/select_chemistry_tube_series.py", ["pick", "lift"]),
    "select_chemistry_tube_spatial": _phases("select_chemistry_tube_spatial", "SelectChemistryTubeSpatialTask", "primitive/select_chemistry_tube_series.py", ["pick", "lift"]),
    "select_drink": _phases("select_drink", "SelectDrinkTask", "primitive/select_drink_series.py", ["pick", "lift", "pull"]),
    "select_drink_common_sense": _phases("select_drink_common_sense", "SelectDrinkCommonSenseTask", "primitive/select_drink_series.py", ["pick", "lift", "pull"]),
    "select_drink_semantic": _phases("select_drink_semantic", "SelectDrinkSemanticTask", "primitive/select_drink_series.py", ["pick", "lift", "pull"]),
    "select_fruit": _phases("select_fruit", "SelectFruitTask", "primitive/select_fruit_series.py", ["pick", "place"]),
    "select_fruit_common_sense": _phases("select_fruit_common_sense", "SelectFruitCommonSenseTask", "primitive/select_fruit_series.py", ["pick", "place"]),
    "select_fruit_semantic": _phases("select_fruit_semantic", "SelectFruitSemanticTask", "primitive/select_fruit_series.py", ["pick", "place"]),
    "select_fruit_spatial": _phases("select_fruit_spatial", "SelectFruitSpatialTask", "primitive/select_fruit_series.py", ["pick", "place"]),
    "select_nth_largest_poker": _phases("select_nth_largest_poker", "SelectNthLargestPokerTask", "primitive/select_poker_series.py", ["pick", "lift"]),
    "select_poker": _phases("select_poker", "SelectPokerTask", "primitive/select_poker_series.py", ["pick", "lift"]),
    "select_poker_semantic": _phases("select_poker_semantic", "SelectPokerSemanticTask", "primitive/select_poker_series.py", ["pick", "lift"]),
    "select_poker_spatial": _phases("select_poker_spatial", "SelectpokerSpatialTask", "primitive/select_poker_series.py", ["pick", "lift"]),
    "select_toy": _phases("select_toy", "SelectToyTask", "primitive/select_toy_series.py", ["pick", "lift", "place"]),
    "select_toy_common_sense": _phases("select_toy_common_sense", "SelectToyCommonSense", "primitive/select_toy_series.py", ["pick", "lift", "place"]),
    "select_toy_semantic": _phases("select_toy_semantic", "SelectToySemantic", "primitive/select_toy_series.py", ["pick", "lift", "place"]),
    "select_toy_spatial": _phases("select_toy_spatial", "SelectToySpatialTask", "primitive/select_toy_series.py", ["pick", "lift", "place"]),
    # Complex task templates.
    "book_rearrange": _phases("book_rearrange", "BookRearrangeTask", "composite/rearrange_book_series.py", ["pick", "moveto", "push", "pull", "pick", "moveto", "push", "pull", "pick", "moveto", "push", "pull"]),
    "cook_dishes": _phases("cook_dishes", "CookDishesTask", "composite/cook_dishes.py", ["pick", "lift", "place"]),
    "hammer_nail_and_hang_picture": _phases("hammer_nail_and_hang_picture", "HammerNailandHangPictureTask", "composite/hammer_nail_series.py", ["pick", "lift", "pull", "moveto", "push", "place", "lift", "pick", "lift", "moveto", "push"]),
    "take_chemistry_experiment": _phases("take_chemistry_experiment", "TakeChemistryExperimentTask", "composite/take_chemistry_experiment.py", ["pick", "lift", "moveto", "pour", "wait", "pull", "pour", "moveto", "move_offset", "open_gripper", "pick", "lift", "moveto", "pour", "wait", "pull", "pour", "moveto", "move_offset", "open_gripper"]),
    "texas_holdem": _phases("texas_holdem", "PokerPlayTask", "composite/play_poker_series.py", ["pick", "lift", "place"]),
    # PhysicsLaw answer tasks are button-press tasks.
    "density_qa": _phases("density_qa", "DensityQATask", "primitive/physical_series/density_qa.py", ["press"]),
    "friction_qa": _phases("friction_qa", "FrictionQATask", "primitive/physical_series/friction_qa.py", ["press"]),
    "magnetism_qa": _phases("magnetism_qa", "MagnetismQATask", "primitive/physical_series/magnetism_qa.py", ["press"]),
    "reflection_qa": _phases("reflection_qa", "ReflectionQATask", "primitive/physical_series/reflection_qa.py", ["press"]),
    "speed_of_sound_qa": _phases("speed_of_sound_qa", "SoundSpeedQATask", "primitive/physical_series/sound_speed_qa.py", ["press"]),
    "thermal_expansion_qa": _phases("thermal_expansion_qa", "ThermalExpansionQATask", "primitive/physical_series/thermal_expansion_qa.py", ["press"]),
}


def get_task_template(task_name: str) -> TaskTemplate | None:
    """Return a public task template when available."""

    # Look up by public task directory name exactly.
    return TASK_TEMPLATES.get(task_name)


def serialize_task_template(template: TaskTemplate) -> dict:
    """Serialize a task template for Gate0 artifacts."""

    # Convert dataclasses manually to keep field names explicit and stable.
    return {
        "task_name": template.task_name,
        "official_class": template.official_class,
        "official_module": template.official_module,
        "phase_count": len(template.phases),
        "phases": [
            {
                "phase_index": phase.phase_index,
                "expected_skill": phase.expected_skill,
                "requires_target": phase.requires_target,
                "template_name": phase.template_name,
                "parameter_source": phase.parameter_source,
            }
            for phase in template.phases
        ],
    }


def serialize_registry() -> list[dict]:
    """Serialize the entire public 4500 template registry."""

    # Sort by task name for deterministic JSON.
    return [serialize_task_template(TASK_TEMPLATES[name]) for name in sorted(TASK_TEMPLATES)]
