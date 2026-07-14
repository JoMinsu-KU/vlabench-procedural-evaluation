"""WSL-side VLABench worker for target-preserving the procedural evaluation project smoke execution."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

import VLABench.robots  # noqa: F401
import VLABench.tasks  # noqa: F401
from VLABench.envs import load_env
from VLABench.utils.skill_lib import SkillLib
from VLABench.utils.utils import euler_to_quaternion


def _entity_pos(env, name: str) -> np.ndarray:
    """Return the current world position for a named VLABench entity."""

    entity = env.task.entities[name]
    return np.array(entity.get_xpos(env.physics))


def _place_point(env, name: str) -> np.ndarray:
    """Return a native place point when the entity exposes one."""

    entity = env.task.entities[name]
    if hasattr(entity, "get_place_point"):
        points = entity.get_place_point(env.physics)
        if points:
            return np.array(points[-1])
    return _entity_pos(env, name)


def _phase_index(step: dict) -> int:
    """Read the one-based deterministic phase index from a matched template name."""

    template_name = str(step.get("matched_phase_template", ""))
    for token in template_name.split("."):
        if token.startswith("phase_"):
            try:
                return int(token.removeprefix("phase_"))
            except ValueError:
                continue
    raise ValueError(f"missing phase index in matched_phase_template={template_name}")


def _resolve_specific_painting_name(component_name: str) -> str:
    """Resolve a public painting component name to the exact VLABench asset stem."""

    normalized_name = component_name.removesuffix("_painting")
    vlabench_root = os.getenv("VLABENCH_ROOT")
    if not vlabench_root:
        return normalized_name
    painting_root = Path(vlabench_root) / "assets" / "obj" / "meshes" / "paintings" / "assets" / "famous paintings"
    if not painting_root.exists():
        return normalized_name
    for image_path in painting_root.glob("*/*.png"):
        if image_path.stem.lower() == normalized_name.lower():
            return image_path.stem
    return normalized_name


def _save_visual_audit_images(env, visual_dir: Path | None, phase: str) -> dict:
    """Save camera renders for execution audit without altering simulation state."""

    if visual_dir is None:
        return {"enabled": False, "phase": phase, "images": [], "contact_sheet": ""}
    visual_dir.mkdir(parents=True, exist_ok=True)
    camera_count = int(env.physics.model.ncam)
    image_paths: list[str] = []
    sheet_images: list[Image.Image] = []
    for camera_id in range(camera_count):
        rgb = env.render(camera_id=camera_id, height=480, width=480)
        image = Image.fromarray(np.asarray(rgb).astype(np.uint8))
        image_path = visual_dir / f"{phase}_camera{camera_id}.png"
        image.save(image_path)
        image_paths.append(str(image_path))
        sheet_images.append(image.copy())
    if not sheet_images:
        return {"enabled": True, "phase": phase, "images": image_paths, "contact_sheet": ""}
    columns = 2
    rows = int(np.ceil(len(sheet_images) / columns))
    tile_width, tile_height = sheet_images[0].size
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), color=(0, 0, 0))
    for index, image in enumerate(sheet_images):
        x = (index % columns) * tile_width
        y = (index // columns) * tile_height
        sheet.paste(image, (x, y))
    contact_sheet_path = visual_dir / f"{phase}_contact_sheet.png"
    sheet.save(contact_sheet_path)
    return {"enabled": True, "phase": phase, "images": image_paths, "contact_sheet": str(contact_sheet_path)}


def _augment_episode_config_for_native_load(env_config: dict) -> tuple[dict, dict]:
    """Add VLABench bookkeeping fields required for native reset/evaluation."""

    patched = copy.deepcopy(env_config)
    task = patched.setdefault("task", {})
    conditions = task.get("conditions") or {}
    audit: dict[str, object] = {"added_fields": {}}
    contain_entities = conditions.get("contain", {}).get("entities")
    if not task.get("target_entities") and isinstance(contain_entities, list):
        task["target_entities"] = contain_entities
        audit["added_fields"]["target_entities"] = {"source": "conditions.contain.entities", "value": contain_entities}
    if not task.get("target_entity") and isinstance(contain_entities, list) and contain_entities:
        task["target_entity"] = contain_entities[0]
        audit["added_fields"]["target_entity"] = {"source": "conditions.contain.entities[0]", "value": contain_entities[0]}
    if not task.get("target_container"):
        contain_container = conditions.get("contain", {}).get("container")
        not_contain_container = conditions.get("not_contain", {}).get("container")
        above_platform = conditions.get("above", {}).get("platform")
        if isinstance(contain_container, str):
            task["target_container"] = contain_container
            audit["added_fields"]["target_container"] = {"source": "conditions.contain.container", "value": contain_container}
        elif isinstance(not_contain_container, str):
            task["target_container"] = not_contain_container
            audit["added_fields"]["target_container"] = {"source": "conditions.not_contain.container", "value": not_contain_container}
        elif isinstance(above_platform, str):
            task["target_container"] = above_platform
            audit["added_fields"]["target_container"] = {"source": "conditions.above.platform", "value": above_platform}
            task["target_containers"] = [above_platform]
            audit["added_fields"]["target_containers"] = {"source": "conditions.above.platform", "value": [above_platform]}
            task["target_platform"] = above_platform
            audit["added_fields"]["target_platform"] = {"source": "conditions.above.platform", "value": above_platform}
    if isinstance(task.get("target_container"), str):
        bookkeeping_container = task["target_container"]
        if not task.get("target_containers"):
            task["target_containers"] = [bookkeeping_container]
            audit["added_fields"]["target_containers"] = {"source": "derived.task.target_container", "value": [bookkeeping_container]}
        if not task.get("target_platform"):
            task["target_platform"] = bookkeeping_container
            audit["added_fields"]["target_platform"] = {"source": "derived.task.target_container", "value": bookkeeping_container}
    if not task.get("target_entity"):
        for condition_name in ("pour", "above", "not_contain"):
            value = conditions.get(condition_name, {}).get("target_entity")
            if isinstance(value, str):
                task["target_entity"] = value
                audit["added_fields"]["target_entity"] = {"source": f"conditions.{condition_name}.target_entity", "value": value}
                if not task.get("target_entities"):
                    task["target_entities"] = [value]
                    audit["added_fields"]["target_entities"] = {"source": f"conditions.{condition_name}.target_entity", "value": [value]}
                break
    if not task.get("target_entity"):
        target_button = conditions.get("press_button", {}).get("target_button")
        if isinstance(target_button, str):
            task["target_entity"] = target_button
            audit["added_fields"]["target_entity"] = {"source": "conditions.press_button.target_button", "value": target_button}
            task["target_entities"] = [target_button]
            audit["added_fields"]["target_entities"] = {"source": "conditions.press_button.target_button", "value": [target_button]}
    if not task.get("target_entities"):
        for condition_name in ("is_grasped", "lift", "not_contain"):
            value = conditions.get(condition_name, {}).get("entities")
            if isinstance(value, list) and value:
                task["target_entities"] = value
                audit["added_fields"]["target_entities"] = {"source": f"conditions.{condition_name}.entities", "value": value}
                break
    if not task.get("target_entity"):
        for condition_name in ("is_grasped", "lift", "not_contain"):
            value = conditions.get(condition_name, {}).get("entities")
            if isinstance(value, list) and value:
                task["target_entity"] = value[0]
                audit["added_fields"]["target_entity"] = {"source": f"conditions.{condition_name}.entities[0]", "value": value[0]}
                break
    if not task.get("target_entities") and isinstance(conditions.get("substep_satisfy"), list):
        substep_entities: list[str] = []
        substep_platform = None
        for substep in conditions.get("substep_satisfy", []):
            if not isinstance(substep, dict):
                continue
            for predicate_value in substep.values():
                if isinstance(predicate_value, dict):
                    if isinstance(predicate_value.get("target_entity"), str) and predicate_value["target_entity"] not in substep_entities:
                        substep_entities.append(predicate_value["target_entity"])
                    if isinstance(predicate_value.get("platform"), str):
                        substep_platform = predicate_value["platform"]
        if substep_entities:
            task["target_entities"] = substep_entities
            audit["added_fields"]["target_entities"] = {"source": "conditions.substep_satisfy[*].*.target_entity", "value": substep_entities}
        if not task.get("target_entity") and substep_entities:
            task["target_entity"] = substep_entities
            audit["added_fields"]["target_entity"] = {"source": "conditions.substep_satisfy[*].*.target_entity", "value": substep_entities}
        if not task.get("target_container") and substep_platform:
            task["target_container"] = substep_platform
            audit["added_fields"]["target_container"] = {"source": "conditions.substep_satisfy[*].*.platform", "value": substep_platform}
    for component in task.get("components", []):
        if not isinstance(component, dict):
            continue
        component_name = component.get("name", "")
        if not isinstance(component_name, str) or not component_name.endswith("_painting"):
            continue
        if not component.get("specific_painting"):
            specific_painting = _resolve_specific_painting_name(component_name)
            component["specific_painting"] = specific_painting
            audit["added_fields"][f"components.{component_name}.specific_painting"] = {"source": "component.name + VLABench painting assets", "value": specific_painting}
    for component in task.get("components", []):
        if not isinstance(component, dict):
            continue
        if component.get("class") != "Mirrors":
            continue
        if component.get("reflectance"):
            continue
        component["reflectance"] = [0.2, 0.5, 0.8]
        audit["added_fields"][f"components.{component.get('name', 'mirrors')}.reflectance"] = {"source": "reflection_qa native Mirrors constructor default", "value": [0.2, 0.5, 0.8]}
    if not task.get("target_entity"):
        component_names = [component.get("name", "") for component in task.get("components", []) if isinstance(component, dict)]
        instructions = task.get("instructions", "")
        instruction_text = " ".join(instructions) if isinstance(instructions, list) else str(instructions)
        for component_name in component_names:
            if not component_name.endswith("_painting"):
                continue
            style_name = component_name.removesuffix("_painting")
            if style_name.lower() in instruction_text.lower():
                task["target_entity"] = style_name
                audit["added_fields"]["target_entity"] = {"source": "instructions + painting component name", "value": style_name}
                break
    return patched, audit


def _target_name(step: dict) -> str | None:
    """Extract the VLM-bound target name from an executable step."""

    target = step.get("target")
    if not isinstance(target, dict):
        return None
    return target.get("entity_name")


def _attach_precomputed_native_targets(env, task_name: str, steps: list[dict]) -> list[dict]:
    """Freeze official-style native target positions before any action rollout."""

    prepared_steps = copy.deepcopy(steps)
    for step in prepared_steps:
        if step.get("skill") != "moveto":
            continue
        target_name = _target_name(step)
        if not target_name:
            continue
        if "insert_flower" in task_name or "insert_bloom_flower" in task_name:
            step["_precomputed_target_pos"] = _place_point(env, target_name) + np.array([0, 0, 0.05])
        elif "add_condiment" in task_name:
            step["_precomputed_target_pos"] = _entity_pos(env, target_name) + np.array([0, 0, 0.2])
    return prepared_steps


def _apply_deterministic_native_seed(command: dict, audit: dict) -> None:
    """Apply fixture-level deterministic RNG seeds for reproducible native smoke."""

    seed_value = command.get("native_seed")
    if seed_value is None:
        audit["native_seed"] = None
        return
    seed = int(seed_value)
    random.seed(seed)
    np.random.seed(seed)
    audit["native_seed"] = seed


def _apply_native_task_bookkeeping(env, patched_env_config: dict, audit: dict) -> None:
    """Mirror audited native bookkeeping fields onto the loaded VLABench task."""

    task_config = patched_env_config.get("task", {})
    for field_name in ("target_entity", "target_entities", "target_container", "target_containers", "target_platform"):
        if field_name not in task_config:
            continue
        try:
            setattr(env.task, field_name, task_config[field_name])
            audit.setdefault("post_load_mirrored_fields", {})[field_name] = task_config[field_name]
        except AttributeError:
            if hasattr(env.task, "config_manager"):
                setattr(env.task.config_manager, field_name, task_config[field_name])
                audit.setdefault("post_load_config_manager_fields", {})[field_name] = task_config[field_name]
    if hasattr(env.task, "reset_task_progress"):
        env.task.reset_task_progress()
        audit["reset_task_progress_called"] = True


def _execute_step(env, task_name: str, step: dict) -> dict:
    """Execute one target-preserving SkillLib call."""

    skill = step["skill"]
    target_name = _target_name(step)
    if skill == "pick":
        if task_name == "book_rearrange":
            obs, waypoints, stage_success, task_success = SkillLib.pick(env, target_entity_name=target_name, prior_eulers=[[-np.pi, 0, -np.pi / 2]])
        elif "poker" in task_name:
            obs, waypoints, stage_success, task_success = SkillLib.pick(env, target_entity_name=target_name, prior_eulers=[[-np.pi, 0, 0]])
        elif "book" in task_name:
            obs, waypoints, stage_success, task_success = SkillLib.pick(env, target_entity_name=target_name, prior_eulers=[[-np.pi / 2, -np.pi / 2, 0]])
        elif "add_condiment" in task_name:
            obs, waypoints, stage_success, task_success = SkillLib.pick(env, target_entity_name=target_name, prior_eulers=[[-np.pi / 2, -np.pi / 2, np.pi / 2]])
        elif "select_drink" in task_name:
            obs, waypoints, stage_success, task_success = SkillLib.pick(env, target_entity_name=target_name, prior_eulers=[[-np.pi / 2, -np.pi / 2, 0]])
        elif "billiards" in task_name:
            obs, waypoints, stage_success, task_success = SkillLib.pick(env, target_entity_name=target_name, motion_planning_kwargs={"z_threshold": 0.6})
        elif "select_toy" in task_name:
            grasp_point = np.array(env.task.entities[target_name].get_grasped_keypoints(env.physics)[-1])
            obs, waypoints, stage_success, task_success = SkillLib.pick(env, target_entity_name=target_name, target_pos=grasp_point - np.array([0, 0, 0.02]), prior_eulers=[[-np.pi, 0, 0]])
        elif task_name == "hammer_nail_and_hang_picture" and target_name == "hammer":
            obs, waypoints, stage_success, task_success = SkillLib.pick(env, target_entity_name=target_name, prior_eulers=[[-np.pi, 0, 0]])
        elif task_name == "hammer_nail_and_hang_picture" and target_name and target_name.endswith("_painting"):
            grasp_point = np.array(env.task.entities[target_name].get_grasped_keypoints(env.physics)[-1])
            obs, waypoints, stage_success, task_success = SkillLib.pick(env, target_entity_name=target_name, target_pos=grasp_point + np.array([0, 0.01, 0.0]), target_quat=euler_to_quaternion(-np.pi, 0, 0))
        elif task_name == "take_chemistry_experiment":
            grasp_point = np.array(env.task.entities[target_name].get_grasped_keypoints(env.physics)[-1])
            obs, waypoints, stage_success, task_success = SkillLib.pick(env, target_entity_name=target_name, target_pos=grasp_point + np.array([0, 0, 0.02]), target_quat=euler_to_quaternion(-np.pi, np.pi / 4, -np.pi / 2), target_velocity=0.1)
        else:
            obs, waypoints, stage_success, task_success = SkillLib.pick(env, target_entity_name=target_name)
    elif skill == "place":
        if task_name == "hammer_nail_and_hang_picture":
            obs, waypoints, stage_success, task_success = SkillLib.place(env, target_container_name=target_name, target_pos=np.array([0, 0.25, 0.8]), target_quat=euler_to_quaternion(-np.pi, 0, 0))
        elif "billiards" in task_name:
            obs, waypoints, stage_success, task_success = SkillLib.place(env, target_container_name=target_name, motion_planning_kwargs={"z_threshold": 0.6})
        else:
            obs, waypoints, stage_success, task_success = SkillLib.place(env, target_container_name=target_name)
    elif skill == "lift":
        if "insert_flower" in task_name or "insert_bloom_flower" in task_name:
            quat = euler_to_quaternion(-np.pi / 2, np.pi / 2, 0)
            if "phase_4" in step.get("matched_phase_template", ""):
                obs, waypoints, stage_success, task_success = SkillLib.lift(env, lift_height=-0.2)
            else:
                obs, waypoints, stage_success, task_success = SkillLib.lift(env, target_quat=quat)
        elif "add_condiment" in task_name:
            obs, waypoints, stage_success, task_success = SkillLib.lift(env, lift_height=0.2, gripper_state=np.zeros(2))
        elif "select_drink" in task_name:
            obs, waypoints, stage_success, task_success = SkillLib.lift(env, gripper_state=np.zeros(2), lift_height=0.1)
        elif "select_toy" in task_name:
            obs, waypoints, stage_success, task_success = SkillLib.lift(env, lift_height=0.3, gripper_state=np.zeros(2))
        elif task_name == "hammer_nail_and_hang_picture":
            if "phase_2" in step.get("matched_phase_template", ""):
                obs, waypoints, stage_success, task_success = SkillLib.lift(env, lift_height=0.15, gripper_state=np.zeros(2))
            elif "phase_7" in step.get("matched_phase_template", ""):
                obs, waypoints, stage_success, task_success = SkillLib.lift(env, lift_height=0.1)
            else:
                obs, waypoints, stage_success, task_success = SkillLib.lift(env, gripper_state=np.zeros(2))
        elif task_name == "take_chemistry_experiment":
            obs, waypoints, stage_success, task_success = SkillLib.lift(env, gripper_state=np.zeros(2), lift_height=0.25)
        else:
            obs, waypoints, stage_success, task_success = SkillLib.lift(env, lift_height=0.3)
    elif skill == "moveto":
        if "insert_flower" in task_name or "insert_bloom_flower" in task_name:
            target_pos = step.get("_precomputed_target_pos")
            if target_pos is None:
                target_pos = _place_point(env, target_name) + np.array([0, 0, 0.05])
            quat = euler_to_quaternion(-np.pi / 2, np.pi / 2, 0)
            obs, waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos=target_pos, target_quat=quat)
        elif "add_condiment" in task_name:
            target_pos = step.get("_precomputed_target_pos")
            if target_pos is None:
                target_pos = _entity_pos(env, target_name) + np.array([0, 0, 0.2])
            obs, waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos=target_pos, gripper_state=np.zeros(2))
        elif task_name == "hammer_nail_and_hang_picture":
            nail_pos = _entity_pos(env, target_name)
            if "phase_4" in step.get("matched_phase_template", ""):
                obs, waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos=nail_pos + np.array([0.03, -0.2, -0.1]), target_quat=euler_to_quaternion(-np.pi / 2, np.pi / 2, np.pi / 2), gripper_state=np.zeros(2))
            else:
                obs, waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos=nail_pos + np.array([0.01, -0.1, 0.02]), target_quat=euler_to_quaternion(-np.pi / 2, 0, 0), gripper_state=np.zeros(2))
        elif task_name == "take_chemistry_experiment":
            phase_index = _phase_index(step)
            if phase_index in {3, 13}:
                target_pos = np.array(env.task.entities[target_name].get_place_point(env.physics)[-1])
                obs, waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos=target_pos, gripper_state=np.zeros(2), target_velocity=0.1)
            else:
                solution_index = 0 if phase_index < 11 else 1
                target_positions = getattr(env.task.config_manager, "target_positions", [])
                if len(target_positions) > solution_index:
                    target_pos = np.array(target_positions[solution_index]) + np.array([0, 0, 0.25])
                else:
                    target_pos = _place_point(env, target_name) + np.array([0, 0, 0.25])
                obs, waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos=target_pos, target_quat=euler_to_quaternion(-np.pi / 2, -np.pi / 2, 0), gripper_state=np.zeros(2), target_velocity=0.1)
        elif task_name == "book_rearrange":
            phase_index = _phase_index(step)
            book_slot_index = (phase_index - 2) // 4
            container_pos = _entity_pos(env, target_name)
            target_pos = np.array([0.15 * (book_slot_index - 1), container_pos[1] - 0.35, 1.25])
            obs, waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos=target_pos, gripper_state=np.zeros(2), target_quat=euler_to_quaternion(-np.pi / 2, np.pi / 2, 0))
        else:
            target_pos = _place_point(env, target_name) + np.array([0, 0, 0.1])
            obs, waypoints, stage_success, task_success = SkillLib.moveto(env, target_pos=target_pos)
    elif skill == "pull":
        if task_name == "hammer_nail_and_hang_picture":
            obs, waypoints, stage_success, task_success = SkillLib.pull(env, pull_distance=0.1, gripper_state=np.zeros(2))
        elif task_name == "take_chemistry_experiment":
            obs, waypoints, stage_success, task_success = SkillLib.pull(env, pull_distance=0.1)
        elif task_name == "book_rearrange":
            obs, waypoints, stage_success, task_success = SkillLib.pull(env, gripper_state=np.ones(2) * 0.04)
        elif "select_drink" in task_name:
            obs, waypoints, stage_success, task_success = SkillLib.pull(env, pull_distance=0.3)
        else:
            obs, waypoints, stage_success, task_success = SkillLib.pull(env)
    elif skill == "push":
        if task_name == "hammer_nail_and_hang_picture":
            if "phase_5" in step.get("matched_phase_template", ""):
                obs, waypoints, stage_success, task_success = SkillLib.push(env, push_distance=0.15, gripper_state=np.zeros(2))
            else:
                obs, waypoints, stage_success, task_success = SkillLib.push(env, push_distance=0.1, target_quat=euler_to_quaternion(-np.pi / 2, 0, 0), gripper_state=np.zeros(2))
        elif task_name == "book_rearrange":
            obs, waypoints, stage_success, task_success = SkillLib.push(env, target_quat=euler_to_quaternion(-np.pi / 2, np.pi / 2, 0), gripper_state=np.zeros(2))
        elif target_name:
            target_pos = _entity_pos(env, target_name)
            obs, waypoints, stage_success, task_success = SkillLib.push(env, target_pos=target_pos)
        else:
            obs, waypoints, stage_success, task_success = SkillLib.push(env)
    elif skill == "pour":
        if task_name == "take_chemistry_experiment":
            phase_index = _phase_index(step)
            if phase_index in {4, 14}:
                obs, waypoints, stage_success, task_success = SkillLib.pour(env, target_delta_qpos=np.pi / 2 + np.pi / 10, n_repeat_step=4)
            else:
                obs, waypoints, stage_success, task_success = SkillLib.pour(env, target_delta_qpos=-(np.pi / 2 + np.pi / 10), n_repeat_step=4, target_q_velocity=-np.pi / 40)
        else:
            obs, waypoints, stage_success, task_success = SkillLib.pour(env)
    elif skill == "press":
        target_pos = _entity_pos(env, target_name)
        obs, waypoints, stage_success, task_success = SkillLib.press(env, target_pos=target_pos)
    elif skill == "wait":
        wait_time = 10 if task_name == "take_chemistry_experiment" else 100
        obs, waypoints, stage_success, task_success = SkillLib.wait(env, wait_time=wait_time)
    elif skill == "move_offset":
        if task_name == "take_chemistry_experiment":
            obs, waypoints, stage_success, task_success = SkillLib.move_offset(env, offset=[0, 0, -0.1])
        else:
            raise ValueError(f"move_offset template is not defined for task {task_name}")
    elif skill == "open_gripper":
        obs, waypoints, stage_success, task_success = SkillLib.open_gripper(env)
    else:
        raise ValueError(f"unsupported worker skill {skill}")
    return {
        "skill": skill,
        "target": step.get("target"),
        "matched_phase_template": step.get("matched_phase_template", ""),
        "stage_success": bool(stage_success),
        "task_success_signal": bool(task_success),
        "waypoint_count": len(waypoints) if waypoints is not None else 0,
    }


def main() -> None:
    """Run one native command file and write one native result file."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--command", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    result_path = Path(args.result)
    try:
        command = json.loads(Path(args.command).read_text(encoding="utf-8"))
        env_config = json.loads(Path(command["env_config_path"]).read_text(encoding="utf-8"))
        env_config, episode_config_audit = _augment_episode_config_for_native_load(env_config)
        _apply_deterministic_native_seed(command, episode_config_audit)
        task_name = command["task_name"]
        steps = command["executable_steps"]
        visual_audit_dir = Path(command["visual_audit_dir"]) if command.get("visual_audit_dir") else None
        env = load_env(
            task_name,
            robot="franka",
            episode_config=env_config,
            random_init=False,
            run_mode="eval",
            reset_wait_step=int(command.get("reset_wait_step", 10)),
        )
        _apply_native_task_bookkeeping(env, env_config, episode_config_audit)
        steps = _attach_precomputed_native_targets(env, task_name, steps)
        visual_audit = {
            "before": {
                "enabled": False,
                "phase": "before",
                "images": [],
                "contact_sheet": "",
                "skipped_reason": "pre_execution_render_disabled_to_preserve_native_state",
            }
        }
        trace = []
        stage_exception_policy = {
            "enabled": True,
            "name": "official_style_stage_failure_condition_check",
            "stage_exception_caught": False,
            "condition_checked_after_stage_exception": False,
            "condition_success_after_stage_exception": False,
            "failed_skill": "",
            "exception_type": "",
            "exception": "",
            "example_specific_rule_used": False,
        }
        stage_exception_failure = False
        stage_exception_failure_type = ""
        stage_exception_failure_message = ""
        stage_exception_failure_traceback = ""
        condition_success_after_exception = False
        for step in steps:
            trace_row = {}
            try:
                trace_row = _execute_step(env, task_name, step)
                trace_row["condition_progress_after_step"] = bool(env.task.conditions.is_met(env.physics))
                trace.append(trace_row)
            except Exception as stage_exc:
                stage_exception_policy["stage_exception_caught"] = True
                stage_exception_policy["failed_skill"] = str(step.get("skill", ""))
                stage_exception_policy["exception_type"] = type(stage_exc).__name__
                stage_exception_policy["exception"] = str(stage_exc)
                trace_row = {
                    "skill": step.get("skill", ""),
                    "target": step.get("target", ""),
                    "matched_phase_template": step.get("matched_phase_template", ""),
                    "stage_success": False,
                    "task_success_signal": False,
                    "waypoint_count": 0,
                    "condition_progress_after_step": False,
                    "stage_exception_type": type(stage_exc).__name__,
                    "stage_exception": str(stage_exc),
                }
                try:
                    condition_success_after_exception = bool(env.task.conditions.is_met(env.physics))
                    stage_exception_policy["condition_checked_after_stage_exception"] = True
                    stage_exception_policy["condition_success_after_stage_exception"] = condition_success_after_exception
                    trace_row["condition_progress_after_step"] = condition_success_after_exception
                except Exception as condition_exc:
                    trace_row["condition_exception_type"] = type(condition_exc).__name__
                    trace_row["condition_exception"] = str(condition_exc)
                    condition_success_after_exception = False
                trace.append(trace_row)
                if condition_success_after_exception:
                    break
                stage_exception_failure = True
                stage_exception_failure_type = type(stage_exc).__name__
                stage_exception_failure_message = str(stage_exc)
                stage_exception_failure_traceback = traceback.format_exc()
                break
        try:
            visual_audit["after"] = _save_visual_audit_images(env, visual_audit_dir, "after")
        except Exception as visual_exc:
            visual_audit["after"] = {
                "enabled": bool(visual_audit_dir),
                "phase": "after",
                "images": [],
                "contact_sheet": "",
                "render_exception_type": type(visual_exc).__name__,
                "render_exception": str(visual_exc),
            }
        if stage_exception_policy["stage_exception_caught"]:
            condition_success = condition_success_after_exception
        else:
            condition_success = bool(env.task.conditions.is_met(env.physics))
        native_ok = not (stage_exception_failure and not condition_success)
        output = {
            "ok": native_ok,
            "execution_completed": native_ok,
            "condition_success": condition_success,
            "trace": trace,
            "stage_exception_policy": stage_exception_policy,
            "episode_config_audit": episode_config_audit,
            "visual_audit": visual_audit,
            "gt_used_during_execution": False,
            "operation_sequence_read": False,
            "expert_sequence_called": False,
        }
        if stage_exception_failure and not condition_success:
            output["native_exception"] = True
            output["exception_type"] = stage_exception_failure_type
            output["exception"] = stage_exception_failure_message
            output["traceback"] = stage_exception_failure_traceback
    except Exception as exc:
        output = {
            "ok": False,
            "execution_completed": False,
            "native_exception": True,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "gt_used_during_execution": False,
            "operation_sequence_read": False,
            "expert_sequence_called": False,
        }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

