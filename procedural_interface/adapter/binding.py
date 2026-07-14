"""Numeric target binding for public-view VLABench components."""

from __future__ import annotations

from procedural_interface.evaluation.taxonomy import TARGET_BINDING_FAILURE, status_record


def bind_numeric_target(env_config: dict, target: int) -> dict:
    """Bind a VLM-selected numeric target to a component record."""

    components = env_config.get("task", {}).get("components", [])
    if not isinstance(components, list):
        return {"ok": False, "status": status_record(TARGET_BINDING_FAILURE, "missing_components", "task.components is not a list")}
    if target < 0 or target >= len(components):
        return {"ok": False, "status": status_record(TARGET_BINDING_FAILURE, "target_out_of_range", f"target={target}, components={len(components)}")}
    component = components[target]
    if not isinstance(component, dict) or not component.get("name"):
        return {"ok": False, "status": status_record(TARGET_BINDING_FAILURE, "component_without_name", f"target={target}")}
    return {
        "ok": True,
        "target_number": target,
        "entity_name": component.get("name"),
        "entity_class": component.get("class", ""),
        "target_source": "vlm_output",
        "ground_truth_target_used": False,
    }
