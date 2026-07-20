"""Numeric target binding for public-view VLABench components."""

from __future__ import annotations

from procedural_interface.evaluation.taxonomy import TARGET_BINDING_FAILURE, status_record


def bind_numeric_target(env_config: dict, target: int) -> dict:
    """Bind a VLM-selected numeric target to a component record."""

    # Read components from the allowed internal scene config.
    components = env_config.get("task", {}).get("components", [])
    # Reject non-list component structures rather than guessing.
    if not isinstance(components, list):
        return {"ok": False, "status": status_record(TARGET_BINDING_FAILURE, "missing_components", "task.components is not a list")}
    # Reject negative and out-of-range target numbers deterministically.
    if target < 0 or target >= len(components):
        return {"ok": False, "status": status_record(TARGET_BINDING_FAILURE, "target_out_of_range", f"target={target}, components={len(components)}")}
    # Fetch the component selected by the model.
    component = components[target]
    # Require a component dictionary with at least a name for native binding.
    if not isinstance(component, dict) or not component.get("name"):
        return {"ok": False, "status": status_record(TARGET_BINDING_FAILURE, "component_without_name", f"target={target}")}
    # Return model-selected target provenance without correcting the target.
    return {
        "ok": True,
        "target_number": target,
        "entity_name": component.get("name"),
        "entity_class": component.get("class", ""),
        "target_source": "vlm_output",
        "ground_truth_target_used": False,
    }
