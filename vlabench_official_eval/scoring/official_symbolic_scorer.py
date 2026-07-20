"""Thin wrapper around the official VLABench VLM symbolic scorer."""

from __future__ import annotations

import json
import os
import importlib.util
from pathlib import Path
from typing import Any


ZERO_SCORE = {
    "skill_match_score": 0,
    "entity_match_score": 0,
    "skill_with_entity_match_score": 0,
    "exact_match_score": 0,
    "total_score": 0,
}


def _canonicalize_numeric_target_ids(value: Any) -> Any:
    """Convert only pure numeric target-id strings to integers."""

    # Preserve non-string values exactly so official numeric outputs remain untouched.
    if not isinstance(value, str):
        return value
    # Remove surrounding whitespace because models sometimes output " 2 ".
    stripped = value.strip()
    # Convert only unsigned integer identifiers; labels such as "right_1" must remain strings.
    if stripped.isdigit():
        return int(stripped)
    # Return all semantic labels unchanged.
    return value


def normalize_numeric_target_ids(skill_sequence: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Canonicalize target number types without changing skills, order, or target identity."""

    # Track whether the auxiliary normalization actually changed any value.
    changed = False
    # Build a new sequence so the official parsed output audit remains unchanged.
    normalized_sequence: list[dict[str, Any]] = []
    # Walk through each skill call in order and preserve the original sequence length.
    for step in skill_sequence:
        # Copy the step dict so we do not mutate model_output in-place.
        new_step = dict(step)
        # Copy params defensively because the official parser stores nested dictionaries.
        params = dict(step.get("params") or {})
        # Normalize only parameter values, not parameter keys.
        for key, value in list(params.items()):
            # Apply the narrow numeric-string-to-int rule.
            normalized_value = _canonicalize_numeric_target_ids(value)
            # Record whether anything changed for audit.
            if normalized_value != value or type(normalized_value) is not type(value):
                changed = True
            # Store the normalized parameter value.
            params[key] = normalized_value
        # Preserve the original skill name exactly.
        new_step["params"] = params
        # Append the normalized step in the same order.
        normalized_sequence.append(new_step)
    # Return both the normalized sequence and an audit flag.
    return normalized_sequence, changed


def ensure_vlabench_root() -> Path:
    """Resolve the official VLABench package root without importing VLABench top-level."""

    # Respect an explicit environment variable if the runtime already provides one.
    if os.environ.get("VLABENCH_ROOT"):
        return Path(os.environ["VLABENCH_ROOT"]).resolve()
    # Fall back to a conventional repository-local third-party checkout.
    candidates = [
        Path.cwd() / "third_party" / "VLABench" / "VLABench",
    ]
    # Select the first candidate that contains the official scorer file.
    for candidate in candidates:
        if (candidate / "evaluation" / "utils.py").exists():
            os.environ["VLABENCH_ROOT"] = str(candidate)
            return candidate.resolve()
    # Fail with actionable paths instead of importing VLABench top-level and pulling extra model deps.
    raise FileNotFoundError(
        {
            "message": "Could not resolve VLABENCH_ROOT. Set VLABENCH_ROOT to the official VLABench/VLABench package directory.",
            "checked_candidates": [str(path) for path in candidates],
        }
    )


def load_official_get_final_score():
    """Load official VLABench get_final_score directly from evaluation/utils.py."""

    # Resolve the official package root without importing VLABench top-level.
    root = ensure_vlabench_root()
    # This is the official file that defines get_final_score.
    utils_path = root / "evaluation" / "utils.py"
    # Build an import spec with a private module name.
    spec = importlib.util.spec_from_file_location("_vlabench_official_evaluation_utils", utils_path)
    # A missing spec indicates the official file cannot be loaded.
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official VLABench utils from {utils_path}")
    # Create a module object from the spec.
    module = importlib.util.module_from_spec(spec)
    # Execute only the official utils.py file, avoiding VLABench package __init__ side effects.
    spec.loader.exec_module(module)
    # Return the exact official scoring function.
    return module.get_final_score


def load_seq_independent_tasks() -> set[str]:
    """Load official VLABench list of sequence-independent tasks."""

    # Locate the official package/config root.
    root = ensure_vlabench_root()
    # This is the same file used by VLABench.evaluation.evaluator.vlm.VLMEvaluator.
    config_path = root / "configs" / "evaluation" / "seq_independent_task.json"
    # Missing config means the installed VLABench package is incomplete.
    if not config_path.exists():
        return set()
    # The official file is a JSON list.
    data = json.loads(config_path.read_text(encoding="utf-8"))
    # Convert to a set for fast membership checks.
    return set(data if isinstance(data, list) else [])


def score_with_official_vlabench(
    *,
    task_name: str,
    gt_skill_sequence: list[dict[str, Any]],
    model_output: dict[str, Any],
    seq_independent_tasks: set[str],
) -> dict[str, Any]:
    """Score one model output with official VLABench get_final_score."""

    # Load the official function directly from VLABench/evaluation/utils.py.
    get_final_score = load_official_get_final_score()

    # Match official evaluator behavior: format errors receive all-zero scores.
    if "format_error" in model_output:
        score = dict(ZERO_SCORE)
        score["official_dependency"] = "format_error"
        score["symbolic_success"] = False
        return score
    # The parsed skill sequence must be a list of official skill-call dicts.
    model_skill_sequence = model_output.get("skill_sequence")
    # Invalid parsed structure is still a format failure for scoring.
    if not isinstance(model_skill_sequence, list):
        score = dict(ZERO_SCORE)
        score["official_dependency"] = "invalid_skill_sequence"
        score["symbolic_success"] = False
        return score
    # VLABench treats a small subset as sequence-independent.
    dependency = "Seq-independent" if task_name in seq_independent_tasks else "Sequential"
    # Delegate to the official graph/sequence scorer.
    score = get_final_score(gt_skill_sequence, model_skill_sequence, dependency=dependency)
    # Store the dependency mode for audit and reproducibility.
    score["official_dependency"] = dependency
    # G0 symbolic success is exact official precise matching at 100.
    score["symbolic_success"] = float(score.get("exact_match_score", 0)) >= 100.0
    # Return the official score plus binary success.
    return score


def score_with_type_normalized_vlabench(
    *,
    task_name: str,
    gt_skill_sequence: list[dict[str, Any]],
    model_output: dict[str, Any],
    seq_independent_tasks: set[str],
) -> dict[str, Any]:
    """Score an auxiliary target-id-type-normalized view of the same model output."""

    # Match official evaluator behavior for unparsable outputs.
    if "format_error" in model_output:
        # Prefix zero scores so they do not overwrite official strict fields.
        return {
            "type_normalized_skill_match_score": 0,
            "type_normalized_entity_match_score": 0,
            "type_normalized_skill_with_entity_match_score": 0,
            "type_normalized_exact_match_score": 0,
            "type_normalized_total_score": 0,
            "type_normalized_symbolic_success": False,
            "type_normalized_numeric_cast_applied": False,
        }
    # Read the parsed official skill sequence.
    model_skill_sequence = model_output.get("skill_sequence")
    # Invalid parsed structures remain invalid under the auxiliary view.
    if not isinstance(model_skill_sequence, list):
        return {
            "type_normalized_skill_match_score": 0,
            "type_normalized_entity_match_score": 0,
            "type_normalized_skill_with_entity_match_score": 0,
            "type_normalized_exact_match_score": 0,
            "type_normalized_total_score": 0,
            "type_normalized_symbolic_success": False,
            "type_normalized_numeric_cast_applied": False,
        }
    # Canonicalize only numeric target identifiers represented as strings.
    normalized_sequence, changed = normalize_numeric_target_ids(model_skill_sequence)
    # Reuse the official scorer on the normalized view.
    normalized_score = score_with_official_vlabench(
        task_name=task_name,
        gt_skill_sequence=gt_skill_sequence,
        model_output={"skill_sequence": normalized_sequence},
        seq_independent_tasks=seq_independent_tasks,
    )
    # Prefix every metric so official strict results remain primary and untouched.
    return {
        "type_normalized_skill_match_score": normalized_score.get("skill_match_score", 0),
        "type_normalized_entity_match_score": normalized_score.get("entity_match_score", 0),
        "type_normalized_skill_with_entity_match_score": normalized_score.get("skill_with_entity_match_score", 0),
        "type_normalized_exact_match_score": normalized_score.get("exact_match_score", 0),
        "type_normalized_total_score": normalized_score.get("total_score", 0),
        "type_normalized_symbolic_success": normalized_score.get("symbolic_success", False),
        "type_normalized_numeric_cast_applied": changed,
    }
