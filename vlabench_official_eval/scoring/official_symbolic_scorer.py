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

    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return value


def normalize_numeric_target_ids(skill_sequence: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    """Canonicalize target number types without changing skills, order, or target identity."""

    changed = False
    normalized_sequence: list[dict[str, Any]] = []
    for step in skill_sequence:
        new_step = dict(step)
        params = dict(step.get("params") or {})
        for key, value in list(params.items()):
            normalized_value = _canonicalize_numeric_target_ids(value)
            if normalized_value != value or type(normalized_value) is not type(value):
                changed = True
            params[key] = normalized_value
        new_step["params"] = params
        normalized_sequence.append(new_step)
    return normalized_sequence, changed


def ensure_vlabench_root() -> Path:
    """Resolve the official VLABench package root without importing VLABench top-level."""

    if os.environ.get("VLABENCH_ROOT"):
        return Path(os.environ["VLABENCH_ROOT"]).resolve()
    candidates = [
        Path("${VLABENCH_ROOT}"),
        Path.cwd() / "third_party" / "VLABench" / "VLABench",
    ]
    for candidate in candidates:
        if (candidate / "evaluation" / "utils.py").exists():
            os.environ["VLABENCH_ROOT"] = str(candidate)
            return candidate.resolve()
    raise FileNotFoundError(
        {
            "message": "Could not resolve VLABENCH_ROOT. Set VLABENCH_ROOT to the official VLABench/VLABench package directory.",
            "checked_candidates": [str(path) for path in candidates],
        }
    )


def load_official_get_final_score():
    """Load official VLABench get_final_score directly from evaluation/utils.py."""

    root = ensure_vlabench_root()
    utils_path = root / "evaluation" / "utils.py"
    spec = importlib.util.spec_from_file_location("_vlabench_official_evaluation_utils", utils_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load official VLABench utils from {utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_final_score


def load_seq_independent_tasks() -> set[str]:
    """Load official VLABench list of sequence-independent tasks."""

    root = ensure_vlabench_root()
    config_path = root / "configs" / "evaluation" / "seq_independent_task.json"
    if not config_path.exists():
        return set()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return set(data if isinstance(data, list) else [])


def score_with_official_vlabench(
    *,
    task_name: str,
    gt_skill_sequence: list[dict[str, Any]],
    model_output: dict[str, Any],
    seq_independent_tasks: set[str],
) -> dict[str, Any]:
    """Score one model output with official VLABench get_final_score."""

    get_final_score = load_official_get_final_score()

    if "format_error" in model_output:
        score = dict(ZERO_SCORE)
        score["official_dependency"] = "format_error"
        score["symbolic_success"] = False
        return score
    model_skill_sequence = model_output.get("skill_sequence")
    if not isinstance(model_skill_sequence, list):
        score = dict(ZERO_SCORE)
        score["official_dependency"] = "invalid_skill_sequence"
        score["symbolic_success"] = False
        return score
    dependency = "Seq-independent" if task_name in seq_independent_tasks else "Sequential"
    score = get_final_score(gt_skill_sequence, model_skill_sequence, dependency=dependency)
    score["official_dependency"] = dependency
    score["symbolic_success"] = float(score.get("exact_match_score", 0)) >= 100.0
    return score


def score_with_type_normalized_vlabench(
    *,
    task_name: str,
    gt_skill_sequence: list[dict[str, Any]],
    model_output: dict[str, Any],
    seq_independent_tasks: set[str],
) -> dict[str, Any]:
    """Score an auxiliary target-id-type-normalized view of the same model output."""

    if "format_error" in model_output:
        return {
            "type_normalized_skill_match_score": 0,
            "type_normalized_entity_match_score": 0,
            "type_normalized_skill_with_entity_match_score": 0,
            "type_normalized_exact_match_score": 0,
            "type_normalized_total_score": 0,
            "type_normalized_symbolic_success": False,
            "type_normalized_numeric_cast_applied": False,
        }
    model_skill_sequence = model_output.get("skill_sequence")
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
    normalized_sequence, changed = normalize_numeric_target_ids(model_skill_sequence)
    normalized_score = score_with_official_vlabench(
        task_name=task_name,
        gt_skill_sequence=gt_skill_sequence,
        model_output={"skill_sequence": normalized_sequence},
        seq_independent_tasks=seq_independent_tasks,
    )
    return {
        "type_normalized_skill_match_score": normalized_score.get("skill_match_score", 0),
        "type_normalized_entity_match_score": normalized_score.get("entity_match_score", 0),
        "type_normalized_skill_with_entity_match_score": normalized_score.get("skill_with_entity_match_score", 0),
        "type_normalized_exact_match_score": normalized_score.get("exact_match_score", 0),
        "type_normalized_total_score": normalized_score.get("total_score", 0),
        "type_normalized_symbolic_success": normalized_score.get("symbolic_success", False),
        "type_normalized_numeric_cast_applied": changed,
    }
