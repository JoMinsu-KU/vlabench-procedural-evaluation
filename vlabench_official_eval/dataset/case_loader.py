"""Load the 2,133-case subset in the input shape used by VLABench VLM evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OfficialVLMCase:
    """One VLABench VLM evaluation example with public inputs and private scoring target."""

    case_id: str
    category: str
    task_name: str
    example_name: str
    example_index: int
    input_image_path: Path
    input_mask_path: Path
    instruction: str
    gt_operation_sequence_path: Path
    gt_skill_sequence: list[dict[str, Any]]


def load_manifest(manifest_path: Path) -> list[str]:
    """Read a JSON list of case ids such as Category/task/example7."""

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"Manifest must be a JSON list of case-id strings: {manifest_path}")
    return data


def split_case_id(case_id: str) -> tuple[str, str, str, int]:
    """Split Category/task/exampleN into fields used by both dataset roots."""

    parts = case_id.split("/")
    if len(parts) != 3:
        raise ValueError(f"Invalid case_id format: {case_id}")
    category = parts[0]
    task_name = parts[1]
    example_name = parts[2]
    if not example_name.startswith("example"):
        raise ValueError(f"Invalid example name in case_id: {case_id}")
    example_index = int(example_name[len("example") :])
    return category, task_name, example_name, example_index


def load_case(*, case_id: str, public_root: Path, gt_root: Path) -> OfficialVLMCase:
    """Load one case using official VLM inputs and GT sequence only for scoring."""

    category, task_name, example_name, example_index = split_case_id(case_id)
    public_case_root = public_root / category / task_name / example_name
    gt_case_root = gt_root / category / task_name / example_name
    input_image_path = public_case_root / "input.png"
    input_mask_path = public_case_root / "input_mask.png"
    instruction_path = public_case_root / "instruction.txt"
    gt_operation_sequence_path = gt_case_root / "output" / "operation_sequence.json"
    required_paths = [input_image_path, input_mask_path, instruction_path, gt_operation_sequence_path]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError({"case_id": case_id, "missing_paths": missing_paths})
    instruction = instruction_path.read_text(encoding="utf-8").strip()
    gt_operation = json.loads(gt_operation_sequence_path.read_text(encoding="utf-8"))
    gt_skill_sequence = gt_operation.get("skill_sequence")
    if not isinstance(gt_skill_sequence, list):
        raise ValueError(f"GT operation_sequence.json has no list skill_sequence: {gt_operation_sequence_path}")
    return OfficialVLMCase(
        case_id=case_id,
        category=category,
        task_name=task_name,
        example_name=example_name,
        example_index=example_index,
        input_image_path=input_image_path,
        input_mask_path=input_mask_path,
        instruction=instruction,
        gt_operation_sequence_path=gt_operation_sequence_path,
        gt_skill_sequence=gt_skill_sequence,
    )


def select_case_ids(*, manifest: list[str], max_cases: int = 0, case_list_file: Path | None = None) -> list[str]:
    """Apply optional case-list and max-case filters without changing manifest semantics."""

    selected = list(manifest)
    if case_list_file is not None:
        requested = [line.strip() for line in case_list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        requested_set = set(requested)
        selected = [case_id for case_id in selected if case_id in requested_set]
        missing_requested = [case_id for case_id in requested if case_id not in set(selected)]
        if missing_requested:
            raise ValueError({"missing_requested_case_ids": missing_requested})
    if max_cases > 0:
        selected = selected[:max_cases]
    return selected
