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

    # The manifest is intentionally a simple JSON list so both evaluation branches share it.
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    # A malformed manifest would break paired evaluation, so fail before any model call.
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"Manifest must be a JSON list of case-id strings: {manifest_path}")
    # Preserve manifest order to keep official and procedural outputs pairable by row index.
    return data


def split_case_id(case_id: str) -> tuple[str, str, str, int]:
    """Split Category/task/exampleN into fields used by both dataset roots."""

    # Case ids are generated from the verified 2,133-case manifest and must have three parts.
    parts = case_id.split("/")
    # Any mismatch here means the same example cannot be mapped consistently across roots.
    if len(parts) != 3:
        raise ValueError(f"Invalid case_id format: {case_id}")
    # The first part is the VLABench evaluation category, for example CommenSence or M&T.
    category = parts[0]
    # The second part is the concrete VLABench task family.
    task_name = parts[1]
    # The third part is exampleN.
    example_name = parts[2]
    # Official VLABench stores example ids as integer suffixes after the literal prefix.
    if not example_name.startswith("example"):
        raise ValueError(f"Invalid example name in case_id: {case_id}")
    # The integer index is useful for compatibility with official evaluator conventions.
    example_index = int(example_name[len("example") :])
    # Return both the original example name and parsed numeric index.
    return category, task_name, example_name, example_index


def load_case(*, case_id: str, public_root: Path, gt_root: Path) -> OfficialVLMCase:
    """Load one case using official VLM inputs and GT sequence only for scoring."""

    # Split once so public inputs and GT outputs are resolved from the same case id.
    category, task_name, example_name, example_index = split_case_id(case_id)
    # The public view contains only model-visible files.
    public_case_root = public_root / category / task_name / example_name
    # The raw VLM evaluation tree contains output/operation_sequence.json for scoring.
    gt_case_root = gt_root / category / task_name / example_name
    # Official VLABench VLM input image without numeric labels.
    input_image_path = public_case_root / "input.png"
    # Official VLABench VLM input image with object numbers.
    input_mask_path = public_case_root / "input_mask.png"
    # Natural-language instruction supplied to the model.
    instruction_path = public_case_root / "instruction.txt"
    # Official ground-truth symbolic sequence used only after inference.
    gt_operation_sequence_path = gt_case_root / "output" / "operation_sequence.json"
    # Check every required file before inference so missing data is reported as dataset error.
    required_paths = [input_image_path, input_mask_path, instruction_path, gt_operation_sequence_path]
    # Build a readable list for diagnostics instead of failing later with a nested FileNotFoundError.
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    # Missing files mean the paired 2,133 subset is not fully available for official scoring.
    if missing_paths:
        raise FileNotFoundError({"case_id": case_id, "missing_paths": missing_paths})
    # Read only the public instruction for prompt construction.
    instruction = instruction_path.read_text(encoding="utf-8").strip()
    # Read GT sequence after prompt-visible data, but never pass it to the model caller.
    gt_operation = json.loads(gt_operation_sequence_path.read_text(encoding="utf-8"))
    # VLABench official scorer expects the list under skill_sequence.
    gt_skill_sequence = gt_operation.get("skill_sequence")
    # Fail fast if the GT file does not match the official output schema.
    if not isinstance(gt_skill_sequence, list):
        raise ValueError(f"GT operation_sequence.json has no list skill_sequence: {gt_operation_sequence_path}")
    # Return a frozen case object so later stages cannot silently mutate the GT or prompt inputs.
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

    # Start from the canonical manifest order.
    selected = list(manifest)
    # A case-list file is useful for pilots and regression tests.
    if case_list_file is not None:
        # Empty lines are ignored so users can keep manually edited lists readable.
        requested = [line.strip() for line in case_list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        # Use a set only for membership, not for output order.
        requested_set = set(requested)
        # Preserve manifest order to keep outputs deterministic.
        selected = [case_id for case_id in selected if case_id in requested_set]
        # Report misspelled requested ids immediately.
        missing_requested = [case_id for case_id in requested if case_id not in set(selected)]
        # A wrong case id would otherwise make a pilot look smaller than intended.
        if missing_requested:
            raise ValueError({"missing_requested_case_ids": missing_requested})
    # MaxCases is used only for smoke tests and pilots.
    if max_cases > 0:
        selected = selected[:max_cases]
    # Return the filtered case id list.
    return selected
