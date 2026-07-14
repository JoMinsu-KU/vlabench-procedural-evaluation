"""Run VLM procedural plans through the Procedural Execution Interface.

This runner implements the procedural planning and native execution evaluation.
It does not score against operation_sequence.json.
It asks the VLM for a procedural JSON plan, validates the plan, binds numeric
targets from the numbered-mask image, executes the accepted plan in MuJoCo, and
stores the native success/failure evidence.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from procedural_interface.adapter.skillbook import skillbook_as_dict
from procedural_interface.core.io import ensure_dir, write_json
from procedural_interface.dataset.public_view_loader import SampleRef
from procedural_interface.evaluation.runner import (
    MAX_NATIVE_RETRY_ATTEMPTS,
    NATIVE_RETRY_POLICY,
    _bind_fixture_plan,
    _reset_wait_step_for_task,
    _status_from_native,
    _summarize_native_attempt,
)
from procedural_interface.execution.native import run_native_execution
from vlabench_official_eval.dataset.case_loader import load_manifest, select_case_ids, split_case_id


DATASET_SNAPSHOT = "vlabench_vlm4500_public_2133_matched"
FIXTURE_VERSION = "model_procedural_v1_no_gt_no_role"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the procedural VLM execution experiment."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "datasets" / "manifests" / "matched_2133_symbolic_procedural_cases.json")
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "procedural_eval" / "run")
    parser.add_argument("--case-list-file", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--native-timeout-sec", type=int, default=240)
    parser.add_argument("--native-max-attempts", type=int, default=MAX_NATIVE_RETRY_ATTEMPTS)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--response-format-json", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument("--inference-only", action="store_true")
    return parser.parse_args()


def image_data_url(image_path: Path) -> str:
    """Encode one PNG as an OpenAI-compatible data URL."""

    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    return "data:image/png;base64," + image_b64


def build_procedural_prompt(*, instruction: str) -> str:
    """Build the model-facing procedural planning prompt."""

    return """In Skill Lab, we provide a series of robot skills to support efficient automated operations for various tasks. Each skill has a specific call format, which includes the skill name and corresponding parameters.

The available skills are as follows:

1. **pick**: Used to grasp and pick up a target object.
   - Call format:
     ```json
     {
       "skill": "pick",
       "target": Target Number
     }
     ```

2. **lift**: Lift the robotic arm vertically.
   - Call format:
     ```json
     {
       "skill": "lift"
     }
     ```

3. **moveto**: Move the robotic arm to a specified target object or location.
   - Call format:
     ```json
     {
       "skill": "moveto",
       "target": Target Number
     }
     ```

4. **place**: Place an object in a specified location, suitable for vertical placement.
   - Call format:
     ```json
     {
       "skill": "place",
       "target": Target Number
     }
     ```

5. **pull**: Pull the robotic arm horizontally.
   - Call format:
     ```json
     {
       "skill": "pull"
     }
     ```

6. **pour**: Pour a liquid or granular substance.
   - Call format:
     ```json
     {
       "skill": "pour"
     }
     ```

7. **press**: Press a specified location or button.
   - Call format:
     ```json
     {
       "skill": "press",
       "target": Target Number
     }
     ```

8. **push**: Push a target object horizontally.
   - Call format:
     ```json
     {
       "skill": "push",
       "target": Target Number
     }
     ```

9. **wait**: Wait for a task-defined duration.
   - Call format:
     ```json
     {
       "skill": "wait"
     }
     ```

10. **move_offset**: Move the robotic arm by a task-defined offset.
   - Call format:
     ```json
     {
       "skill": "move_offset"
     }
     ```

11. **open_gripper**: Open the robotic gripper.
   - Call format:
     ```json
     {
       "skill": "open_gripper"
     }
     ```

These call formats ensure that each skill operation has clearly defined parameters, allowing the system to accurately execute the specified automated tasks.

---

You will receive the following input:

1. **Image input**: Two images
   - The first image shows four different perspectives of objects (without labels).
   - The second image contains the same four perspectives of objects, but each view is labeled with a number (representing each object's identifier).

2. **Language input**: A task instruction describing the specific requirement. Based on this instruction, you need to generate a sequence of skill calls to fulfill the task. Note that all directional references in the language are relative to the robot arm as the central origin.

---


Based on the image and language inputs, generate a sequence of skill calls. Each skill call sequence should contain the skill name (extracted from the task instruction), the skill operation parameters (if the skill requires parameters), and the target entity or container number (obtained from the labeled image).

---


1. **Extract Task Instruction**: Identify operation requirements (e.g., adjust, inspect, move) and the target object number or view from the language input.

2. **Combine with Image Information**: Using the labeled image, match the objects described in the task instruction with their corresponding identifiers to determine the target number.

3. **Generate Skill Call Sequence**: Based on the task instruction, use the extracted operation requirements as skill names and assign them the appropriate number information.

4. **Output Format**: Generate a JSON object in the following structure:
```json
{
    "steps": [
        {
            "skill": "Skill Name 1",
            "target": Target Number
        },
        {
            "skill": "Skill Name 2"
        }
    ]
}
```

For skills that require parameters, use `"target": Target Number`. For skills that do not require parameters, omit the target field.

Since the evaluation process will validate and execute the skill sequence, the output skill call sequence should ideally match one or more of the following patterns to ensure correctness:

Sub-skill sequence patterns:
- ["pick", "place"]
- ["pick", "lift"]
- ["pick", "pull"]
- ["pick", "lift", "place"]
- ["pick", "lift", "moveto", "lift"]
- ["pick", "lift", "moveto", "pour"]
- ["pick", "lift", "moveto", "push"]
- ["pick", "lift", "moveto", "push", "pull"]
- ["pick", "lift", "moveto", "pour", "wait", "pull", "pour", "moveto", "move_offset", "open_gripper"]
- ["press"]

For example, if the output only includes a single "pick" skill, it will be considered incorrect during the evaluation.
Please pay close attention to spatial information and ensure accurate pattern selection. For example, if the object needs to be picked up and displayed, use the pattern `["pick", "lift"]`. If the object needs to be placed directly into a specified location, use the pattern `["pick", "place"]`. If the object needs to be inserted into a target location, use the pattern `["pick", "lift", "moveto", "lift"]`.
"""


def build_messages(*, input_image_path: Path, input_mask_path: Path, instruction: str) -> list[dict[str, Any]]:
    """Build the multimodal request in the same image order as official VLABench."""

    content: list[dict[str, Any]] = [{"type": "text", "text": build_procedural_prompt(instruction=instruction)}]
    content.append({"type": "text", "text": "Input picture"})
    content.append({"type": "image_url", "image_url": {"url": image_data_url(input_image_path)}})
    content.append({"type": "text", "text": "Input picture with numbered tags"})
    content.append({"type": "image_url", "image_url": {"url": image_data_url(input_mask_path)}})
    content.append({"type": "text", "text": "Language instruction:"})
    content.append({"type": "text", "text": instruction})
    content.append({"type": "text", "text": "Please give the output skill sequence"})
    return [{"role": "user", "content": content}]


def call_openai_compatible_vlm(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    response_format_json: bool,
) -> dict[str, Any]:
    """Call a vLLM/OpenAI-compatible chat-completions endpoint."""

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        raw_body = response.read().decode("utf-8")
    parsed_body = json.loads(raw_body)
    content = parsed_body.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {
        "ok": True,
        "elapsed_sec": time.time() - started,
        "content": content,
        "raw_response": parsed_body,
        "usage": parsed_body.get("usage"),
        "served_model": parsed_body.get("model"),
    }


def extract_json_object_text(raw_text: str) -> tuple[str, str]:
    """Extract a JSON-object candidate without repairing semantic content."""

    direct = raw_text.strip()
    if direct.startswith("{") and direct.endswith("}"):
        return direct, "direct_json_object"
    fenced_matches = re.findall(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    for match in fenced_matches:
        candidate = match.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate, "fenced_json_object"
    start = direct.find("{")
    end = direct.rfind("}")
    if 0 <= start < end:
        return direct[start : end + 1], "brace_extracted_json_object"
    return raw_text, "no_json_object_extracted"


def normalize_integer_string_targets(json_text: str) -> dict[str, Any]:
    """Cast only exact integer-string target values to integers."""

    policy = "integer-string-target-only"
    result: dict[str, Any] = {
        "ok": False,
        "policy": policy,
        "target_numeric_cast_applied": False,
        "semantic_repair_applied": False,
        "normalized_text": json_text,
        "raw_steps": None,
        "normalized_steps": None,
        "error": "",
    }
    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        result["error"] = str(exc)
        return result
    if not isinstance(payload, dict):
        result["error"] = "top_level_not_object"
        return result
    steps = payload.get("steps")
    if not isinstance(steps, list):
        result["error"] = "steps_not_list"
        return result
    result["raw_steps"] = steps
    normalized_payload = dict(payload)
    normalized_steps: list[Any] = []
    cast_applied = False
    for step in steps:
        if not isinstance(step, dict):
            normalized_steps.append(step)
            continue
        normalized_step = dict(step)
        if isinstance(normalized_step.get("target"), str) and re.fullmatch(r"\d+", normalized_step["target"].strip()):
            normalized_step["target"] = int(normalized_step["target"].strip())
            cast_applied = True
        normalized_steps.append(normalized_step)
    normalized_payload["steps"] = normalized_steps
    normalized_text = json.dumps(normalized_payload, ensure_ascii=False)
    result.update(
        {
            "ok": True,
            "target_numeric_cast_applied": cast_applied,
            "normalized_text": normalized_text,
            "normalized_steps": normalized_steps,
        }
    )
    return result


def split_public_case(public_root: Path, case_id: str) -> dict[str, Any]:
    """Resolve public input paths for one case id without reading GT files."""

    category, task_name, example_name, example_index = split_case_id(case_id)
    case_root = public_root / category / task_name / example_name
    input_image_path = case_root / "input.png"
    input_mask_path = case_root / "input_mask.png"
    instruction_path = case_root / "instruction.txt"
    env_config_path = case_root / "env_config.json"
    missing = [str(path) for path in [input_image_path, input_mask_path, instruction_path, env_config_path] if not path.exists()]
    if missing:
        raise FileNotFoundError({"case_id": case_id, "missing_paths": missing})
    return {
        "case_id": case_id,
        "category": category,
        "task_name": task_name,
        "example_name": example_name,
        "example_index": example_index,
        "case_root": case_root,
        "input_image_path": input_image_path,
        "input_mask_path": input_mask_path,
        "instruction": instruction_path.read_text(encoding="utf-8").strip(),
    }


def fixture_from_model_plan(*, case: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Wrap normalized VLM steps in a model-evaluation fixture."""

    safe_case_id = case["case_id"].replace("/", "__")
    return {
        "fixture_id": f"model_eval__{safe_case_id}",
        "fixture_version": FIXTURE_VERSION,
        "fixture_mode": "model_evaluation",
        "family": case["task_name"],
        "negative_type": "",
        "fixture_generated_from_gt": False,
        "fixture_generation_phase": "model_inference",
        "source_fields_used": ["input.png", "input_mask.png", "instruction.txt", "vlm_response"],
        "frozen_after_generation": True,
        "dataset_snapshot": DATASET_SNAPSHOT,
        "source_case_id": case["case_id"],
        "target_source": "vlm_numbered_mask",
        "counts_as_model_evaluation": True,
        "native_execution_should_run": True,
        "expected_gate1_ok": None,
        "expected_native_ok": None,
        "expected_condition_success": None,
        "steps": steps,
        "human_readable_targets": {},
    }


def row_output_path(output_root: Path, case_id: str) -> Path:
    """Map a case id to a stable per-case JSON path."""

    safe_case_id = case_id.replace("/", "__")
    return output_root / "case_outputs" / f"{safe_case_id}.json"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSONL row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact procedural evaluation CSV."""

    fieldnames = [
        "case_id",
        "category",
        "task_name",
        "example_name",
        "ok",
        "prompt_only",
        "model_parse_ok",
        "strict_parse_ok",
        "type_normalized_parse_ok",
        "target_numeric_cast_applied",
        "normalization_policy",
        "semantic_repair_applied",
        "json_extraction_status",
        "actual_gate1_ok",
        "native_execution_attempted",
        "actual_native_ok",
        "actual_condition_success",
        "procedural_success",
        "actual_failure_code",
        "actual_failure_subtype",
        "actual_failure_detail",
        "native_attempt_count",
        "native_selected_attempt",
        "elapsed_sec",
        "served_model",
        "error_type",
        "error_detail",
        "trace_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def should_retry_native_for_model(native_result: dict[str, Any] | None) -> bool:
    """Retry only same-plan completed rollouts with condition false."""

    if native_result is None:
        return False
    if not native_result.get("ok"):
        return False
    if native_result.get("condition_success"):
        return False
    return True


def execute_model_fixture(*, public_root: Path, fixture: dict[str, Any], trace_root: Path, timeout_sec: int, native_max_attempts: int) -> dict[str, Any]:
    """Execute one model-generated fixture through binding and native execution."""

    category, task_name, example_id, _ = split_case_id(fixture["source_case_id"])
    sample = SampleRef(category=category, task_name=task_name, example_id=example_id, root=public_root / category / task_name / example_id)
    fixture_trace_root = ensure_dir(trace_root / fixture["fixture_id"])
    write_json(fixture_trace_root / "fixture.json", fixture)
    gate_result = _bind_fixture_plan(sample, fixture)
    write_json(fixture_trace_root / "gate1_phase_result.json", gate_result)
    native_attempted = False
    native_result: dict[str, Any] | None = None
    native_attempt_summaries: list[dict[str, Any]] = []
    selected_native_attempt = 0
    if gate_result.get("ok"):
        native_attempted = True
        native_command = {
            "task_name": sample.task_name,
            "env_config_path": str(sample.env_config_path),
            "executable_steps": gate_result["executable_steps"],
            "reset_wait_step": _reset_wait_step_for_task(sample.task_name),
            "native_seed": None,
            "visual_audit_dir": str(fixture_trace_root / "visual_audit"),
        }
        write_json(fixture_trace_root / "native_command.json", native_command)
        for attempt_index in range(1, max(1, native_max_attempts) + 1):
            attempt_root = fixture_trace_root / f"execution_attempt_{attempt_index:02d}"
            attempt_result = run_native_execution(native_command, attempt_root, timeout_sec=timeout_sec)
            write_json(fixture_trace_root / f"native_result_attempt_{attempt_index:02d}.json", attempt_result)
            native_attempt_summaries.append(_summarize_native_attempt(attempt_index, attempt_result))
            native_result = attempt_result
            selected_native_attempt = attempt_index
            if not should_retry_native_for_model(attempt_result):
                break
        write_json(fixture_trace_root / "native_result.json", native_result)
    status = _status_from_native(gate_result, native_result)
    procedural_success = bool(gate_result.get("ok")) and bool(native_result.get("ok")) if native_result else False
    procedural_success = procedural_success and bool(native_result.get("condition_success")) if native_result else False
    row = {
        "fixture_id": fixture["fixture_id"],
        "family": fixture["family"],
        "fixture_mode": fixture["fixture_mode"],
        "negative_type": "",
        "source_case_id": fixture["source_case_id"],
        "fixture_generated_from_gt": False,
        "gt_used_during_execution": False,
        "adapter_read_conditions": False,
        "lowerer_read_conditions": False,
        "operation_sequence_read": False,
        "expert_sequence_called": False,
        "target_source": fixture["target_source"],
        "counts_as_model_evaluation": True,
        "native_execution_attempted": native_attempted,
        "native_execution_should_run": True,
        "native_retry_policy": NATIVE_RETRY_POLICY if native_attempted else "",
        "native_max_attempts": max(1, native_max_attempts) if native_attempted else 0,
        "native_attempt_count": len(native_attempt_summaries),
        "native_selected_attempt": selected_native_attempt,
        "native_attempt_summaries": json.dumps(native_attempt_summaries, ensure_ascii=False),
        "expected_gate1_ok": None,
        "expected_native_ok": None,
        "expected_condition_success": None,
        "actual_gate1_ok": bool(gate_result.get("ok")),
        "actual_native_ok": bool(native_result.get("ok")) if native_result else False,
        "actual_condition_success": bool(native_result.get("condition_success")) if native_result else False,
        "procedural_success": procedural_success,
        "actual_failure_code": status["code"],
        "actual_failure_subtype": status["subtype"],
        "actual_failure_detail": status["detail"],
        "exception_type": native_result.get("exception_type", "") if native_result else "",
        "episode_config_added_fields": json.dumps(native_result.get("episode_config_audit", {}).get("added_fields", {}), ensure_ascii=False) if native_result else "",
        "before_contact_sheet": native_result.get("visual_audit", {}).get("before", {}).get("contact_sheet", "") if native_result else "",
        "after_contact_sheet": native_result.get("visual_audit", {}).get("after", {}).get("contact_sheet", "") if native_result else "",
        "passed": procedural_success,
        "trace_dir": str(fixture_trace_root),
    }
    write_json(fixture_trace_root / "result_summary.json", row)
    return row


def run_one_case(*, case_id: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run one case through VLM procedural inference and optional native execution."""

    case = split_public_case(args.public_root, case_id)
    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "category": case["category"],
        "task_name": case["task_name"],
        "example_name": case["example_name"],
        "example_index": case["example_index"],
        "input_image_path": str(case["input_image_path"]),
        "input_mask_path": str(case["input_mask_path"]),
        "gt_used_for_prompt": False,
        "counts_as_model_evaluation": True,
        "prompt_only": args.prompt_only,
        "inference_only": args.inference_only,
    }
    messages = build_messages(input_image_path=case["input_image_path"], input_mask_path=case["input_mask_path"], instruction=case["instruction"])
    row["message_block_count"] = len(messages[0]["content"])
    prompt_path = args.output_root / "prompts" / f"{case_id.replace('/', '__')}.txt"
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    prompt_path.write_text(messages[0]["content"][0]["text"], encoding="utf-8")
    if args.prompt_only:
        row.update({"ok": True, "model_parse_ok": False, "json_extraction_status": "prompt_only", "procedural_success": False})
        return row
    response = call_openai_compatible_vlm(
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        response_format_json=args.response_format_json,
    )
    row["elapsed_sec"] = response["elapsed_sec"]
    row["served_model"] = response.get("served_model")
    row["usage"] = response.get("usage")
    row["raw_response"] = response.get("raw_response")
    raw_content = response["content"]
    row["origin_output"] = raw_content
    json_text, extraction_status = extract_json_object_text(raw_content)
    row["json_extraction_status"] = extraction_status
    from procedural_interface.interface.parser import parse_plan_text

    strict_parsed = parse_plan_text(json_text)
    row["strict_parsed_plan"] = strict_parsed
    row["strict_parse_ok"] = bool(strict_parsed.get("ok"))
    normalization = normalize_integer_string_targets(json_text)
    row["normalization_policy"] = normalization["policy"]
    row["target_numeric_cast_applied"] = bool(normalization.get("target_numeric_cast_applied"))
    row["semantic_repair_applied"] = False
    row["raw_steps"] = normalization.get("raw_steps")
    row["normalized_steps"] = normalization.get("normalized_steps")
    normalized_parsed = parse_plan_text(str(normalization.get("normalized_text", json_text)))
    row["type_normalized_parsed_plan"] = normalized_parsed
    row["type_normalized_parse_ok"] = bool(normalized_parsed.get("ok"))
    parsed = strict_parsed if strict_parsed.get("ok") else normalized_parsed
    row["parsed_plan"] = parsed
    row["model_parse_ok"] = bool(parsed.get("ok"))
    row["execution_used_type_normalized_plan"] = bool((not strict_parsed.get("ok")) and normalized_parsed.get("ok"))
    if not parsed.get("ok"):
        status = parsed["status"]
        row.update(
            {
                "ok": True,
                "actual_gate1_ok": False,
                "native_execution_attempted": False,
                "actual_native_ok": False,
                "actual_condition_success": False,
                "procedural_success": False,
                "actual_failure_code": status["code"],
                "actual_failure_subtype": status["subtype"],
                "actual_failure_detail": status["detail"],
                "native_attempt_count": 0,
                "native_selected_attempt": 0,
                "trace_dir": "",
            }
        )
        return row
    if args.inference_only:
        row["normalized_steps"] = parsed["steps"]
        row.update(
            {
                "ok": True,
                "actual_gate1_ok": None,
                "native_execution_attempted": False,
                "actual_native_ok": False,
                "actual_condition_success": False,
                "procedural_success": False,
                "actual_failure_code": "",
                "actual_failure_subtype": "inference_only",
                "actual_failure_detail": "native execution skipped by --inference-only",
                "native_attempt_count": 0,
                "native_selected_attempt": 0,
                "trace_dir": "",
            }
        )
        return row
    fixture = fixture_from_model_plan(case=case, steps=parsed["steps"])
    row["fixture"] = fixture
    exec_row = execute_model_fixture(
        public_root=args.public_root,
        fixture=fixture,
        trace_root=args.output_root / "execution_traces",
        timeout_sec=args.native_timeout_sec,
        native_max_attempts=args.native_max_attempts,
    )
    row.update(exec_row)
    row["ok"] = True
    return row


def summarize(rows: list[dict[str, Any]], *, started_at: float, args: argparse.Namespace) -> dict[str, Any]:
    """Build aggregate procedural execution metrics."""

    total = len(rows)
    completed = sum(1 for row in rows if row.get("ok") is True)
    parse_ok = sum(1 for row in rows if row.get("model_parse_ok") is True)
    strict_parse_ok = sum(1 for row in rows if row.get("strict_parse_ok") is True)
    type_normalized_parse_ok = sum(1 for row in rows if row.get("type_normalized_parse_ok") is True)
    execution_used_type_normalized_plan = sum(1 for row in rows if row.get("execution_used_type_normalized_plan") is True)
    target_numeric_cast_applied = sum(1 for row in rows if row.get("target_numeric_cast_applied") is True)
    gate1_ok = sum(1 for row in rows if row.get("actual_gate1_ok") is True)
    native_attempted = sum(1 for row in rows if row.get("native_execution_attempted") is True)
    native_ok = sum(1 for row in rows if row.get("actual_native_ok") is True)
    procedural_success = sum(1 for row in rows if row.get("procedural_success") is True)
    runtime_errors = sum(1 for row in rows if row.get("ok") is not True)
    failure_codes: dict[str, int] = {}
    failure_subtypes: dict[str, int] = {}
    by_category: dict[str, dict[str, int]] = {}
    by_task: dict[str, dict[str, int]] = {}
    for row in rows:
        category = str(row.get("category", ""))
        task_key = f"{row.get('category', '')}/{row.get('task_name', '')}"
        by_category.setdefault(category, {"total": 0, "parse_ok": 0, "strict_parse_ok": 0, "type_normalized_parse_ok": 0, "target_numeric_cast_applied": 0, "gate1_ok": 0, "native_attempted": 0, "native_ok": 0, "procedural_success": 0})
        by_task.setdefault(task_key, {"total": 0, "parse_ok": 0, "strict_parse_ok": 0, "type_normalized_parse_ok": 0, "target_numeric_cast_applied": 0, "gate1_ok": 0, "native_attempted": 0, "native_ok": 0, "procedural_success": 0})
        by_category[category]["total"] += 1
        by_task[task_key]["total"] += 1
        for field, bucket_name in [
            ("model_parse_ok", "parse_ok"),
            ("strict_parse_ok", "strict_parse_ok"),
            ("type_normalized_parse_ok", "type_normalized_parse_ok"),
            ("target_numeric_cast_applied", "target_numeric_cast_applied"),
            ("actual_gate1_ok", "gate1_ok"),
            ("native_execution_attempted", "native_attempted"),
            ("actual_native_ok", "native_ok"),
            ("procedural_success", "procedural_success"),
        ]:
            if row.get(field) is True:
                by_category[category][bucket_name] += 1
                by_task[task_key][bucket_name] += 1
        code = str(row.get("actual_failure_code", ""))
        code = "C0" if row.get("procedural_success") is True else (code or "UNCLASSIFIED")
        failure_codes[code] = failure_codes.get(code, 0) + 1
        subtype_key = f"{code}:{row.get('actual_failure_subtype', '')}"
        failure_subtypes[subtype_key] = failure_subtypes.get(subtype_key, 0) + 1
    return {
        "status": "completed" if runtime_errors == 0 else "completed_with_runtime_errors",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": time.time() - started_at,
        "experiment": "B_procedural_execution_interface",
        "manifest": str(args.manifest),
        "public_root": str(args.public_root),
        "output_root": str(args.output_root),
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "response_format_json": args.response_format_json,
        "inference_only": args.inference_only,
        "native_timeout_sec": args.native_timeout_sec,
        "native_max_attempts": args.native_max_attempts,
        "native_retry_policy": NATIVE_RETRY_POLICY,
        "total_cases": total,
        "completed_rows": completed,
        "parse_ok": parse_ok,
        "parse_ok_rate": parse_ok / total if total else 0,
        "strict_parse_ok": strict_parse_ok,
        "strict_parse_ok_rate": strict_parse_ok / total if total else 0,
        "type_normalized_parse_ok": type_normalized_parse_ok,
        "type_normalized_parse_ok_rate": type_normalized_parse_ok / total if total else 0,
        "execution_used_type_normalized_plan": execution_used_type_normalized_plan,
        "target_numeric_cast_applied": target_numeric_cast_applied,
        "normalization_policy": "integer-string-target-only",
        "semantic_repair_applied": False,
        "gate1_ok": gate1_ok,
        "gate1_ok_rate": gate1_ok / total if total else 0,
        "native_attempted": native_attempted,
        "native_attempted_rate": native_attempted / total if total else 0,
        "native_ok": native_ok,
        "native_ok_rate": native_ok / total if total else 0,
        "procedural_success": procedural_success,
        "procedural_success_rate": procedural_success / total if total else 0,
        "runtime_errors": runtime_errors,
        "failure_codes": failure_codes,
        "failure_subtypes": failure_subtypes,
        "by_category": by_category,
        "by_task": by_task,
    }


def main() -> int:
    """Run the procedural VLM execution evaluator."""

    args = parse_args()
    started_at = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    selected_case_ids = select_case_ids(manifest=manifest, max_cases=args.max_cases, case_list_file=args.case_list_file)
    write_json(
        args.output_root / "run_config.json",
        {
            "selected_case_count": len(selected_case_ids),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "gt_used_for_prompt": False,
            "operation_sequence_used_for_scoring": False,
            "counts_as_model_evaluation": True,
            "normalization_policy": "integer-string-target-only",
            "semantic_repair_applied": False,
        },
    )
    rows: list[dict[str, Any]] = []
    jsonl_path = args.output_root / "procedural_vlm_execution_rows.jsonl"
    if jsonl_path.exists() and not args.resume:
        jsonl_path.unlink()
    for index, case_id in enumerate(selected_case_ids, start=1):
        case_output_path = row_output_path(args.output_root, case_id)
        if args.resume and case_output_path.exists():
            row = json.loads(case_output_path.read_text(encoding="utf-8"))
        else:
            try:
                row = run_one_case(case_id=case_id, args=args)
            except Exception as exc:  # noqa: BLE001 - long runs must preserve per-case failures.
                parts = (case_id.split("/") + ["", "", ""])[:3]
                row = {
                    "case_id": case_id,
                    "category": parts[0],
                    "task_name": parts[1],
                    "example_name": parts[2],
                    "ok": False,
                    "prompt_only": args.prompt_only,
                    "model_parse_ok": False,
                    "actual_gate1_ok": False,
                    "native_execution_attempted": False,
                    "actual_native_ok": False,
                    "actual_condition_success": False,
                    "procedural_success": False,
                    "actual_failure_code": "INFRA",
                    "actual_failure_subtype": "runner_exception",
                    "actual_failure_detail": str(exc),
                    "elapsed_sec": 0,
                    "error_type": type(exc).__name__,
                    "error_detail": str(exc),
                }
            write_json(case_output_path, row)
        rows.append(row)
        append_jsonl(jsonl_path, row)
        print(
            json.dumps(
                {
                    "index": index,
                    "total": len(selected_case_ids),
                    "case_id": case_id,
                    "ok": row.get("ok"),
                    "parse_ok": row.get("model_parse_ok"),
                    "gate1_ok": row.get("actual_gate1_ok"),
                    "native_attempted": row.get("native_execution_attempted"),
                    "native_ok": row.get("actual_native_ok"),
                    "condition_success": row.get("actual_condition_success"),
                    "failure_code": row.get("actual_failure_code"),
                    "failure_subtype": row.get("actual_failure_subtype"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.save_interval > 0 and index % args.save_interval == 0:
            write_csv(args.output_root / "procedural_vlm_execution_results.csv", rows)
            write_json(args.output_root / "summary.json", summarize(rows, started_at=started_at, args=args))
    write_csv(args.output_root / "procedural_vlm_execution_results.csv", rows)
    summary = summarize(rows, started_at=started_at, args=args)
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
