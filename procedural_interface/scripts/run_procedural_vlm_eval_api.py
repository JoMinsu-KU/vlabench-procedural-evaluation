"""Run VLM procedural plans through the Procedural Execution Interface.

This runner is Experiment B.
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
from procedural_interface.evaluation.oracle_ablations import (
    ABLATION_MODES,
    FULL_MODE,
    PHASE_ORACLE_MODE,
    TARGET_ORACLE_MODE,
    build_oracle_prompt,
    resolve_phase_oracle_output,
    resolve_target_oracle_output,
)
from procedural_interface.execution.native import run_native_execution
from vlabench_official_eval.dataset.case_loader import load_manifest, select_case_ids, split_case_id


DATASET_SNAPSHOT = "vlabench_vlm4500_public_2133_matched"
FIXTURE_VERSION = "model_procedural_v1_no_gt_no_role"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for the procedural VLM execution experiment."""

    # Keep this runner discoverable from procedural_interface/scripts.
    parser = argparse.ArgumentParser(description=__doc__)
    # The manifest is the shared 2,133-case list used by Experiment A and B.
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "datasets" / "manifests" / "matched_2133_symbolic_procedural_cases.json")
    # The public root contains input.png, input_mask.png, instruction.txt, and env_config.json.
    parser.add_argument("--public-root", type=Path, required=True)
    # Output root stores VLM raw output, parsed plan, native traces, CSV, JSONL, and summary.
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "procedural_eval" / "run")
    # Optional pilot list keeps full-run and pilot selection aligned with Experiment A.
    parser.add_argument("--case-list-file", type=Path, default=None)
    # Max cases is used for one-case smoke tests before pilot/full execution.
    parser.add_argument("--max-cases", type=int, default=0)
    # OpenAI-compatible endpoint exposed by the active vLLM server.
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    # Served model id; it must match /v1/models on the active vLLM server.
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    # Oracle ablations are explicit conditions; full preserves the original B path.
    parser.add_argument("--ablation-mode", choices=ABLATION_MODES, default=FULL_MODE)
    # Greedy decoding is the default single-plan protocol.
    parser.add_argument("--temperature", type=float, default=0.0)
    # Procedural plans can be longer than official symbolic outputs, especially complex tasks.
    parser.add_argument("--max-tokens", type=int, default=1024)
    # HTTP timeout per VLM request.
    parser.add_argument("--timeout-sec", type=int, default=180)
    # Native timeout per MuJoCo execution attempt.
    parser.add_argument("--native-timeout-sec", type=int, default=240)
    # Same-plan native retry count absorbs stochastic motion-planner failures without changing VLM output.
    parser.add_argument("--native-max-attempts", type=int, default=MAX_NATIVE_RETRY_ATTEMPTS)
    # Save interval writes summary and CSV snapshots during long runs.
    parser.add_argument("--save-interval", type=int, default=1)
    # JSON response mode is optional because some OpenAI-compatible servers implement it differently.
    parser.add_argument("--response-format-json", action="store_true")
    # Resume skips case outputs that already exist.
    parser.add_argument("--resume", action="store_true")
    # Prompt-only mode writes prompts and fixtures without API/native execution for inspection.
    parser.add_argument("--prompt-only", action="store_true")
    # Inference-only mode calls the VLM and parser but skips MuJoCo/native execution.
    parser.add_argument("--inference-only", action="store_true")
    # Return parsed arguments.
    return parser.parse_args()


def image_data_url(image_path: Path) -> str:
    """Encode one PNG as an OpenAI-compatible data URL."""

    # Read the exact public-view image bytes from disk.
    image_bytes = image_path.read_bytes()
    # Encode bytes as ASCII base64 for JSON transport.
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    # vLLM accepts image_url content blocks with data URLs.
    return "data:image/png;base64," + image_b64


def build_procedural_prompt(*, instruction: str) -> str:
    """Build the model-facing procedural planning prompt."""

    # Keep the prompt structure close to the official VLABench VLM prompt.
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

### Task Requirements:

Based on the image and language inputs, generate a sequence of skill calls. Each skill call sequence should contain the skill name (extracted from the task instruction), the skill operation parameters (if the skill requires parameters), and the target entity or container number (obtained from the labeled image).

---

### Generation Steps:

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


def build_condition_prompt(*, instruction: str, task_name: str, ablation_mode: str) -> str:
    """Select the unchanged B prompt or one explicit oracle contract."""

    # Build the original full-contract prompt exactly as before.
    full_prompt = build_procedural_prompt(instruction=instruction)
    # Preserve the full Experiment B prompt byte-for-byte in the default mode.
    if ablation_mode == FULL_MODE:
        return full_prompt
    # Reuse the identical 11-skill descriptions before the first separator.
    skillbook_header = full_prompt.split("\n---\n", maxsplit=1)[0]
    # Build the selected oracle contract without exposing case-specific target GT.
    return build_oracle_prompt(mode=ablation_mode, task_name=task_name, skillbook_header=skillbook_header)


def build_messages(*, input_image_path: Path, input_mask_path: Path, instruction: str, task_name: str, ablation_mode: str) -> list[dict[str, Any]]:
    """Build the multimodal request in the same image order as official VLABench."""

    # Start with the procedural prompt and schema contract.
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": build_condition_prompt(instruction=instruction, task_name=task_name, ablation_mode=ablation_mode),
        }
    ]
    # Provide the unnumbered scene image first, matching the official VLM evaluation order.
    content.append({"type": "text", "text": "Input picture"})
    # Attach the raw input image as a base64 data URL.
    content.append({"type": "image_url", "image_url": {"url": image_data_url(input_image_path)}})
    # Provide the numbered image second so the model can map targets to integers.
    content.append({"type": "text", "text": "Input picture with numbered tags"})
    # Attach the numbered mask image as a base64 data URL.
    content.append({"type": "image_url", "image_url": {"url": image_data_url(input_mask_path)}})
    # Match the official VLABench message label before the task instruction.
    content.append({"type": "text", "text": "Language instruction:"})
    # Add the model-visible instruction.
    content.append({"type": "text", "text": instruction})
    # Match the official final request wording as closely as possible.
    content.append({"type": "text", "text": "Please give the output skill sequence"})
    # Return one user message for OpenAI-compatible chat completions.
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

    # Build the OpenAI-compatible payload with deterministic default decoding.
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Request JSON mode only when the caller explicitly enables it.
    if response_format_json:
        # vLLM implements the OpenAI-compatible JSON object response mode for supported models.
        payload["response_format"] = {"type": "json_object"}
    # Create a dependency-light HTTP request.
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Record elapsed time for throughput and timeout diagnosis.
    started = time.time()
    # Let transport exceptions bubble to the per-case error handler.
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        # Decode the OpenAI-compatible response body.
        raw_body = response.read().decode("utf-8")
    # Parse the JSON response.
    parsed_body = json.loads(raw_body)
    # Extract the first-choice message content.
    content = parsed_body.get("choices", [{}])[0].get("message", {}).get("content", "")
    # Return both compact metadata and full raw response for auditability.
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

    # Direct text is the first candidate because JSON mode should return plain JSON.
    direct = raw_text.strip()
    # Direct JSON object is accepted as-is.
    if direct.startswith("{") and direct.endswith("}"):
        # Return the candidate and extraction status.
        return direct, "direct_json_object"
    # Look for fenced JSON because some models ignore the no-markdown instruction.
    fenced_matches = re.findall(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    # Try fenced candidates in order.
    for match in fenced_matches:
        # Strip whitespace around the fenced payload.
        candidate = match.strip()
        # Accept only object-shaped JSON text.
        if candidate.startswith("{") and candidate.endswith("}"):
            # Return the fenced candidate without changing its fields.
            return candidate, "fenced_json_object"
    # As a last syntactic extraction, isolate the first {...} block when present.
    start = direct.find("{")
    # Find the last closing brace so nested step objects remain included.
    end = direct.rfind("}")
    # Use this only when both braces exist in the right order.
    if 0 <= start < end:
        # Return the extracted object text without adding or changing any fields.
        return direct[start : end + 1], "brace_extracted_json_object"
    # Return raw text so the strict parser records an invalid_json failure.
    return raw_text, "no_json_object_extracted"


def normalize_integer_string_targets(json_text: str) -> dict[str, Any]:
    """Cast only exact integer-string target values to integers."""

    # The normalization policy intentionally allows only "2" -> 2 style target casts.
    policy = "integer-string-target-only"
    # Start with a conservative failure result.
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
    # Try to parse the extracted JSON object without repairing syntax.
    try:
        # Load the model JSON candidate.
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        # Keep invalid JSON as a strict parser failure.
        result["error"] = str(exc)
        # Return without modification.
        return result
    # Only the expected object shape can be normalized.
    if not isinstance(payload, dict):
        # Record why no normalization happened.
        result["error"] = "top_level_not_object"
        # Return without modification.
        return result
    # Read the model steps.
    steps = payload.get("steps")
    # Only a list of step objects can be normalized.
    if not isinstance(steps, list):
        # Record why no normalization happened.
        result["error"] = "steps_not_list"
        # Return without modification.
        return result
    # Preserve raw steps for audit before any cast.
    result["raw_steps"] = steps
    # Create a normalized top-level object without adding/removing top-level fields.
    normalized_payload = dict(payload)
    # Create a fresh normalized steps list.
    normalized_steps: list[Any] = []
    # Track whether any target cast happened.
    cast_applied = False
    # Iterate over every model-emitted step.
    for step in steps:
        # Non-object steps are left untouched so the parser can reject them.
        if not isinstance(step, dict):
            # Append the original malformed step.
            normalized_steps.append(step)
            # Continue without semantic repair.
            continue
        # Copy the step without adding/removing skill/order fields.
        normalized_step = dict(step)
        # Cast only when the key is literally target and the value is an exact integer string.
        if isinstance(normalized_step.get("target"), str) and re.fullmatch(r"\d+", normalized_step["target"].strip()):
            # Convert the integer-like string to an int.
            normalized_step["target"] = int(normalized_step["target"].strip())
            # Record that type-level normalization was applied.
            cast_applied = True
        # Append the normalized step.
        normalized_steps.append(normalized_step)
    # Store normalized steps under the original steps key.
    normalized_payload["steps"] = normalized_steps
    # Serialize normalized payload for the strict parser.
    normalized_text = json.dumps(normalized_payload, ensure_ascii=False)
    # Complete the normalization result.
    result.update(
        {
            "ok": True,
            "target_numeric_cast_applied": cast_applied,
            "normalized_text": normalized_text,
            "normalized_steps": normalized_steps,
        }
    )
    # Return normalization audit.
    return result


def split_public_case(public_root: Path, case_id: str) -> dict[str, Any]:
    """Resolve public input paths for one case id without reading GT files."""

    # Split the stable Category/task/exampleN case id.
    category, task_name, example_name, example_index = split_case_id(case_id)
    # Build the public case root.
    case_root = public_root / category / task_name / example_name
    # Resolve model-visible raw image.
    input_image_path = case_root / "input.png"
    # Resolve model-visible numbered-mask image.
    input_mask_path = case_root / "input_mask.png"
    # Resolve model-visible instruction text.
    instruction_path = case_root / "instruction.txt"
    # Resolve internal env_config used only by target binding/native execution, never prompt.
    env_config_path = case_root / "env_config.json"
    # Check required files early so dataset errors do not look like model failures.
    missing = [str(path) for path in [input_image_path, input_mask_path, instruction_path, env_config_path] if not path.exists()]
    # Raise a structured error if the public case is incomplete.
    if missing:
        raise FileNotFoundError({"case_id": case_id, "missing_paths": missing})
    # Return a small dictionary instead of a GT-bearing OfficialVLMCase object.
    return {
        "case_id": case_id,
        "category": category,
        "task_name": task_name,
        "example_name": example_name,
        "example_index": example_index,
        "case_root": case_root,
        "input_image_path": input_image_path,
        "input_mask_path": input_mask_path,
        "env_config_path": env_config_path,
        "instruction": instruction_path.read_text(encoding="utf-8").strip(),
    }


def fixture_from_model_plan(
    *,
    case: dict[str, Any],
    steps: list[dict[str, Any]],
    ablation_mode: str = FULL_MODE,
    human_readable_targets: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Wrap normalized VLM steps in a model-evaluation fixture."""

    # Create a path-safe fixture id from the case id.
    safe_case_id = case["case_id"].replace("/", "__")
    # Target oracle is the only condition that inserts case-specific reference targets.
    target_oracle = ablation_mode == TARGET_ORACLE_MODE
    # Phase oracle is the only condition that fixes phase/order before model inference.
    phase_oracle = ablation_mode == PHASE_ORACLE_MODE
    # Return the same fixture shape consumed by the procedural adapter runner.
    return {
        "fixture_id": f"model_eval__{safe_case_id}" if ablation_mode == FULL_MODE else f"model_eval__{ablation_mode}__{safe_case_id}",
        "fixture_version": FIXTURE_VERSION,
        "fixture_mode": "model_evaluation",
        "family": case["task_name"],
        "negative_type": "",
        "fixture_generated_from_gt": bool(target_oracle or phase_oracle),
        "fixture_generation_phase": f"model_inference_{ablation_mode}",
        "source_fields_used": (
            ["input.png", "input_mask.png", "instruction.txt", "vlm_response", "conditions", "components"]
            if target_oracle
            else ["input.png", "input_mask.png", "instruction.txt", "vlm_response", "task_phase_registry"]
            if phase_oracle
            else ["input.png", "input_mask.png", "instruction.txt", "vlm_response"]
        ),
        "frozen_after_generation": True,
        "dataset_snapshot": DATASET_SNAPSHOT,
        "source_case_id": case["case_id"],
        "target_source": "oracle_conditions_components" if target_oracle else "vlm_numbered_mask",
        "phase_source": "benchmark_derived_task_registry" if phase_oracle else "vlm_output",
        "ablation_mode": ablation_mode,
        "phase_oracle_used": phase_oracle,
        "target_oracle_used": target_oracle,
        "ground_truth_target_used": target_oracle,
        "oracle_used_during_execution": bool(target_oracle or phase_oracle),
        "oracle_builder_read_conditions": target_oracle,
        "counts_as_model_evaluation": True,
        "native_execution_should_run": True,
        "expected_gate1_ok": None,
        "expected_native_ok": None,
        "expected_condition_success": None,
        "steps": steps,
        "human_readable_targets": {str(number): name for number, name in sorted((human_readable_targets or {}).items())},
    }


def row_output_path(output_root: Path, case_id: str) -> Path:
    """Map a case id to a stable per-case JSON path."""

    # Replace slash separators so Windows and POSIX paths stay portable.
    safe_case_id = case_id.replace("/", "__")
    # Store per-case evidence separately for resume and manual inspection.
    return output_root / "case_outputs" / f"{safe_case_id}.json"


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSONL row."""

    # Ensure the output directory exists before opening the file.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Append one compact JSON row for streaming inspection.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact procedural evaluation CSV."""

    # Keep the highest-signal fields in front for spreadsheet inspection.
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
    # Create the report directory.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write UTF-8 with BOM for Windows spreadsheet compatibility.
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        # Ignore nested audit payloads in this compact CSV.
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        # Write header once.
        writer.writeheader()
        # Write all rows.
        writer.writerows(rows)


def should_retry_native_for_model(native_result: dict[str, Any] | None) -> bool:
    """Retry only same-plan completed rollouts with condition false."""

    # Missing native output is infrastructure, not a stochastic C1 retry.
    if native_result is None:
        return False
    # Native exceptions and timeouts must remain visible as C5.
    if not native_result.get("ok"):
        return False
    # Stop as soon as the condition succeeds.
    if native_result.get("condition_success"):
        return False
    # A completed native rollout with condition false is the same-plan retry boundary.
    return True


def execute_model_fixture(*, public_root: Path, fixture: dict[str, Any], trace_root: Path, timeout_sec: int, native_max_attempts: int) -> dict[str, Any]:
    """Execute one model-generated fixture through binding and native execution."""

    # Split fixture source id into SampleRef fields.
    category, task_name, example_id, _ = split_case_id(fixture["source_case_id"])
    # Build the SampleRef used by binding and native command generation.
    sample = SampleRef(category=category, task_name=task_name, example_id=example_id, root=public_root / category / task_name / example_id)
    # Create a dedicated trace directory.
    fixture_trace_root = ensure_dir(trace_root / fixture["fixture_id"])
    # Save the exact model-evaluation fixture.
    write_json(fixture_trace_root / "fixture.json", fixture)
    # Run parser, affordance check, and task-specific phase binding.
    gate_result = _bind_fixture_plan(sample, fixture)
    # Save validation/binding trace before native execution.
    write_json(fixture_trace_root / "gate1_phase_result.json", gate_result)
    # Initialize native attempt state.
    native_attempted = False
    # Initialize native result as absent.
    native_result: dict[str, Any] | None = None
    # Store compact per-attempt summaries.
    native_attempt_summaries: list[dict[str, Any]] = []
    # Store selected attempt index.
    selected_native_attempt = 0
    # Execute only when parser/affordance/phase binding accepted the VLM plan.
    if gate_result.get("ok"):
        # Mark that native execution was attempted.
        native_attempted = True
        # Build the native command strictly from VLM-derived executable steps.
        native_command = {
            "task_name": sample.task_name,
            "env_config_path": str(sample.env_config_path),
            "executable_steps": gate_result["executable_steps"],
            "reset_wait_step": _reset_wait_step_for_task(sample.task_name),
            "native_seed": None,
            "visual_audit_dir": str(fixture_trace_root / "visual_audit"),
        }
        # Save the exact native command for audit.
        write_json(fixture_trace_root / "native_command.json", native_command)
        # Attempt the same native command up to the configured retry count.
        for attempt_index in range(1, max(1, native_max_attempts) + 1):
            # Store each attempt separately so evidence is never overwritten.
            attempt_root = fixture_trace_root / f"execution_attempt_{attempt_index:02d}"
            # Run the native worker.
            attempt_result = run_native_execution(native_command, attempt_root, timeout_sec=timeout_sec)
            # Save this attempt result at fixture root.
            write_json(fixture_trace_root / f"native_result_attempt_{attempt_index:02d}.json", attempt_result)
            # Append compact attempt summary.
            native_attempt_summaries.append(_summarize_native_attempt(attempt_index, attempt_result))
            # Select the latest attempt.
            native_result = attempt_result
            # Record selected attempt number.
            selected_native_attempt = attempt_index
            # Stop unless this is a same-plan C1 retry case.
            if not should_retry_native_for_model(attempt_result):
                break
        # Save the final selected native result.
        write_json(fixture_trace_root / "native_result.json", native_result)
    # Convert validation/native outcome into the procedural taxonomy.
    status = _status_from_native(gate_result, native_result)
    # Compute the model-evaluation success flag.
    procedural_success = bool(gate_result.get("ok")) and bool(native_result.get("ok")) if native_result else False
    # Require benchmark condition success for final procedural success.
    procedural_success = procedural_success and bool(native_result.get("condition_success")) if native_result else False
    # Build the row using the same audit fields as smoke reports plus model-evaluation fields.
    row = {
        "fixture_id": fixture["fixture_id"],
        "family": fixture["family"],
        "fixture_mode": fixture["fixture_mode"],
        "negative_type": "",
        "source_case_id": fixture["source_case_id"],
        "fixture_generated_from_gt": bool(fixture.get("fixture_generated_from_gt", False)),
        "gt_used_during_execution": bool(fixture.get("oracle_used_during_execution", False)),
        "oracle_used_during_execution": bool(fixture.get("oracle_used_during_execution", False)),
        "phase_oracle_used": bool(fixture.get("phase_oracle_used", False)),
        "target_oracle_used": bool(fixture.get("target_oracle_used", False)),
        "phase_source": fixture.get("phase_source", "vlm_output"),
        "ground_truth_target_used": bool(fixture.get("ground_truth_target_used", False)),
        "oracle_builder_read_conditions": bool(fixture.get("oracle_builder_read_conditions", False)),
        "adapter_read_conditions": False,
        "lowerer_read_conditions": False,
        "operation_sequence_read": False,
        "expert_sequence_called": False,
        "target_source": fixture["target_source"],
        "counts_as_model_evaluation": bool(fixture.get("counts_as_model_evaluation", True)),
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
    # Save row-level execution summary in the trace directory.
    write_json(fixture_trace_root / "result_summary.json", row)
    # Return the row.
    return row


def run_one_case(*, case_id: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run one case through VLM procedural inference and optional native execution."""

    # Load only public/model-visible inputs plus env_config path for execution.
    case = split_public_case(args.public_root, case_id)
    # Build a base row that remains useful when inference or parsing fails.
    row: dict[str, Any] = {
        "case_id": case["case_id"],
        "category": case["category"],
        "task_name": case["task_name"],
        "example_name": case["example_name"],
        "example_index": case["example_index"],
        "input_image_path": str(case["input_image_path"]),
        "input_mask_path": str(case["input_mask_path"]),
        "gt_used_for_prompt": False,
        "ablation_mode": args.ablation_mode,
        "phase_oracle_used_for_prompt": args.ablation_mode == PHASE_ORACLE_MODE,
        "target_oracle_used_for_prompt": False,
        "phase_source": "benchmark_derived_task_registry" if args.ablation_mode == PHASE_ORACLE_MODE else "vlm_output",
        "target_source": "oracle_conditions_components" if args.ablation_mode == TARGET_ORACLE_MODE else "vlm_numbered_mask",
        "counts_as_model_evaluation": True,
        "prompt_only": args.prompt_only,
        "inference_only": args.inference_only,
    }
    # Build the exact multimodal request.
    messages = build_messages(
        input_image_path=case["input_image_path"],
        input_mask_path=case["input_mask_path"],
        instruction=case["instruction"],
        task_name=case["task_name"],
        ablation_mode=args.ablation_mode,
    )
    # Store prompt metadata, not image bytes.
    row["message_block_count"] = len(messages[0]["content"])
    # Store text prompt for auditability.
    prompt_path = args.output_root / "prompts" / f"{case_id.replace('/', '__')}.txt"
    # Ensure prompt directory exists.
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    # Write only the text prompt, not base64 images.
    prompt_path.write_text(messages[0]["content"][0]["text"], encoding="utf-8")
    # If prompt-only is requested, skip API and native execution.
    if args.prompt_only:
        row.update({"ok": True, "model_parse_ok": False, "json_extraction_status": "prompt_only", "procedural_success": False})
        return row
    # Call the active VLM endpoint.
    response = call_openai_compatible_vlm(
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        response_format_json=args.response_format_json,
    )
    # Store raw response metadata.
    row["elapsed_sec"] = response["elapsed_sec"]
    # Store served model id for multi-backbone runs.
    row["served_model"] = response.get("served_model")
    # Store token usage when available.
    row["usage"] = response.get("usage")
    # Store full raw API response for audit.
    row["raw_response"] = response.get("raw_response")
    # Keep model content in the row for parser diagnosis.
    raw_content = response["content"]
    # Store raw model text.
    row["origin_output"] = raw_content
    # Extract object-shaped JSON without semantic repair.
    json_text, extraction_status = extract_json_object_text(raw_content)
    # Record extraction status.
    row["json_extraction_status"] = extraction_status
    # Import the original parser only for the unchanged full B condition.
    from procedural_interface.interface.parser import parse_plan_text

    # Keep oracle-only human-readable target names out of execution decisions.
    human_readable_targets: dict[int, str] = {}
    # Preserve the original full B parsing and normalization path exactly.
    if args.ablation_mode == FULL_MODE:
        # Run the no-role strict parser.
        strict_parsed = parse_plan_text(json_text)
        # Store strict parser result for audit.
        row["strict_parsed_plan"] = strict_parsed
        # Store strict parser flag.
        row["strict_parse_ok"] = bool(strict_parsed.get("ok"))
        # Apply only integer-string target normalization.
        normalization = normalize_integer_string_targets(json_text)
        # Store normalization policy.
        row["normalization_policy"] = normalization["policy"]
        # Store whether any numeric target cast happened.
        row["target_numeric_cast_applied"] = bool(normalization.get("target_numeric_cast_applied"))
        # Store semantic repair flag, which must remain false.
        row["semantic_repair_applied"] = False
        # Store raw step payload when available.
        row["raw_steps"] = normalization.get("raw_steps")
        # Store type-normalized step payload when available.
        row["normalized_steps"] = normalization.get("normalized_steps")
        # Parse normalized text only after the strict parse/audit is recorded.
        normalized_parsed = parse_plan_text(str(normalization.get("normalized_text", json_text)))
        # Store normalized parser result for audit.
        row["type_normalized_parsed_plan"] = normalized_parsed
        # Store normalized parser success flag.
        row["type_normalized_parse_ok"] = bool(normalized_parsed.get("ok"))
        # Use strict parsed plan when possible, otherwise use integer-string-normalized parsed plan.
        parsed = strict_parsed if strict_parsed.get("ok") else normalized_parsed
        # Store selected execution parser success flag.
        row["model_parse_ok"] = bool(parsed.get("ok"))
        # Record whether execution used normalized JSON rather than strict JSON.
        row["execution_used_type_normalized_plan"] = bool((not strict_parsed.get("ok")) and normalized_parsed.get("ok"))
    # Phase oracle parses targets only and materializes the fixed phase skeleton.
    elif args.ablation_mode == PHASE_ORACLE_MODE:
        # Resolve model targets without reading case-specific reference targets.
        resolution = resolve_phase_oracle_output(json_text, case["task_name"])
        # Preserve the complete condition-specific parser audit.
        row["oracle_resolution"] = resolution
        # Map oracle parser fields onto the common result schema.
        row["strict_parsed_plan"] = resolution
        row["strict_parse_ok"] = bool(resolution.get("strict_parse_ok"))
        row["type_normalized_parsed_plan"] = resolution
        row["type_normalized_parse_ok"] = bool(resolution.get("type_normalized_parse_ok"))
        row["normalization_policy"] = "integer-string-target-only"
        row["target_numeric_cast_applied"] = bool(resolution.get("target_numeric_cast_applied"))
        row["semantic_repair_applied"] = False
        row["raw_steps"] = resolution.get("raw_targets")
        row["normalized_steps"] = resolution.get("normalized_targets")
        row["model_parse_ok"] = bool(resolution.get("model_parse_ok"))
        row["execution_used_type_normalized_plan"] = bool((not resolution.get("strict_parse_ok")) and resolution.get("ok"))
        # Convert to the common parsed-plan shape only after oracle contract validation.
        parsed = {"ok": bool(resolution.get("ok")), "steps": resolution.get("steps", []), "status": resolution.get("status", {})}
    # Target oracle parses skill/order first, then attaches frozen reference targets.
    else:
        # Keep env_config loading lazy so invalid model sequences cannot access oracle targets.
        env_config_loader = lambda: json.loads(case["env_config_path"].read_text(encoding="utf-8"))
        # Validate model skill/order before the resolver invokes the oracle-data loader.
        resolution = resolve_target_oracle_output(json_text, case["task_name"], env_config_loader)
        # Preserve whether condition/components were actually read for this row.
        row["oracle_target_read"] = bool(resolution.get("oracle_target_read"))
        # Preserve the complete condition-specific parser/contract audit.
        row["oracle_resolution"] = resolution
        # Target-oracle JSON has no numeric fields to normalize.
        row["strict_parsed_plan"] = resolution
        row["strict_parse_ok"] = bool(resolution.get("strict_parse_ok"))
        row["type_normalized_parsed_plan"] = resolution
        row["type_normalized_parse_ok"] = bool(resolution.get("type_normalized_parse_ok"))
        row["normalization_policy"] = "none_target_oracle_skill_sequence_only"
        row["target_numeric_cast_applied"] = False
        row["semantic_repair_applied"] = False
        row["raw_steps"] = resolution.get("model_skills")
        row["normalized_steps"] = resolution.get("model_skills")
        row["model_parse_ok"] = bool(resolution.get("model_parse_ok"))
        row["execution_used_type_normalized_plan"] = False
        # Keep target names as audit metadata; native binding uses numeric targets only.
        human_readable_targets = dict(resolution.get("human_readable_targets", {}))
        # Convert to the common parsed-plan shape only after sequence validation.
        parsed = {"ok": bool(resolution.get("ok")), "steps": resolution.get("steps", []), "status": resolution.get("status", {})}
    # Store parser result selected for execution.
    row["parsed_plan"] = parsed
    # If parser failed, convert it into an evaluation row without native execution.
    if not parsed.get("ok"):
        # Preserve taxonomy code/subtype from the parser.
        status = parsed["status"]
        # Merge parser failure fields into the flat row.
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
        # Return parser-failure row.
        return row
    # If inference-only is requested, stop after parser/schema validation.
    if args.inference_only:
        # Preserve normalized steps for manual inspection.
        row["normalized_steps"] = parsed["steps"]
        # Mark completion without native execution.
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
        # Return inference-only row.
        return row
    # Wrap normalized VLM steps into a model-evaluation fixture.
    fixture = fixture_from_model_plan(
        case=case,
        steps=parsed["steps"],
        ablation_mode=args.ablation_mode,
        human_readable_targets=human_readable_targets,
    )
    # Save the fixture before native execution.
    row["fixture"] = fixture
    # Execute the fixture through adapter/native.
    exec_row = execute_model_fixture(
        public_root=args.public_root,
        fixture=fixture,
        trace_root=args.output_root / "execution_traces",
        timeout_sec=args.native_timeout_sec,
        native_max_attempts=args.native_max_attempts,
    )
    # Merge execution fields into the model row.
    row.update(exec_row)
    # Mark per-case completion.
    row["ok"] = True
    # Return completed row.
    return row


def summarize(rows: list[dict[str, Any]], *, started_at: float, args: argparse.Namespace) -> dict[str, Any]:
    """Build aggregate procedural execution metrics."""

    # Count all selected rows.
    total = len(rows)
    # Completed rows include model failures because failures are represented as rows.
    completed = sum(1 for row in rows if row.get("ok") is True)
    # Parser/schema success counts accepted JSON procedural plans.
    parse_ok = sum(1 for row in rows if row.get("model_parse_ok") is True)
    # Strict parser success counts plans that already used integer targets.
    strict_parse_ok = sum(1 for row in rows if row.get("strict_parse_ok") is True)
    # Type-normalized parser success counts plans accepted after integer-string target casts.
    type_normalized_parse_ok = sum(1 for row in rows if row.get("type_normalized_parse_ok") is True)
    # Count rows where the execution path used the normalized plan.
    execution_used_type_normalized_plan = sum(1 for row in rows if row.get("execution_used_type_normalized_plan") is True)
    # Count rows where at least one target numeric cast occurred.
    target_numeric_cast_applied = sum(1 for row in rows if row.get("target_numeric_cast_applied") is True)
    # Gate1 success counts parser plus target binding plus phase binding.
    gate1_ok = sum(1 for row in rows if row.get("actual_gate1_ok") is True)
    # Native attempts count rows that reached MuJoCo.
    native_attempted = sum(1 for row in rows if row.get("native_execution_attempted") is True)
    # Native OK counts execution completion without native exception/timeout.
    native_ok = sum(1 for row in rows if row.get("actual_native_ok") is True)
    # Final procedural success requires condition_success true.
    procedural_success = sum(1 for row in rows if row.get("procedural_success") is True)
    # Runtime errors are unexpected script/API exceptions, not model plan failures.
    runtime_errors = sum(1 for row in rows if row.get("ok") is not True)
    # Count taxonomy codes for failure localization.
    failure_codes: dict[str, int] = {}
    # Count taxonomy subtypes for more detailed diagnosis.
    failure_subtypes: dict[str, int] = {}
    # Build category and task summaries.
    by_category: dict[str, dict[str, int]] = {}
    by_task: dict[str, dict[str, int]] = {}
    # Iterate once over rows.
    for row in rows:
        # Read category for grouped summary.
        category = str(row.get("category", ""))
        # Build Category/task key.
        task_key = f"{row.get('category', '')}/{row.get('task_name', '')}"
        # Initialize category bucket.
        by_category.setdefault(category, {"total": 0, "parse_ok": 0, "strict_parse_ok": 0, "type_normalized_parse_ok": 0, "target_numeric_cast_applied": 0, "gate1_ok": 0, "native_attempted": 0, "native_ok": 0, "procedural_success": 0})
        # Initialize task bucket.
        by_task.setdefault(task_key, {"total": 0, "parse_ok": 0, "strict_parse_ok": 0, "type_normalized_parse_ok": 0, "target_numeric_cast_applied": 0, "gate1_ok": 0, "native_attempted": 0, "native_ok": 0, "procedural_success": 0})
        # Update total counts.
        by_category[category]["total"] += 1
        by_task[task_key]["total"] += 1
        # Update stage counts.
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
            # Increment when the row field is true.
            if row.get(field) is True:
                by_category[category][bucket_name] += 1
                by_task[task_key][bucket_name] += 1
        # Count non-success taxonomy codes.
        code = str(row.get("actual_failure_code", ""))
        # Use C0 for successful rows so all rows are represented.
        code = "C0" if row.get("procedural_success") is True else (code or "UNCLASSIFIED")
        # Increment code count.
        failure_codes[code] = failure_codes.get(code, 0) + 1
        # Build subtype key.
        subtype_key = f"{code}:{row.get('actual_failure_subtype', '')}"
        # Increment subtype count.
        failure_subtypes[subtype_key] = failure_subtypes.get(subtype_key, 0) + 1
    # Return a JSON-serializable summary.
    return {
        "status": "completed" if runtime_errors == 0 else "completed_with_runtime_errors",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": time.time() - started_at,
        "experiment": "B_procedural_execution_interface" if args.ablation_mode == FULL_MODE else f"B_{args.ablation_mode}_ablation",
        "ablation_mode": args.ablation_mode,
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
        "normalization_policy": "integer-string-target-only" if args.ablation_mode != TARGET_ORACLE_MODE else "none_target_oracle_skill_sequence_only",
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

    # Parse CLI arguments.
    args = parse_args()
    # Record run start time.
    started_at = time.time()
    # Create output root before API/native calls.
    args.output_root.mkdir(parents=True, exist_ok=True)
    # Load shared 2,133-case manifest.
    manifest = load_manifest(args.manifest)
    # Apply optional pilot filters.
    selected_case_ids = select_case_ids(manifest=manifest, max_cases=args.max_cases, case_list_file=args.case_list_file)
    # Write run configuration for reproducibility.
    write_json(
        args.output_root / "run_config.json",
        {
            "selected_case_count": len(selected_case_ids),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "gt_used_for_prompt": False,
            "ablation_mode": args.ablation_mode,
            "phase_oracle_used_for_prompt": args.ablation_mode == PHASE_ORACLE_MODE,
            "target_oracle_used_for_prompt": False,
            "target_oracle_applied_after_valid_model_sequence": args.ablation_mode == TARGET_ORACLE_MODE,
            "operation_sequence_used_for_scoring": False,
            "counts_as_model_evaluation": True,
            "normalization_policy": "integer-string-target-only" if args.ablation_mode != TARGET_ORACLE_MODE else "none_target_oracle_skill_sequence_only",
            "semantic_repair_applied": False,
        },
    )
    # Keep compact rows in memory for summary snapshots.
    rows: list[dict[str, Any]] = []
    # JSONL path is append-only during execution.
    jsonl_path = args.output_root / "procedural_vlm_execution_rows.jsonl"
    # Remove stale JSONL unless resuming.
    if jsonl_path.exists() and not args.resume:
        jsonl_path.unlink()
    # Iterate selected examples in deterministic manifest order.
    for index, case_id in enumerate(selected_case_ids, start=1):
        # Per-case output enables resume and manual diagnosis.
        case_output_path = row_output_path(args.output_root, case_id)
        # Resume reuses completed rows.
        if args.resume and case_output_path.exists():
            row = json.loads(case_output_path.read_text(encoding="utf-8"))
        else:
            try:
                # Run one VLM procedural inference and native execution.
                row = run_one_case(case_id=case_id, args=args)
            except Exception as exc:  # noqa: BLE001 - long runs must preserve per-case failures.
                # Split fields best-effort for error rows.
                parts = (case_id.split("/") + ["", "", ""])[:3]
                # Build explicit runtime-error row.
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
            # Persist the per-case row before moving to the next case.
            write_json(case_output_path, row)
        # Append to in-memory rows.
        rows.append(row)
        # Append to JSONL stream.
        append_jsonl(jsonl_path, row)
        # Print concise progress so terminal logs show live status.
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
        # Write snapshots during long runs.
        if args.save_interval > 0 and index % args.save_interval == 0:
            write_csv(args.output_root / "procedural_vlm_execution_results.csv", rows)
            write_json(args.output_root / "summary.json", summarize(rows, started_at=started_at, args=args))
    # Write final CSV.
    write_csv(args.output_root / "procedural_vlm_execution_results.csv", rows)
    # Write final summary.
    summary = summarize(rows, started_at=started_at, args=args)
    write_json(args.output_root / "summary.json", summary)
    # Print final summary for terminal users.
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    # Return success because model failures are represented in rows.
    return 0


if __name__ == "__main__":
    sys.exit(main())
