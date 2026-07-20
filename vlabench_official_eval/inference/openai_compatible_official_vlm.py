"""OpenAI-compatible backend for the official VLABench VLM prompt format."""

from __future__ import annotations

import base64
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from vlabench_official_eval.dataset.case_loader import OfficialVLMCase


def _image_data_url(image_path: Path) -> str:
    """Encode a local PNG path as an OpenAI-compatible image URL payload."""

    # VLABench public view images are PNG files.
    image_bytes = image_path.read_bytes()
    # Base64 keeps the request self-contained and avoids serving local files over HTTP.
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    # vLLM OpenAI-compatible chat accepts data URLs in image_url content blocks.
    return "data:image/png;base64," + image_b64


def build_official_messages(*, case: OfficialVLMCase, official_prompt: str, with_cot: bool = False) -> list[dict[str, Any]]:
    """Build the same logical input sequence as VLABench BaseVLM.get_ti_list."""

    # The official prompt is the skill library and output-format instruction.
    content: list[dict[str, Any]] = [{"type": "text", "text": official_prompt}]
    # Official VLABench provides the unnumbered input image first.
    content.append({"type": "text", "text": "Input picture"})
    # The first image is the raw scene observation.
    content.append({"type": "image_url", "image_url": {"url": _image_data_url(case.input_image_path)}})
    # Official VLABench then provides the numbered-mask image.
    content.append({"type": "text", "text": "Input picture with numbered tags"})
    # The second image is the object-numbered view used for target selection.
    content.append({"type": "image_url", "image_url": {"url": _image_data_url(case.input_mask_path)}})
    # The instruction is the natural-language task description.
    content.append({"type": "text", "text": "Language instruction:"})
    # This is model-visible, unlike the GT operation sequence.
    content.append({"type": "text", "text": case.instruction})
    # Official VLABench ends by asking for the skill sequence.
    content.append({"type": "text", "text": "Please give the output skill sequence"})
    # CoT is optional because the official script exposes --with-cot.
    if with_cot:
        content.append({"type": "text", "text": "Please analyze the problem step by step and give the answer"})
    # Return a single user message, matching the official VLM wrappers.
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

    # The payload mirrors OpenAI chat/completions and works with vLLM's OpenAI server.
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Some models return fenced markdown unless JSON response mode is requested.
    if response_format_json:
        payload["response_format"] = {"type": "json_object"}
    # Build the HTTP request explicitly so the script has no third-party client dependency.
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # Record elapsed time for throughput and failure diagnostics.
    started = time.time()
    # Let urllib raise on transport failures; the runner catches and records them per case.
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        raw_body = response.read().decode("utf-8")
    # Parse the OpenAI-compatible response body.
    parsed_body = json.loads(raw_body)
    # The first choice is the deterministic greedy answer when temperature is zero.
    content = parsed_body.get("choices", [{}])[0].get("message", {}).get("content", "")
    # Return both the extracted text and the raw response for auditability.
    return {
        "ok": True,
        "elapsed_sec": time.time() - started,
        "content": content,
        "raw_response": parsed_body,
        "usage": parsed_body.get("usage"),
        "served_model": parsed_body.get("model"),
    }


def parse_official_skill_sequence(raw_text: str) -> dict[str, Any]:
    """Parse official VLABench VLM output into a skill_sequence list."""

    # Keep the raw model text no matter whether parsing succeeds.
    result: dict[str, Any] = {"origin_output": raw_text}
    # First try direct JSON because response_format=json_object often returns a plain object.
    candidates = [raw_text.strip()]
    # Then try fenced JSON, matching the official Qwen2_VL wrapper behavior.
    fenced_matches = re.findall(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    # Fenced candidates are tried after direct parsing.
    candidates.extend(match.strip() for match in fenced_matches)
    # Try every candidate in order.
    for candidate in candidates:
        # Empty candidates cannot parse.
        if not candidate:
            continue
        try:
            # Official prompt asks for a JSON list, but strict JSON mode may wrap it in an object.
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            # Continue because the next candidate may be the fenced block.
            continue
        # Official output may be directly a list of skill calls.
        if isinstance(parsed, list):
            result["skill_sequence"] = parsed
            result["parse_status"] = "parsed_list"
            return result
        # Some API prompts return {"skill_sequence": [...]}.
        if isinstance(parsed, dict) and isinstance(parsed.get("skill_sequence"), list):
            result["skill_sequence"] = parsed["skill_sequence"]
            result["parse_status"] = "parsed_skill_sequence_key"
            return result
        # Some strict prompts return {"sequence": [...]}.
        if isinstance(parsed, dict) and isinstance(parsed.get("sequence"), list):
            result["skill_sequence"] = parsed["sequence"]
            result["parse_status"] = "parsed_sequence_key"
            return result
    # Official evaluator uses format_error to assign zero scores.
    result["format_error"] = "format_error"
    # Store a short reason for downstream audit.
    result["parse_status"] = "failed"
    # Return the official-compatible failure object.
    return result
