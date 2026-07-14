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

    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    return "data:image/png;base64," + image_b64


def build_official_messages(*, case: OfficialVLMCase, official_prompt: str, with_cot: bool = False) -> list[dict[str, Any]]:
    """Build the same logical input sequence as VLABench BaseVLM.get_ti_list."""

    content: list[dict[str, Any]] = [{"type": "text", "text": official_prompt}]
    content.append({"type": "text", "text": "Input picture"})
    content.append({"type": "image_url", "image_url": {"url": _image_data_url(case.input_image_path)}})
    content.append({"type": "text", "text": "Input picture with numbered tags"})
    content.append({"type": "image_url", "image_url": {"url": _image_data_url(case.input_mask_path)}})
    content.append({"type": "text", "text": "Language instruction:"})
    content.append({"type": "text", "text": case.instruction})
    content.append({"type": "text", "text": "Please give the output skill sequence"})
    if with_cot:
        content.append({"type": "text", "text": "Please analyze the problem step by step and give the answer"})
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


def parse_official_skill_sequence(raw_text: str) -> dict[str, Any]:
    """Parse official VLABench VLM output into a skill_sequence list."""

    result: dict[str, Any] = {"origin_output": raw_text}
    candidates = [raw_text.strip()]
    fenced_matches = re.findall(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(match.strip() for match in fenced_matches)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            result["skill_sequence"] = parsed
            result["parse_status"] = "parsed_list"
            return result
        if isinstance(parsed, dict) and isinstance(parsed.get("skill_sequence"), list):
            result["skill_sequence"] = parsed["skill_sequence"]
            result["parse_status"] = "parsed_skill_sequence_key"
            return result
        if isinstance(parsed, dict) and isinstance(parsed.get("sequence"), list):
            result["skill_sequence"] = parsed["sequence"]
            result["parse_status"] = "parsed_sequence_key"
            return result
    result["format_error"] = "format_error"
    result["parse_status"] = "failed"
    return result
