"""Small OpenAI-compatible VLM client used by both evaluation branches."""

from __future__ import annotations

import base64
import json
import time
import urllib.request
from pathlib import Path
from typing import Any


def image_data_url(image_path: Path) -> str:
    image_bytes = image_path.read_bytes()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")
    return "data:image/png;base64," + image_b64


def chat_completion(
    *,
    base_url: str,
    model: str,
    prompt: str,
    image_path: Path | None = None,
    temperature: float = 0,
    max_tokens: int = 512,
    response_format_json: bool = False,
    timeout_sec: int = 120,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if image_path is not None:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(image_path)}})
    content.append({"type": "text", "text": prompt})

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
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
        raw = response.read().decode("utf-8")
    parsed = json.loads(raw)
    message = parsed.get("choices", [{}])[0].get("message", {}).get("content", "")
    return {
        "ok": True,
        "elapsed_sec": time.time() - started,
        "model": parsed.get("model"),
        "content": message,
        "usage": parsed.get("usage"),
        "raw_response": parsed,
    }


def list_models(*, base_url: str, timeout_sec: int = 20) -> dict[str, Any]:
    request = urllib.request.Request(base_url.rstrip("/") + "/models", method="GET")
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        return json.loads(response.read().decode("utf-8"))

