"""Separate image inference from frozen-output procedural evaluation.

The cloud inference path uses only the Python standard library and public inputs.
Native evaluation imports the existing project only inside the evaluate command.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import urllib.error
import urllib.request

sys.dont_write_bytecode = True
VERSION = "schema-decoding-v1-20261021"
HERE = Path(__file__).resolve().parent
CONDITIONS = ("json_object", "json_schema")
INPUT_NAMES = ("input.png", "input_mask.png", "instruction.txt")


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(handle.read()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_case_id(case_id):
    parts = case_id.split("/")
    if len(parts) != 3 or any(part in ("", ".", "..") or "\\" in part for part in parts):
        raise ValueError(f"Unsafe case id: {case_id!r}")
    if not parts[2].startswith("example") or not parts[2][7:].isdigit():
        raise ValueError(f"Invalid example id: {case_id!r}")
    return case_id.replace("/", "__")


def load_case_ids(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    ids = json.loads(text) if path.suffix == ".json" else [line.strip() for line in text.splitlines() if line.strip()]
    if not isinstance(ids, list) or not ids or not all(isinstance(item, str) for item in ids):
        raise ValueError("Case manifest must be a nonempty list of strings")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate case ids are not allowed")
    for case_id in ids:
        safe_case_id(case_id)
    return ids


def select_cases(ids, per_task, seed, requested=None):
    if requested is not None:
        unknown = set(requested) - set(ids)
        if unknown:
            raise ValueError(f"Requested case ids outside manifest: {sorted(unknown)}")
        keep = set(requested)
    elif per_task:
        groups = defaultdict(list)
        for case_id in ids:
            groups[case_id.rsplit("/", 1)[0]].append(case_id)
        rng = random.Random(seed)
        keep = {case_id for key in sorted(groups) for case_id in rng.sample(groups[key], min(per_task, len(groups[key])))}
    else:
        keep = set(ids)
    return [case_id for case_id in ids if case_id in keep]


def require_separate(output, *sources):
    output = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError(f"Output must be separate from protected input: {source}")


@contextmanager
def run_lock(root):
    # POSIX advisory locks are released on process exit, including an interrupted WSL session.
    import fcntl

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is writing {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def initialize_run(root, config, resume):
    root = Path(root)
    path = root / "run_config.json"
    other = [item for item in root.iterdir() if item.name != ".run.lock"]
    if path.exists():
        if not resume:
            raise FileExistsError(f"Existing run will not be overwritten: {root}; use --resume")
        if read_json(path) != config:
            raise ValueError("Resume configuration differs from the frozen run")
    elif other:
        raise FileExistsError(f"Nonempty output directory will not be overwritten: {root}")
    else:
        write_json(path, config)


def import_legacy(project_root):
    sys.path.insert(0, str(Path(project_root).resolve()))
    from procedural_interface.scripts import run_procedural_vlm_eval_api

    return run_procedural_vlm_eval_api


def prepare(args):
    require_separate(args.output_root, args.public_root, args.manifest)
    ids = load_case_ids(args.manifest)
    requested = load_case_ids(args.case_list_file) if args.case_list_file else None
    selected = select_cases(ids, args.per_task, args.selection_seed, requested)
    legacy = import_legacy(args.project_root)
    root = args.output_root
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Bundle directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    schema = read_json(HERE / "procedural_plan_schema.json")
    entries = []
    for case_id in selected:
        key = safe_case_id(case_id)
        source = args.public_root / case_id
        target = root / "inputs" / key
        target.mkdir(parents=True)
        hashes = {}
        # Deliberately copy no env_config, expert sequence, or operation sequence to GCP.
        for name in INPUT_NAMES:
            shutil.copyfile(source / name, target / name)
            hashes[name] = digest(target / name)
        instruction = (target / "instruction.txt").read_text(encoding="utf-8").strip()
        prompt = legacy.build_procedural_prompt(instruction=instruction)
        (target / "prompt.txt").write_text(prompt, encoding="utf-8")
        hashes["prompt.txt"] = digest(target / "prompt.txt")
        entries.append({"case_id": case_id, "input_dir": str(target.relative_to(root)), "sha256": hashes})
    shutil.copyfile(HERE / "schema_decoding.py", root / "schema_decoding.py")
    write_json(root / "procedural_plan_schema.json", schema)
    manifest = {
        "version": VERSION, "case_count": len(entries), "task_count": len({item.rsplit('/', 1)[0] for item in selected}),
        "selection": {"per_task": args.per_task, "seed": args.selection_seed, "source_manifest_sha256": digest(args.manifest)},
        "prompt_source_sha256": digest(legacy.__file__), "schema_sha256": digest(root / "procedural_plan_schema.json"),
        "runner_sha256": digest(root / "schema_decoding.py"), "cases": entries,
        "gt_inputs_exported": False,
    }
    write_json(root / "bundle_manifest.json", manifest)
    (root / "case_ids.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    write_json(root / "serving_config.template.json", {
        "model_revision": "REPLACE_WITH_ACTUAL_HF_COMMIT", "vllm_version": "REPLACE_WITH_INSTALLED_VERSION",
        "dtype": "bfloat16", "server_command": "REPLACE_WITH_FULL_SERVE_COMMAND",
        "processor_revision": "REPLACE_WITH_ACTUAL_REVISION", "chat_template": "REPLACE_WITH_ACTUAL_TEMPLATE_SETTING",
        "image_preprocessing": "REPLACE_WITH_PROCESSOR_CONFIG_OR_CONFIG_PATH",
    })
    print(json.dumps({"bundle": str(root), "cases": len(entries), "tasks": manifest["task_count"], "gt_inputs_exported": False}), flush=True)
    return 0


def load_bundle(root):
    root = Path(root).resolve()
    manifest = read_json(root / "bundle_manifest.json")
    ids = [entry["case_id"] for entry in manifest["cases"]]
    if len(ids) != len(set(ids)) or len(ids) != manifest["case_count"]:
        raise ValueError("Invalid bundle case count or duplicate ids")
    if digest(root / "procedural_plan_schema.json") != manifest["schema_sha256"]:
        raise ValueError("Bundle schema checksum mismatch")
    if digest(root / "schema_decoding.py") != manifest["runner_sha256"]:
        raise ValueError("Bundle runner checksum mismatch")
    for entry in manifest["cases"]:
        key = safe_case_id(entry["case_id"])
        if entry["input_dir"] != f"inputs/{key}":
            raise ValueError("Unexpected bundle input path")
        for name, expected in entry["sha256"].items():
            if name not in (*INPUT_NAMES, "prompt.txt") or digest(root / entry["input_dir"] / name) != expected:
                raise ValueError(f"Bundle input checksum mismatch: {entry['case_id']}/{name}")
    return manifest


def messages_for_entry(bundle, entry):
    root = Path(bundle) / entry["input_dir"]
    def image(name):
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode((root / name).read_bytes()).decode("ascii")}}
    return [{"role": "user", "content": [
        {"type": "text", "text": (root / "prompt.txt").read_text(encoding="utf-8")},
        {"type": "text", "text": "Input picture"}, image("input.png"),
        {"type": "text", "text": "Input picture with numbered tags"}, image("input_mask.png"),
        {"type": "text", "text": "Language instruction:"},
        {"type": "text", "text": (root / "instruction.txt").read_text(encoding="utf-8").strip()},
        {"type": "text", "text": "Please give the output skill sequence"},
    ]}]


def build_payload(messages, model, condition, schema, temperature=0.0, max_tokens=1024, seed=0):
    response_format = {"type": "json_object"} if condition == "json_object" else {
        "type": "json_schema", "json_schema": {"name": "procedural_plan", "strict": True, "schema": schema},
    }
    if condition not in CONDITIONS:
        raise ValueError(f"Unsupported condition: {condition}")
    return {"model": model, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "seed": seed, "response_format": response_format}


def request_record(payload, entry):
    record = json.loads(json.dumps(payload))
    images = iter(("input.png", "input_mask.png"))
    # Image bytes are deduplicated in the immutable bundle; all other request fields are exact.
    for block in record["messages"][0]["content"]:
        if block["type"] == "image_url":
            name = next(images)
            block["image_url"]["url"] = f"bundle-file://{entry['input_dir']}/{name}#sha256={entry['sha256'][name]}"
    return record


def get_json(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_api(base_url, payload, timeout, attempt_dir):
    started = time.monotonic()
    request = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                     data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
    status = None
    raw = b""
    error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
        error = f"HTTP {status}"
    except (OSError, TimeoutError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    (attempt_dir / "response_body.bin").write_bytes(raw)
    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        body = None
        error = error or f"Invalid API response: {exc}"
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        error = error or "API response has no choice"
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = first.get("message") or {}
    if not isinstance(message, dict):
        error = error or "API message is not an object"
        message = {}
    content = message.get("content")
    if content is not None and not isinstance(content, str):
        error = error or "API content is not text"
    if body is not None and isinstance(body, dict) and body.get("model") != payload["model"]:
        error = error or "API served a different or unspecified model"
    return {"transport_ok": error is None, "http_status": status, "error": error,
            "origin_output": content, "finish_reason": first.get("finish_reason"),
            "refusal": message.get("refusal"), "raw_response": body,
            "usage": body.get("usage") if isinstance(body, dict) else None,
            "served_model": body.get("model") if isinstance(body, dict) else None,
            "elapsed_sec": time.monotonic() - started}


def summarize(rows):
    return {"stage": "offline_evaluation" if any("api_called" in row for row in rows) else "inference",
        "total_rows": len(rows), "by_condition": {
        condition: {"rows": len(group), "transport_errors": sum(not row.get("transport_ok", True) for row in group),
                    "truncated": sum(row.get("finish_reason") == "length" for row in group),
                    "parse_valid": sum(row.get("model_parse_ok") is True for row in group),
                    "strict_parse_valid": sum(row.get("strict_parse_ok") is True for row in group),
                    "raw_json_schema_valid": sum(row.get("raw_json_schema_valid") is True for row in group),
                    "type_normalized_json_schema_valid": sum(row.get("type_normalized_json_schema_valid") is True for row in group),
                    "numeric_cast_cases": sum(row.get("target_numeric_cast_applied") is True for row in group),
                    "gate_valid": sum(row.get("actual_gate1_ok") is True for row in group),
                    "native_attempted": sum(row.get("native_execution_attempted") is True for row in group),
                    "success": sum(row.get("procedural_success") is True for row in group),
                    "failures": dict(Counter(row.get("actual_failure_subtype") or row.get("actual_failure_code") or "none" for row in group))}
        for condition in CONDITIONS if (group := [row for row in rows if row["condition"] == condition])}}


def infer(args):
    require_separate(args.output_root, args.bundle)
    manifest = load_bundle(args.bundle)
    schema = read_json(args.bundle / "procedural_plan_schema.json")
    serving = read_json(args.serving_config)
    required = ("model_revision", "vllm_version", "dtype", "server_command", "processor_revision", "chat_template", "image_preprocessing")
    if any(not serving.get(key) or "REPLACE_" in str(serving[key]) for key in required):
        raise ValueError("Fill every field of serving_config.template.json using the actual server configuration")
    if serving["dtype"] not in ("bfloat16", "bf16"):
        raise ValueError("This comparison requires BF16; quantized/FP16 serving is not interchangeable")
    conditions = list(CONDITIONS) if args.condition == "both" else [args.condition]
    entries = manifest["cases"][:args.max_cases or None]
    config = {"version": VERSION, "stage": "inference", "bundle_manifest_sha256": digest(args.bundle / "bundle_manifest.json"),
              "case_ids": [entry["case_id"] for entry in entries], "conditions": conditions, "model": args.model,
              "temperature": args.temperature, "max_tokens": args.max_tokens, "seed": args.seed,
              "serving_config": serving, "runner_sha256": digest(__file__)}
    with run_lock(args.output_root):
        initialize_run(args.output_root, config, args.resume)
        # Fail before generating any cases if the tunnel points at the wrong model.
        inventory = get_json(args.base_url.rstrip("/") + "/models", args.timeout_sec)
        if args.model not in {model.get("id") for model in inventory.get("data", [])}:
            raise ValueError(f"Requested model is not served: {args.model}")
        write_json(args.output_root / "server_models.json", inventory)
        write_json(args.output_root / "serving_config.json", serving)
        rows = []
        for entry in entries:
            messages = messages_for_entry(args.bundle, entry)
            for condition in conditions:
                case_root = args.output_root / condition / "case_outputs" / safe_case_id(entry["case_id"])
                path = case_root / "row.json"
                old = read_json(path) if path.exists() else None
                if old and old.get("transport_ok"):
                    row = old
                else:
                    if old and not args.retry_api_errors:
                        raise RuntimeError("Stored API failure requires --retry-api-errors; evidence will be preserved")
                    previous = old.get("attempt_history", []) if old else []
                    attempt = case_root / f"api_attempt_{len(previous) + 1:02d}"
                    attempt.mkdir(parents=True, exist_ok=False)
                    payload = build_payload(messages, args.model, condition, schema, args.temperature, args.max_tokens, args.seed)
                    write_json(attempt / "request.json", request_record(payload, entry))
                    row = call_api(args.base_url, payload, args.timeout_sec, attempt)
                    row.update({"case_id": entry["case_id"], "condition": condition, "requested_model": args.model,
                                "base_url": args.base_url,
                                "model_revision": serving["model_revision"], "attempt_history": previous + [str(attempt.relative_to(args.output_root))],
                                "request_sha256": digest(attempt / "request.json"),
                                "response_body_sha256": digest(attempt / "response_body.bin"),
                                "semantic_repair_applied": False})
                    write_json(attempt / "row.json", row)
                    write_json(path, row)
                rows.append(row)
                print(json.dumps({"index": len(rows), "total": len(entries) * len(conditions), "case_id": entry["case_id"],
                                  "condition": condition, "transport_ok": row["transport_ok"], "finish_reason": row["finish_reason"]}), flush=True)
                write_json(args.output_root / "summary.json", summarize(rows))
                if not row["transport_ok"]:
                    raise RuntimeError("API error saved; stopped without fallback, regeneration, or changing condition")
        artifacts = {str((args.output_root / row["condition"] / "case_outputs" / safe_case_id(row["case_id"]) / "row.json").relative_to(args.output_root)):
                     digest(args.output_root / row["condition"] / "case_outputs" / safe_case_id(row["case_id"]) / "row.json") for row in rows}
        write_json(args.output_root / "generation_complete.json", {"run_config_sha256": digest(args.output_root / "run_config.json"),
                   "expected_rows": len(entries) * len(conditions), "artifacts": artifacts})
    return 0


def load_frozen_generation(root, bundle):
    manifest = load_bundle(bundle)
    config = read_json(root / "run_config.json")
    complete = read_json(root / "generation_complete.json")
    if config["stage"] != "inference" or complete["run_config_sha256"] != digest(root / "run_config.json"):
        raise ValueError("Generation configuration integrity failure")
    if config["bundle_manifest_sha256"] != digest(bundle / "bundle_manifest.json"):
        raise ValueError("Generation belongs to a different input bundle")
    if not set(config["case_ids"]) <= {entry["case_id"] for entry in manifest["cases"]}:
        raise ValueError("Generation has case ids outside the bundle")
    expected = {f"{condition}/case_outputs/{safe_case_id(case_id)}/row.json" for case_id in config["case_ids"] for condition in config["conditions"]}
    if set(complete["artifacts"]) != expected or len(expected) != complete["expected_rows"]:
        raise ValueError("Missing, extra, or duplicate generation rows")
    rows = []
    for case_id in config["case_ids"]:
        for condition in config["conditions"]:
            relative = f"{condition}/case_outputs/{safe_case_id(case_id)}/row.json"
            path = root / relative
            if digest(path) != complete["artifacts"][relative]:
                raise ValueError(f"Frozen output checksum mismatch: {relative}")
            row = read_json(path)
            if row["case_id"] != case_id or row["condition"] != condition or row["requested_model"] != config["model"]:
                raise ValueError(f"Frozen output identity mismatch: {relative}")
            attempt = root / row["attempt_history"][-1]
            if digest(attempt / "request.json") != row["request_sha256"] or digest(attempt / "response_body.bin") != row["response_body_sha256"]:
                raise ValueError(f"Frozen request/response checksum mismatch: {relative}")
            rows.append((path, row))
    return config, rows


def evaluate_content(legacy, raw_content):
    from procedural_interface.interface.parser import parse_plan_text

    # Follow the unchanged full-B parser: syntactic extraction, then numeric-string casts only.
    text, extraction = legacy.extract_json_object_text(raw_content if isinstance(raw_content, str) else "")
    strict = parse_plan_text(text)
    normalization = legacy.normalize_integer_string_targets(text)
    normalized = parse_plan_text(str(normalization.get("normalized_text", text)))
    parsed = strict if strict.get("ok") else normalized
    return {"strict_parse_ok": bool(strict.get("ok")), "type_normalized_parse_ok": bool(normalized.get("ok")),
            "model_parse_ok": bool(parsed.get("ok")), "parsed_plan": parsed, "strict_parsed_plan": strict,
            "type_normalized_parsed_plan": normalized, "normalization_policy": normalization["policy"],
            "target_numeric_cast_applied": bool(normalization.get("target_numeric_cast_applied")),
            "raw_steps": normalization.get("raw_steps"), "normalized_steps": normalization.get("normalized_steps"),
            "execution_used_type_normalized_plan": not strict.get("ok") and bool(normalized.get("ok")),
            "json_extraction_status": extraction, "semantic_repair_applied": False}


def schema_audit(legacy, raw_content, schema):
    """Measure schema compliance without replacing the existing execution parser."""
    import jsonschema

    text, _ = legacy.extract_json_object_text(raw_content if isinstance(raw_content, str) else "")
    normalized = legacy.normalize_integer_string_targets(text)["normalized_text"]
    results = {}
    for label, candidate in (("raw", text), ("type_normalized", normalized)):
        try:
            jsonschema.validate(json.loads(candidate), schema)
            results[f"{label}_json_schema_valid"] = True
            results[f"{label}_json_schema_error"] = ""
        except (ValueError, jsonschema.ValidationError) as exc:
            results[f"{label}_json_schema_valid"] = False
            results[f"{label}_json_schema_error"] = str(exc).splitlines()[0]
    return results


def evaluate_one(legacy, source, case, output_root, public_root, validation_only, timeout):
    row = {"case_id": source["case_id"], "condition": source["condition"], "served_model": source.get("served_model"),
           "finish_reason": source.get("finish_reason"), "transport_ok": source["transport_ok"], "api_called": False,
           "native_execution_attempted": False, "actual_gate1_ok": False, "actual_native_ok": False,
           "actual_condition_success": False, "procedural_success": False, "native_attempt_count": 0}
    if not source["transport_ok"]:
        row.update(ok=False, actual_failure_code="INFRA", actual_failure_subtype="stored_api_error")
        return row
    row.update(evaluate_content(legacy, source.get("origin_output")))
    parsed = row["parsed_plan"]
    if not parsed["ok"]:
        row.update(ok=True, actual_failure_code=parsed["status"]["code"], actual_failure_subtype=parsed["status"]["subtype"], actual_failure_detail=parsed["status"]["detail"])
        return row
    fixture = legacy.fixture_from_model_plan(case=case, steps=parsed["steps"], ablation_mode=legacy.FULL_MODE)
    if validation_only:
        from procedural_interface.dataset.public_view_loader import SampleRef
        sample = SampleRef(category=case["category"], task_name=case["task_name"], example_id=case["example_name"], root=case["case_root"])
        gate = legacy._bind_fixture_plan(sample, fixture)
        write_json(output_root / source["condition"] / "validation_traces" / (safe_case_id(source["case_id"]) + ".json"), gate)
        row.update(ok=True, actual_gate1_ok=bool(gate["ok"]), actual_failure_code=gate["status"]["code"],
                   actual_failure_subtype=gate["status"]["subtype"] if not gate["ok"] else "validation_only",
                   actual_failure_detail=gate["status"]["detail"])
    else:
        # The existing binder, lowerer, native worker and reset policy are reused without changes.
        result = legacy.execute_model_fixture(public_root=public_root, fixture=fixture,
                    trace_root=output_root / source["condition"] / "execution_traces", timeout_sec=timeout, native_max_attempts=1)
        row.update(result, ok=True)
        row["native_retry_policy"] = "first_attempt_only"
        if row["native_attempt_count"] > 1:
            raise RuntimeError("Single-attempt protocol violation")
    return row


def write_results(root, rows):
    write_json(root / "summary.json", summarize(rows))
    fields = ("case_id", "condition", "served_model", "finish_reason", "ok", "strict_parse_ok", "model_parse_ok",
              "raw_json_schema_valid", "type_normalized_json_schema_valid",
              "target_numeric_cast_applied", "actual_gate1_ok", "native_execution_attempted", "actual_native_ok",
              "actual_condition_success", "procedural_success", "actual_failure_code", "actual_failure_subtype",
              "actual_failure_detail", "native_attempt_count", "api_called", "semantic_repair_applied")
    with (root / "results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(args):
    require_separate(args.output_root, args.generation_root, args.bundle, args.public_root)
    config, frozen = load_frozen_generation(args.generation_root, args.bundle)
    entries = {entry["case_id"]: entry for entry in load_bundle(args.bundle)["cases"]}
    legacy = import_legacy(args.project_root)
    source_files = {str(path.relative_to(args.project_root)): digest(path)
                    for folder in ("procedural_interface", "vlabench_official_eval", "shared")
                    for path in (args.project_root / folder).rglob("*.py")}
    selected_ids = set(config["case_ids"][:args.max_cases or None])
    evaluation_config = {"version": VERSION, "stage": "offline_evaluation", "generation_complete_sha256": digest(args.generation_root / "generation_complete.json"),
                         "case_ids": [case_id for case_id in config["case_ids"] if case_id in selected_ids], "conditions": config["conditions"],
                         "validation_only": args.validation_only, "native_max_attempts": 1, "native_seed": None,
                         "native_timeout_sec": args.native_timeout_sec, "legacy_runner_sha256": digest(legacy.__file__),
                         "runner_sha256": digest(__file__), "public_root": str(args.public_root.resolve()), "api_calls_allowed": False}
    evaluation_config["evaluation_source_sha256"] = source_files
    # Verify every selected public input before launching any native worker.
    cases = {}
    for case_id in selected_ids:
        case = legacy.split_public_case(args.public_root, case_id)
        for name in INPUT_NAMES:
            if digest(case["case_root"] / name) != entries[case_id]["sha256"][name]:
                raise ValueError(f"Local evaluation input differs from cloud inference input: {case_id}/{name}")
        cases[case_id] = case
    evaluation_config["env_config_sha256"] = {case_id: digest(case["env_config_path"]) for case_id, case in cases.items()}
    with run_lock(args.output_root):
        initialize_run(args.output_root, evaluation_config, args.resume)
        rows = []
        for source_path, source in frozen:
            if source["case_id"] not in selected_ids:
                continue
            path = args.output_root / source["condition"] / "case_outputs" / (safe_case_id(source["case_id"]) + ".json")
            if args.resume and path.exists():
                row = read_json(path)
                if row.get("source_sha256") != digest(source_path):
                    raise ValueError("Saved evaluation row belongs to different frozen output")
            else:
                row = evaluate_one(legacy, source, cases[source["case_id"]], args.output_root, args.public_root,
                                   args.validation_only, args.native_timeout_sec)
                row.update(schema_audit(legacy, source.get("origin_output"), read_json(args.bundle / "procedural_plan_schema.json")))
                row.update(source_sha256=digest(source_path), source_path=str(source_path), category=cases[source["case_id"]]["category"], task_name=cases[source["case_id"]]["task_name"])
                write_json(path, row)
            rows.append(row)
            print(json.dumps({"index": len(rows), "total": len(selected_ids) * len(config["conditions"]),
                              "case_id": row["case_id"], "condition": row["condition"], "parse": row.get("model_parse_ok"),
                              "gate": row["actual_gate1_ok"], "native": row["native_execution_attempted"],
                              "success": row["procedural_success"], "failure": row.get("actual_failure_subtype")}), flush=True)
            write_results(args.output_root, rows)
            if row.get("actual_failure_code") == "INFRA":
                raise RuntimeError("Infrastructure failure saved; stopped instead of counting it as a model failure")
        write_json(args.output_root / "evaluation_complete.json", {"rows": len(rows), "api_calls": 0, "native_max_attempts": 1})
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Export only public images, instruction and unchanged B prompt")
    prep.add_argument("--project-root", type=Path, required=True)
    prep.add_argument("--public-root", type=Path, required=True)
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--case-list-file", type=Path)
    prep.add_argument("--per-task", type=int, default=10, help="Zero exports the full manifest")
    prep.add_argument("--selection-seed", type=int, default=20261021)
    prep.add_argument("--output-root", type=Path, required=True)
    prep.set_defaults(handler=prepare)
    generate = commands.add_parser("infer", help="Run the two format conditions without VLABench or MuJoCo imports")
    generate.add_argument("--bundle", type=Path, required=True)
    generate.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    generate.add_argument("--model", required=True)
    generate.add_argument("--serving-config", type=Path, required=True)
    generate.add_argument("--condition", choices=(*CONDITIONS, "both"), default="both")
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--max-tokens", type=int, default=1024)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--timeout-sec", type=int, default=180)
    generate.add_argument("--max-cases", type=int, default=0)
    generate.add_argument("--output-root", type=Path, required=True)
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--retry-api-errors", action="store_true", help="Explicitly retry transport failures, retaining all API attempts")
    generate.set_defaults(handler=infer)
    replay = commands.add_parser("evaluate", help="Evaluate frozen outputs with the existing local procedural backend, no VLM")
    replay.add_argument("--project-root", type=Path, required=True)
    replay.add_argument("--public-root", type=Path, required=True)
    replay.add_argument("--bundle", type=Path, required=True)
    replay.add_argument("--generation-root", type=Path, required=True)
    replay.add_argument("--output-root", type=Path, required=True)
    replay.add_argument("--validation-only", action="store_true")
    replay.add_argument("--native-timeout-sec", type=int, default=240)
    replay.add_argument("--max-cases", type=int, default=0)
    replay.add_argument("--resume", action="store_true")
    replay.set_defaults(handler=evaluate)
    args = parser.parse_args(argv)
    if getattr(args, "max_cases", 0) < 0 or getattr(args, "per_task", 0) < 0:
        parser.error("Case counts must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}, ensure_ascii=False), file=sys.stderr, flush=True)
        raise SystemExit(2)
