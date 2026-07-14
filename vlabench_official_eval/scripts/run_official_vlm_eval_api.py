"""Run official-style VLABench VLM symbolic evaluation through a vLLM API backend."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

from vlabench_official_eval.dataset.case_loader import load_case, load_manifest, select_case_ids
from vlabench_official_eval.inference.openai_compatible_official_vlm import (
    build_official_messages,
    call_openai_compatible_vlm,
    parse_official_skill_sequence,
)
from vlabench_official_eval.scoring.official_symbolic_scorer import (
    ensure_vlabench_root,
    load_seq_independent_tasks,
    score_with_official_vlabench,
    score_with_type_normalized_vlabench,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI flags for official-style symbolic VLM evaluation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "datasets" / "manifests" / "matched_2133_symbolic_procedural_cases.json")
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, default=PROJECT_ROOT / "vlabench_official_eval" / "prompts" / "eval_vlm_en.txt")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "official_sequence_eval" / "run")
    parser.add_argument("--case-list-file", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--with-cot", action="store_true")
    parser.add_argument("--response-format-json", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_json(path: Path, data: Any) -> None:
    """Write UTF-8 JSON with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSONL row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact CSV table for spreadsheet inspection."""

    fieldnames = [
        "case_id",
        "category",
        "task_name",
        "example_name",
        "ok",
        "format_error",
        "parse_status",
        "symbolic_success",
        "skill_match_score",
        "entity_match_score",
        "skill_with_entity_match_score",
        "exact_match_score",
        "total_score",
        "type_normalized_symbolic_success",
        "type_normalized_numeric_cast_applied",
        "type_normalized_skill_match_score",
        "type_normalized_entity_match_score",
        "type_normalized_skill_with_entity_match_score",
        "type_normalized_exact_match_score",
        "type_normalized_total_score",
        "official_dependency",
        "elapsed_sec",
        "error_type",
        "error_detail",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], *, started_at: float, args: argparse.Namespace) -> dict[str, Any]:
    """Build aggregate official symbolic metrics."""

    total = len(rows)
    completed = sum(1 for row in rows if row.get("ok") is True)
    symbolic_success = sum(1 for row in rows if row.get("symbolic_success") is True)
    type_normalized_symbolic_success = sum(1 for row in rows if row.get("type_normalized_symbolic_success") is True)
    format_errors = sum(1 for row in rows if row.get("format_error") is True)
    runtime_errors = sum(1 for row in rows if row.get("ok") is not True)
    metric_names = ["skill_match_score", "entity_match_score", "skill_with_entity_match_score", "exact_match_score", "total_score"]
    type_normalized_metric_names = [
        "type_normalized_skill_match_score",
        "type_normalized_entity_match_score",
        "type_normalized_skill_with_entity_match_score",
        "type_normalized_exact_match_score",
        "type_normalized_total_score",
    ]
    means = {
        name: (sum(float(row.get(name, 0) or 0) for row in rows) / total if total else 0)
        for name in metric_names
    }
    type_normalized_means = {
        name: (sum(float(row.get(name, 0) or 0) for row in rows) / total if total else 0)
        for name in type_normalized_metric_names
    }
    type_normalized_numeric_cast_applied = sum(1 for row in rows if row.get("type_normalized_numeric_cast_applied") is True)
    by_category: dict[str, dict[str, int]] = {}
    by_task: dict[str, dict[str, int]] = {}
    for row in rows:
        category = str(row.get("category", ""))
        task_key = f"{row.get('category', '')}/{row.get('task_name', '')}"
        by_category.setdefault(category, {"total": 0, "symbolic_success": 0, "type_normalized_symbolic_success": 0, "format_error": 0, "runtime_error": 0})
        by_task.setdefault(task_key, {"total": 0, "symbolic_success": 0, "type_normalized_symbolic_success": 0, "format_error": 0, "runtime_error": 0})
        by_category[category]["total"] += 1
        by_task[task_key]["total"] += 1
        if row.get("symbolic_success") is True:
            by_category[category]["symbolic_success"] += 1
            by_task[task_key]["symbolic_success"] += 1
        if row.get("type_normalized_symbolic_success") is True:
            by_category[category]["type_normalized_symbolic_success"] += 1
            by_task[task_key]["type_normalized_symbolic_success"] += 1
        if row.get("format_error") is True:
            by_category[category]["format_error"] += 1
            by_task[task_key]["format_error"] += 1
        if row.get("ok") is not True:
            by_category[category]["runtime_error"] += 1
            by_task[task_key]["runtime_error"] += 1
    return {
        "status": "completed" if runtime_errors == 0 else "completed_with_runtime_errors",
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": time.time() - started_at,
        "manifest": str(args.manifest),
        "public_root": str(args.public_root),
        "gt_root": str(args.gt_root),
        "prompt_path": str(args.prompt_path),
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "with_cot": args.with_cot,
        "response_format_json": args.response_format_json,
        "score_only": args.score_only,
        "total_cases": total,
        "completed_rows": completed,
        "symbolic_success": symbolic_success,
        "symbolic_success_rate": symbolic_success / total if total else 0,
        "type_normalized_symbolic_success": type_normalized_symbolic_success,
        "type_normalized_symbolic_success_rate": type_normalized_symbolic_success / total if total else 0,
        "type_normalized_numeric_cast_applied": type_normalized_numeric_cast_applied,
        "format_errors": format_errors,
        "runtime_errors": runtime_errors,
        "metric_means": means,
        "type_normalized_metric_means": type_normalized_means,
        "by_category": by_category,
        "by_task": by_task,
    }


def row_output_path(output_root: Path, case_id: str) -> Path:
    """Map a case id to a stable per-case JSON path."""

    safe_case_id = case_id.replace("/", "__")
    return output_root / "case_outputs" / f"{safe_case_id}.json"


def run_one_case(
    *,
    case_id: str,
    args: argparse.Namespace,
    official_prompt: str,
    seq_independent_tasks: set[str],
) -> dict[str, Any]:
    """Run or score one official-style VLM example."""

    case = load_case(case_id=case_id, public_root=args.public_root, gt_root=args.gt_root)
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "category": case.category,
        "task_name": case.task_name,
        "example_name": case.example_name,
        "example_index": case.example_index,
        "input_image_path": str(case.input_image_path),
        "input_mask_path": str(case.input_mask_path),
        "gt_operation_sequence_path": str(case.gt_operation_sequence_path),
        "used_official_prompt": True,
        "used_official_scorer": True,
        "gt_used_for_prompt": False,
    }
    messages = build_official_messages(case=case, official_prompt=official_prompt, with_cot=args.with_cot)
    row["message_block_count"] = len(messages[0]["content"])
    if args.score_only:
        existing_path = row_output_path(args.output_root, case_id)
        previous = json.loads(existing_path.read_text(encoding="utf-8"))
        raw_content = previous.get("origin_output", "")
        row["score_only_source"] = str(existing_path)
        row["elapsed_sec"] = 0
    else:
        response = call_openai_compatible_vlm(
            base_url=args.base_url,
            model=args.model,
            messages=messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_sec=args.timeout_sec,
            response_format_json=args.response_format_json,
        )
        raw_content = response["content"]
        row["elapsed_sec"] = response["elapsed_sec"]
        row["usage"] = response.get("usage")
        row["served_model"] = response.get("served_model")
        row["raw_response"] = response.get("raw_response")
    parsed_output = parse_official_skill_sequence(raw_content)
    score = score_with_official_vlabench(
        task_name=case.task_name,
        gt_skill_sequence=case.gt_skill_sequence,
        model_output=parsed_output,
        seq_independent_tasks=seq_independent_tasks,
    )
    type_normalized_score = score_with_type_normalized_vlabench(
        task_name=case.task_name,
        gt_skill_sequence=case.gt_skill_sequence,
        model_output=parsed_output,
        seq_independent_tasks=seq_independent_tasks,
    )
    row.update(parsed_output)
    row.update(score)
    row.update(type_normalized_score)
    row["ok"] = True
    row["format_error"] = "format_error" in parsed_output
    row["gt_skill_sequence"] = case.gt_skill_sequence
    return row


def main() -> int:
    """Run the official-style VLABench VLM symbolic evaluator."""

    args = parse_args()
    started_at = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    vlabench_root = ensure_vlabench_root()
    seq_independent_tasks = load_seq_independent_tasks()
    official_prompt = args.prompt_path.read_text(encoding="utf-8")
    manifest = load_manifest(args.manifest)
    selected_case_ids = select_case_ids(manifest=manifest, max_cases=args.max_cases, case_list_file=args.case_list_file)
    write_json(
        args.output_root / "run_config.json",
        {
            "vlabench_root": str(vlabench_root),
            "selected_case_count": len(selected_case_ids),
            "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        },
    )
    rows: list[dict[str, Any]] = []
    jsonl_path = args.output_root / "official_vlm_symbolic_rows.jsonl"
    if jsonl_path.exists() and not args.resume:
        jsonl_path.unlink()
    for index, case_id in enumerate(selected_case_ids, start=1):
        case_output_path = row_output_path(args.output_root, case_id)
        if args.resume and case_output_path.exists():
            row = json.loads(case_output_path.read_text(encoding="utf-8"))
        else:
            try:
                row = run_one_case(
                    case_id=case_id,
                    args=args,
                    official_prompt=official_prompt,
                    seq_independent_tasks=seq_independent_tasks,
                )
            except Exception as exc:  # noqa: BLE001 - per-case audit must preserve unexpected failures.
                category, task_name, example_name = (case_id.split("/") + ["", "", ""])[:3]
                row = {
                    "case_id": case_id,
                    "category": category,
                    "task_name": task_name,
                    "example_name": example_name,
                    "ok": False,
                    "format_error": False,
                    "symbolic_success": False,
                    "skill_match_score": 0,
                    "entity_match_score": 0,
                    "skill_with_entity_match_score": 0,
                    "exact_match_score": 0,
                    "total_score": 0,
                    "official_dependency": "runtime_error",
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
                    "symbolic_success": row.get("symbolic_success"),
                    "format_error": row.get("format_error"),
                    "exact_match_score": row.get("exact_match_score", 0),
                    "total_score": row.get("total_score", 0),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.save_interval > 0 and index % args.save_interval == 0:
            write_csv(args.output_root / "official_vlm_symbolic_scores.csv", rows)
            write_json(args.output_root / "summary.json", summarize(rows, started_at=started_at, args=args))
    write_csv(args.output_root / "official_vlm_symbolic_scores.csv", rows)
    summary = summarize(rows, started_at=started_at, args=args)
    write_json(args.output_root / "summary.json", summary)
    nested_final_score: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        task_name = str(row.get("task_name", ""))
        example_index = str(row.get("example_name", "")).replace("example", "")
        nested_final_score.setdefault(task_name, {})
        nested_final_score[task_name][example_index] = {
            "skill_match_score": row.get("skill_match_score", 0),
            "entity_match_score": row.get("entity_match_score", 0),
            "skill_with_entity_match_score": row.get("skill_with_entity_match_score", 0),
            "exact_match_score": row.get("exact_match_score", 0),
            "total_score": row.get("total_score", 0),
        }
    write_json(args.output_root / "final_score_official_format.json", nested_final_score)
    nested_type_normalized_score: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        task_name = str(row.get("task_name", ""))
        example_index = str(row.get("example_name", "")).replace("example", "")
        nested_type_normalized_score.setdefault(task_name, {})
        nested_type_normalized_score[task_name][example_index] = {
            "skill_match_score": row.get("type_normalized_skill_match_score", 0),
            "entity_match_score": row.get("type_normalized_entity_match_score", 0),
            "skill_with_entity_match_score": row.get("type_normalized_skill_with_entity_match_score", 0),
            "exact_match_score": row.get("type_normalized_exact_match_score", 0),
            "total_score": row.get("type_normalized_total_score", 0),
        }
    write_json(args.output_root / "final_score_type_normalized_format.json", nested_type_normalized_score)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
