"""Cross-score frozen Full-Procedure outputs under the symbolic endpoint.

This analysis never calls a model and never executes MuJoCo.  It projects the
already archived Full-Procedure output onto the released symbolic schema using
a frozen, task-aware, information-reducing mapping.  It does not insert a
missing model-owned target, reorder actions, or substitute a reference target.

The reverse Symbolic-to-Native direction is deliberately not implemented here:
the released symbolic branch is a scorer, and the repository contains no
verified model-choice-preserving symbolic-to-native executor entry point.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reproducibility_core" / "cross_scoring"

MODELS = {
    "Qwen3-VL-8B": {
        "symbolic": ROOT / "results/model_outputs/qwen/experiment_a/official_vlm_symbolic_rows.jsonl",
        "full": ROOT / "results/model_outputs/qwen/experiment_b/procedural_vlm_execution_rows.jsonl",
    },
    "Gemma-3-12B-it": {
        "symbolic": ROOT / "results/model_outputs/gemma/experiment_a/official_vlm_symbolic_rows.jsonl",
        "full": ROOT / "results/model_outputs/gemma/experiment_b/procedural_vlm_execution_rows.jsonl",
    },
    "InternVL3-8B": {
        "symbolic": ROOT / "results/model_outputs/internvl/experiment_a/official_vlm_symbolic_rows.jsonl",
        "full": ROOT / "results/model_outputs/internvl/experiment_b/procedural_vlm_execution_rows.jsonl",
    },
}

INSERT_TASKS = {
    "insert_bloom_flower",
    "insert_flower",
    "insert_flower_common_sense",
    "insert_flower_semantic",
}

DROP_LIFT_TASKS = {
    "select_drink",
    "select_drink_common_sense",
    "select_drink_semantic",
    "select_toy",
    "select_toy_common_sense",
    "select_toy_semantic",
    "select_toy_spatial",
}

ADD_CONDIMENT_TASKS = {
    "add_condiment",
    "add_condiment_common_sense",
    "add_condiment_semantic",
    "add_condiment_spatial",
}

PROJECTION_SPEC = {
    "protocol_version": "frozen_full_to_symbolic_v1",
    "primary_endpoint": "type_normalized_symbolic_exact_match",
    "input_policy": "archived type-normalized Full-Procedure parse only",
    "default": "identity projection; preserve emitted order, skill labels, and targets",
    "drop_lift_tasks": sorted(DROP_LIFT_TASKS),
    "insert_tasks": {
        "tasks": sorted(INSERT_TASKS),
        "mapping": "drop lift; relabel each emitted moveto(target) as insert(target); preserve all other emitted actions",
        "rationale": "source-derived macro projection from the executor phase vocabulary to the symbolic contract",
    },
    "add_condiment_tasks": {
        "tasks": sorted(ADD_CONDIMENT_TASKS),
        "mapping": "drop lift; consume exactly one emitted moveto target as the emitted pour target; preserve all other emitted actions",
    },
    "forbidden": [
        "target replacement",
        "action reordering",
        "missing target insertion",
        "reference action substitution",
        "invalid output recovery",
    ],
}

# Pattern order matches the pinned VLABench symbolic evaluator.  The retained
# manifest uses Sequential dependency for all 2,133 cases.
SUBTASK_PATTERNS = [
    ["pick", "place"],
    ["pick", "insert"],
    ["pick", "pour", "place"],
    ["pick", "pour"],
    ["pick", "pull"],
    ["pick", "lift"],
    ["pick", "push", "pull"],
    ["pick", "push", "place"],
    ["pick", "push"],
    ["pick", "open_door"],
    ["press"],
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def canonicalize_numeric_target(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.isdigit():
        return int(stripped)
    return value


def normalize_symbolic_sequence(sequence: Any) -> list[dict[str, Any]] | None:
    if not isinstance(sequence, list):
        return None
    normalized: list[dict[str, Any]] = []
    for step in sequence:
        if not isinstance(step, dict) or not isinstance(step.get("name"), str):
            return None
        params = step.get("params") or {}
        if not isinstance(params, dict):
            return None
        normalized.append(
            {
                "name": step["name"],
                "params": {key: canonicalize_numeric_target(value) for key, value in params.items()},
            }
        )
    return normalized


def find_subtasks(sequence: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    names = [step["name"] for step in sequence]
    subtasks: list[list[dict[str, Any]]] = []
    index = 0
    while index < len(names):
        for pattern in SUBTASK_PATTERNS:
            if names[index : index + len(pattern)] == pattern:
                subtasks.append(sequence[index : index + len(pattern)])
                index += len(pattern)
                break
        else:
            index += 1
    return subtasks


def subtask_signature(subtask: list[dict[str, Any]]) -> tuple[str, Any, Any]:
    name = "-".join(step["name"] for step in subtask)
    entities = [step["params"].get("target_entity_name") for step in subtask]
    containers = [step["params"].get("target_container_name") for step in subtask]
    entities = [value for value in entities if value is not None]
    containers = [value for value in containers if value is not None]
    return name, entities[-1] if entities else None, containers[-1] if containers else None


def symbolic_exact_match(model_sequence: Any, gt_sequence: Any) -> bool:
    """Reproduce the pinned official graph exact-match endpoint for Sequential tasks."""

    model = normalize_symbolic_sequence(model_sequence)
    gt = normalize_symbolic_sequence(gt_sequence)
    if model is None or gt is None:
        return False
    standard_subtasks = [subtask_signature(item) for item in find_subtasks(gt)]
    model_subtasks = [subtask_signature(item) for item in find_subtasks(model)]
    if not standard_subtasks:
        return False
    exact_count = 0
    for index, standard in enumerate(standard_subtasks):
        if index >= len(model_subtasks) or model_subtasks[index] != standard:
            continue
        if index == 0 or model_subtasks[index - 1] == standard_subtasks[index - 1]:
            exact_count += 1
    return exact_count == len(standard_subtasks)


def target_param_key(skill: str) -> str:
    if skill in {"place", "insert", "pour"}:
        return "target_container_name"
    return "target_entity_name"


def emit_symbolic_step(skill: str, target_present: bool, target: Any) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if target_present:
        params[target_param_key(skill)] = canonicalize_numeric_target(target)
    return {"name": skill, "params": params}


def project_full_steps(task_name: str, steps: Any) -> tuple[bool, str, list[dict[str, Any]]]:
    """Apply the frozen lossy projection without semantic repair."""

    if not isinstance(steps, list):
        return False, "model_output_unprojectable", []
    parsed: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict) or not isinstance(step.get("skill"), str):
            return False, "model_output_unprojectable", []
        parsed.append(step)

    if task_name in ADD_CONDIMENT_TASKS:
        moveto_steps = [step for step in parsed if step["skill"] == "moveto"]
        if len(moveto_steps) > 1:
            return False, "ambiguous_multiple_moveto_targets", []
        pending_target_present = bool(moveto_steps and "target" in moveto_steps[0])
        pending_target = moveto_steps[0].get("target") if moveto_steps else None
        projected: list[dict[str, Any]] = []
        for step in parsed:
            skill = step["skill"]
            if skill in {"lift", "moveto"}:
                continue
            if skill == "pour":
                projected.append(emit_symbolic_step("pour", pending_target_present, pending_target))
            else:
                projected.append(emit_symbolic_step(skill, "target" in step, step.get("target")))
        return True, "projected_add_condiment_macro", projected

    if task_name in INSERT_TASKS:
        projected = []
        for step in parsed:
            skill = step["skill"]
            if skill == "lift":
                continue
            if skill == "moveto":
                projected.append(emit_symbolic_step("insert", "target" in step, step.get("target")))
            else:
                projected.append(emit_symbolic_step(skill, "target" in step, step.get("target")))
        return True, "projected_insert_macro", projected

    if task_name in DROP_LIFT_TASKS:
        projected = [
            emit_symbolic_step(step["skill"], "target" in step, step.get("target"))
            for step in parsed
            if step["skill"] != "lift"
        ]
        return True, "projected_drop_intermediate_lift", projected

    projected = [
        emit_symbolic_step(step["skill"], "target" in step, step.get("target"))
        for step in parsed
    ]
    return True, "projected_identity", projected


def validate_archived_symbolic_endpoint(symbolic_rows: list[dict[str, Any]]) -> int:
    mismatches = 0
    for row in symbolic_rows:
        reproduced = symbolic_exact_match(row.get("skill_sequence"), row.get("gt_skill_sequence"))
        if reproduced != bool(row.get("type_normalized_symbolic_success")):
            mismatches += 1
    return mismatches


def no_repair_controls() -> list[dict[str, Any]]:
    controls: list[dict[str, Any]] = []

    def record(name: str, task: str, source: list[dict[str, Any]], expected: list[dict[str, Any]]) -> None:
        ok, status, projected = project_full_steps(task, source)
        controls.append(
            {
                "control": name,
                "passed": ok and projected == expected,
                "status": status,
                "source": source,
                "projected": projected,
                "expected": expected,
            }
        )

    record(
        "wrong_target_preserved",
        "select_toy",
        [{"skill": "pick", "target": 9}, {"skill": "lift"}, {"skill": "place", "target": 1}],
        [emit_symbolic_step("pick", True, 9), emit_symbolic_step("place", True, 1)],
    )
    record(
        "order_swap_preserved",
        "select_toy",
        [{"skill": "place", "target": 1}, {"skill": "pick", "target": 2}],
        [emit_symbolic_step("place", True, 1), emit_symbolic_step("pick", True, 2)],
    )
    record(
        "missing_action_not_inserted",
        "select_toy",
        [{"skill": "pick", "target": 2}],
        [emit_symbolic_step("pick", True, 2)],
    )
    record(
        "unexpected_action_preserved",
        "select_toy",
        [{"skill": "pick", "target": 2}, {"skill": "push", "target": 4}, {"skill": "place", "target": 1}],
        [emit_symbolic_step("pick", True, 2), emit_symbolic_step("push", True, 4), emit_symbolic_step("place", True, 1)],
    )
    record(
        "missing_container_target_not_repaired",
        "add_condiment",
        [{"skill": "pick", "target": 2}, {"skill": "pour"}],
        [emit_symbolic_step("pick", True, 2), emit_symbolic_step("pour", False, None)],
    )
    return controls


def percentage(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def jaccard(both: int, left_only: int, right_only: int) -> float:
    denominator = both + left_only + right_only
    return percentage(both, denominator)


def first_attempt_condition_success(row: dict[str, Any]) -> bool:
    summaries = row.get("native_attempt_summaries")
    if isinstance(summaries, str):
        try:
            summaries = json.loads(summaries)
        except json.JSONDecodeError:
            summaries = []
    if isinstance(summaries, list) and summaries:
        first = summaries[0]
        if isinstance(first, dict):
            return bool(first.get("condition_success"))
    return False


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    spec_path = OUTPUT_DIR / "projection_spec.json"
    spec_path.write_text(json.dumps(PROJECTION_SPEC, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    controls = no_repair_controls()
    if not all(row["passed"] for row in controls):
        raise AssertionError("A frozen projection no-repair control failed")
    (OUTPUT_DIR / "projection_no_repair_controls.json").write_text(
        json.dumps(controls, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    validation_mismatches: dict[str, int] = {}

    for model, paths in MODELS.items():
        symbolic_rows = read_jsonl(paths["symbolic"])
        full_rows = read_jsonl(paths["full"])
        if len(symbolic_rows) != 2133 or len(full_rows) != 2133:
            raise AssertionError(f"Unexpected row count for {model}")
        symbolic_by_case = {row["case_id"]: row for row in symbolic_rows}
        full_by_case = {row["case_id"]: row for row in full_rows}
        if set(symbolic_by_case) != set(full_by_case):
            raise AssertionError(f"Case mismatch for {model}")

        mismatch_count = validate_archived_symbolic_endpoint(symbolic_rows)
        validation_mismatches[model] = mismatch_count
        if mismatch_count:
            raise AssertionError(f"Symbolic endpoint reproduction mismatch for {model}: {mismatch_count}")

        for case_id in sorted(symbolic_by_case):
            sym = symbolic_by_case[case_id]
            full = full_by_case[case_id]
            parser_accepted = bool(full.get("type_normalized_parse_ok"))
            steps = full.get("normalized_steps") if parser_accepted else None
            projectable, projection_status, projected_sequence = project_full_steps(full["task_name"], steps)
            projected_success = projectable and symbolic_exact_match(projected_sequence, sym.get("gt_skill_sequence"))
            original_symbolic_success = bool(sym.get("type_normalized_symbolic_success"))
            # The row-level final fields select the first successful retry when
            # one exists.  Recover the manuscript's primary first-attempt label
            # from the archived attempt summaries instead.
            full_success = first_attempt_condition_success(full)
            detail_rows.append(
                {
                    "model": model,
                    "case_id": case_id,
                    "category": full["category"],
                    "task_name": full["task_name"],
                    "full_parser_accepted": parser_accepted,
                    "projection_supported": projectable,
                    "projection_status": projection_status,
                    "projected_symbolic_sequence": json.dumps(projected_sequence, ensure_ascii=False, separators=(",", ":")),
                    "original_symbolic_success": original_symbolic_success,
                    "projected_full_symbolic_success": projected_success,
                    "full_procedure_success": full_success,
                    "projected_success_full_failure": projected_success and not full_success,
                    "original_symbolic_projected_agreement": original_symbolic_success == projected_success,
                }
            )

        model_rows = [row for row in detail_rows if row["model"] == model]
        n = len(model_rows)
        projectable_n = sum(row["projection_supported"] for row in model_rows)
        original_n = sum(row["original_symbolic_success"] for row in model_rows)
        projected_n = sum(row["projected_full_symbolic_success"] for row in model_rows)
        full_n = sum(row["full_procedure_success"] for row in model_rows)
        projected_full_failure_n = sum(row["projected_success_full_failure"] for row in model_rows)
        full_success_projected_failure_n = sum(
            row["full_procedure_success"] and not row["projected_full_symbolic_success"]
            for row in model_rows
        )
        agreement_n = sum(row["original_symbolic_projected_agreement"] for row in model_rows)
        both = sum(row["original_symbolic_success"] and row["projected_full_symbolic_success"] for row in model_rows)
        original_only = sum(row["original_symbolic_success"] and not row["projected_full_symbolic_success"] for row in model_rows)
        projected_only = sum(not row["original_symbolic_success"] and row["projected_full_symbolic_success"] for row in model_rows)
        summary_rows.append(
            {
                "model": model,
                "n": n,
                "full_output_projection_possible_n": projectable_n,
                "full_output_projection_possible_pct": f"{percentage(projectable_n, n):.2f}",
                "original_symbolic_success_n": original_n,
                "original_symbolic_success_pct": f"{percentage(original_n, n):.2f}",
                "projected_full_symbolic_success_n": projected_n,
                "projected_full_symbolic_success_pct": f"{percentage(projected_n, n):.2f}",
                "full_procedure_success_n": full_n,
                "full_procedure_success_pct": f"{percentage(full_n, n):.2f}",
                "projected_success_full_failure_n": projected_full_failure_n,
                "projected_success_full_failure_pct": f"{percentage(projected_full_failure_n, n):.2f}",
                "full_success_projected_failure_n": full_success_projected_failure_n,
                "original_vs_projected_agreement_n": agreement_n,
                "original_vs_projected_agreement_pct": f"{percentage(agreement_n, n):.2f}",
                "original_vs_projected_both_n": both,
                "original_only_n": original_only,
                "projected_only_n": projected_only,
                "original_vs_projected_jaccard_pct": f"{jaccard(both, original_only, projected_only):.2f}",
            }
        )

    pooled_n = len(detail_rows)
    pooled_projectable = sum(row["projection_supported"] for row in detail_rows)
    pooled_original = sum(row["original_symbolic_success"] for row in detail_rows)
    pooled_projected = sum(row["projected_full_symbolic_success"] for row in detail_rows)
    pooled_full = sum(row["full_procedure_success"] for row in detail_rows)
    pooled_projected_full_failure = sum(row["projected_success_full_failure"] for row in detail_rows)
    pooled_full_success_projected_failure = sum(
        row["full_procedure_success"] and not row["projected_full_symbolic_success"]
        for row in detail_rows
    )
    pooled_agreement = sum(row["original_symbolic_projected_agreement"] for row in detail_rows)
    pooled_both = sum(row["original_symbolic_success"] and row["projected_full_symbolic_success"] for row in detail_rows)
    pooled_original_only = sum(row["original_symbolic_success"] and not row["projected_full_symbolic_success"] for row in detail_rows)
    pooled_projected_only = sum(not row["original_symbolic_success"] and row["projected_full_symbolic_success"] for row in detail_rows)
    summary_rows.append(
        {
            "model": "Pooled",
            "n": pooled_n,
            "full_output_projection_possible_n": pooled_projectable,
            "full_output_projection_possible_pct": f"{percentage(pooled_projectable, pooled_n):.2f}",
            "original_symbolic_success_n": pooled_original,
            "original_symbolic_success_pct": f"{percentage(pooled_original, pooled_n):.2f}",
            "projected_full_symbolic_success_n": pooled_projected,
            "projected_full_symbolic_success_pct": f"{percentage(pooled_projected, pooled_n):.2f}",
            "full_procedure_success_n": pooled_full,
            "full_procedure_success_pct": f"{percentage(pooled_full, pooled_n):.2f}",
            "projected_success_full_failure_n": pooled_projected_full_failure,
            "projected_success_full_failure_pct": f"{percentage(pooled_projected_full_failure, pooled_n):.2f}",
            "full_success_projected_failure_n": pooled_full_success_projected_failure,
            "original_vs_projected_agreement_n": pooled_agreement,
            "original_vs_projected_agreement_pct": f"{percentage(pooled_agreement, pooled_n):.2f}",
            "original_vs_projected_both_n": pooled_both,
            "original_only_n": pooled_original_only,
            "projected_only_n": pooled_projected_only,
            "original_vs_projected_jaccard_pct": f"{jaccard(pooled_both, pooled_original_only, pooled_projected_only):.2f}",
        }
    )

    detail_path = OUTPUT_DIR / "full_to_symbolic_projection_rows.csv"
    summary_path = OUTPUT_DIR / "full_to_symbolic_summary.csv"
    write_csv(detail_path, detail_rows)
    write_csv(summary_path, summary_rows)
    write_csv(ROOT / "results" / "reported_statistics" / "frozen_output_cross_scoring.csv", summary_rows)

    status_counts = Counter(row["projection_status"] for row in detail_rows)
    audit = {
        "protocol_version": PROJECTION_SPEC["protocol_version"],
        "model_inference_calls": 0,
        "native_rollouts": 0,
        "input_rows": pooled_n,
        "symbolic_endpoint_reproduction_mismatches": validation_mismatches,
        "projection_status_counts": dict(sorted(status_counts.items())),
        "no_repair_controls_passed": sum(row["passed"] for row in controls),
        "no_repair_controls_total": len(controls),
        "projection_spec_sha256": sha256(spec_path),
        "reverse_symbolic_to_native": {
            "performed": False,
            "reason": "No verified model-choice-preserving symbolic-to-native executor entry point exists in the released evaluation code; constructing one would add a new adapter and native experiment.",
        },
    }
    (OUTPUT_DIR / "cross_scoring_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"summary": summary_rows, "audit": audit}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
