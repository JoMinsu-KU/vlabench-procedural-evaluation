from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = ANALYSIS_DIR.parent
CODE_ROOT = REPOSITORY_ROOT
FIXTURE_PATH = (
    REPOSITORY_ROOT / "results" / "positive_control" / "provenance" / "fixtures.json"
)
REPORT_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "validation_evidence"
    / "full_b_no_repair_validation_20260721.json"
)
TABLE_PATH = ANALYSIS_DIR / "tables" / "full_b_no_repair_mutation_controls.csv"

sys.path.insert(0, str(CODE_ROOT))

from procedural_interface.adapter.phase_binding import bind_plan_to_phases  # noqa: E402
from procedural_interface.adapter.task_phase_templates import TASK_TEMPLATES  # noqa: E402
from procedural_interface.interface.parser import parse_plan_text  # noqa: E402


def _plan_text(steps: list[dict]) -> str:
    return json.dumps({"steps": steps}, ensure_ascii=False)


def _synthetic_env(steps: list[dict]) -> dict:
    """Build named in-range components for target-preservation checks only."""

    targets = [int(step["target"]) for step in steps if "target" in step]
    component_count = max(targets, default=0) + 2
    return {
        "task": {
            "components": [
                {"name": f"synthetic_component_{index}", "class": "SyntheticObject"}
                for index in range(component_count)
            ]
        }
    }


def _bind(steps: list[dict], task_name: str, env_config: dict) -> dict:
    parsed = parse_plan_text(_plan_text(steps))
    if not parsed.get("ok"):
        return {"ok": False, "stage": "parser", "status": parsed["status"]}
    bound = bind_plan_to_phases(parsed["steps"], env_config, TASK_TEMPLATES[task_name])
    return {
        "ok": bool(bound.get("ok")),
        "stage": "binding",
        "status": bound.get("status", {}),
        "parsed_steps": parsed["steps"],
        "phase_trace": bound.get("phase_trace", []),
    }


def main() -> None:
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8-sig"))
    if len(fixtures) != 2_133:
        raise ValueError(f"expected 2,133 Full-B fixtures, found {len(fixtures)}")

    checks = {
        "missing_phase": 0,
        "adjacent_phase_swap": 0,
        "extra_phase": 0,
        "missing_target": 0,
        "alternative_target_preserved": 0,
    }
    errors: list[dict[str, str]] = []

    for fixture in fixtures:
        case_id = str(fixture["source_case_id"])
        task_name = case_id.split("/")[1]
        steps = copy.deepcopy(fixture["steps"])
        env_config = _synthetic_env(steps)

        if task_name not in TASK_TEMPLATES:
            errors.append({"case_id": case_id, "check": "template", "error": "missing task template"})
            continue

        baseline = _bind(steps, task_name, env_config)
        if not baseline.get("ok"):
            errors.append({"case_id": case_id, "check": "baseline", "error": str(baseline.get("status"))})
            continue

        missing_phase = _bind(steps[:-1], task_name, env_config)
        checks["missing_phase"] += 1
        if missing_phase.get("ok"):
            errors.append({"case_id": case_id, "check": "missing_phase", "error": "accepted"})

        if len(steps) > 1:
            swapped = copy.deepcopy(steps)
            swapped[0], swapped[1] = swapped[1], swapped[0]
            swap_result = _bind(swapped, task_name, env_config)
            checks["adjacent_phase_swap"] += 1
            if swap_result.get("ok"):
                errors.append({"case_id": case_id, "check": "adjacent_phase_swap", "error": "accepted"})

        extra = copy.deepcopy(steps)
        extra.append(copy.deepcopy(steps[-1]))
        extra_result = _bind(extra, task_name, env_config)
        checks["extra_phase"] += 1
        if extra_result.get("ok"):
            errors.append({"case_id": case_id, "check": "extra_phase", "error": "accepted"})

        target_index = next((index for index, step in enumerate(steps) if "target" in step), None)
        if target_index is None:
            errors.append({"case_id": case_id, "check": "target_coverage", "error": "no target-bearing step"})
            continue

        missing_target_steps = copy.deepcopy(steps)
        missing_target_steps[target_index].pop("target")
        missing_target_result = _bind(missing_target_steps, task_name, env_config)
        checks["missing_target"] += 1
        if missing_target_result.get("ok"):
            errors.append({"case_id": case_id, "check": "missing_target", "error": "accepted"})

        alternative_steps = copy.deepcopy(steps)
        original_target = int(alternative_steps[target_index]["target"])
        component_count = len(env_config["task"]["components"])
        alternative_target = (original_target + 1) % component_count
        alternative_steps[target_index]["target"] = alternative_target
        alternative_result = _bind(alternative_steps, task_name, env_config)
        checks["alternative_target_preserved"] += 1
        if not alternative_result.get("ok"):
            errors.append(
                {"case_id": case_id, "check": "alternative_target_preserved", "error": str(alternative_result.get("status"))}
            )
            continue
        trace_row = alternative_result["phase_trace"][target_index]
        target_binding = trace_row.get("target_binding") or {}
        if (
            int(trace_row.get("vlm_target")) != alternative_target
            or int(target_binding.get("target_number")) != alternative_target
            or bool(trace_row.get("sequence_reordered"))
            or bool(trace_row.get("missing_phase_inserted"))
            or bool(trace_row.get("ground_truth_target_used"))
        ):
            errors.append(
                {
                    "case_id": case_id,
                    "check": "alternative_target_preserved",
                    "error": "selected target or ownership trace changed",
                }
            )

    total_checks = sum(checks.values())
    report = {
        "status": "passed" if not errors else "failed",
        "case_count": len(fixtures),
        "task_count": len({str(fixture["source_case_id"]).split("/")[1] for fixture in fixtures}),
        "source_fixture": FIXTURE_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
        "source_code": ".",
        "checks": checks,
        "total_mutation_checks": total_checks,
        "validation_error_count": len(errors),
        "semantic_repair_applied": False,
        "scope_note": (
            "The alternative-target check uses a synthetic named component registry to test numeric-target "
            "preservation in the Full-B parser and phase binder. It is not an affordance or task-success test."
        ),
        "errors": errors,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = [
        {
            "mutation_control": "Full B: remove final phase",
            "checks": checks["missing_phase"],
            "expected_behavior": "Reject without phase insertion",
            "validation_errors": sum(error["check"] == "missing_phase" for error in errors),
        },
        {
            "mutation_control": "Full B: swap first two phases when possible",
            "checks": checks["adjacent_phase_swap"],
            "expected_behavior": "Reject without sequence reordering",
            "validation_errors": sum(error["check"] == "adjacent_phase_swap" for error in errors),
        },
        {
            "mutation_control": "Full B: append one extra phase",
            "checks": checks["extra_phase"],
            "expected_behavior": "Reject without phase deletion or skipping",
            "validation_errors": sum(error["check"] == "extra_phase" for error in errors),
        },
        {
            "mutation_control": "Full B: remove one required target",
            "checks": checks["missing_target"],
            "expected_behavior": "Reject without target insertion",
            "validation_errors": sum(error["check"] == "missing_target" for error in errors),
        },
        {
            "mutation_control": "Full B: substitute one in-range numeric target",
            "checks": checks["alternative_target_preserved"],
            "expected_behavior": "Preserve the selected target without GT replacement",
            "validation_errors": sum(error["check"] == "alternative_target_preserved" for error in errors),
        },
    ]
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLE_PATH, index=False)

    if errors:
        raise RuntimeError(f"Full-B no-repair validation failed with {len(errors)} errors")
    if checks != {
        "missing_phase": 2_133,
        "adjacent_phase_swap": 2_091,
        "extra_phase": 2_133,
        "missing_target": 2_133,
        "alternative_target_preserved": 2_133,
    }:
        raise ValueError(f"unexpected mutation coverage: {checks}")
    if total_checks != 10_623:
        raise ValueError(f"expected 10,623 checks, found {total_checks}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
