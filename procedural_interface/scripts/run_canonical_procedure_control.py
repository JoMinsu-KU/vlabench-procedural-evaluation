"""Run the canonical-procedure control over the fixed 2,133-case subset.

Fixtures are constructed offline from reference metadata. Runtime execution
consumes only frozen procedural steps and numeric targets.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any

from procedural_interface.evaluation.runner import _run_one_fixture
from procedural_interface.adapter.task_phase_templates import TASK_TEMPLATES


DEFAULT_DATASET = None
DEFAULT_CASE_LIST_JSON = PROJECT_ROOT / "datasets" / "manifests" / "verified_subset_2133_cases.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "procedural_eval" / "verified_subset_adapter_smoke"
DEFAULT_TIMEOUT = 240


def sort_case_ids(cases: list[str]) -> list[str]:
    return sorted(cases, key=lambda case_id: (case_id.split("/")[0], case_id.split("/")[1], int(case_id.split("/")[2].replace("example", ""))))


def read_case_list_json(case_list_json: Path) -> list[str]:
    cases = json.loads(case_list_json.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError(f"case list JSON must contain a list: {case_list_json}")
    return sort_case_ids([str(case_id) for case_id in cases])


def read_legacy_pair_csv(pair_csv: Path) -> list[str]:
    with pair_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    cases = [row["sample_id"] for row in rows if row.get("ledger_official_status") == "B0_official_eligible"]
    return sort_case_ids(cases)


def component_index(env_config: dict[str, Any], name: str) -> int:
    components = env_config.get("task", {}).get("components", [])
    for index, component in enumerate(components):
        if component.get("name") == name:
            return index
    raise ValueError(f"component not found: {name}")


def first_condition_entity(task: dict[str, Any]) -> str:
    for condition_name in ["contain", "not_contain", "is_grasped", "lift"]:
        entities = task.get("conditions", {}).get(condition_name, {}).get("entities")
        if isinstance(entities, list) and entities:
            return entities[0]
    for condition_name in ["pour", "above"]:
        target_entity = task.get("conditions", {}).get(condition_name, {}).get("target_entity")
        if isinstance(target_entity, str):
            return target_entity
    raise ValueError("cannot resolve target entity from conditions")


def condition_container(task: dict[str, Any]) -> str:
    for condition_name in ["contain", "not_contain"]:
        container = task.get("conditions", {}).get(condition_name, {}).get("container")
        if isinstance(container, str):
            return container
    platform = task.get("conditions", {}).get("above", {}).get("platform")
    if isinstance(platform, str):
        return platform
    raise ValueError("cannot resolve target container/platform from conditions")


def press_button(task: dict[str, Any]) -> str:
    button = task.get("conditions", {}).get("press_button", {}).get("target_button")
    if isinstance(button, str):
        return button
    raise ValueError("cannot resolve press button")


def build_steps(task_name: str, env_config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[int, str]]:
    task = env_config.get("task", {})
    template = TASK_TEMPLATES[task_name]
    targets: dict[int, str] = {}
    entity_name = None
    container_name = None
    button_name = None
    if any(phase.expected_skill == "pick" for phase in template.phases):
        entity_name = first_condition_entity(task)
    if any(phase.expected_skill in {"place", "moveto"} for phase in template.phases):
        container_name = condition_container(task)
    if any(phase.expected_skill == "press" for phase in template.phases):
        button_name = press_button(task)
    steps: list[dict[str, Any]] = []
    for phase in template.phases:
        skill = phase.expected_skill
        if skill == "pick":
            assert entity_name is not None
            target_number = component_index(env_config, entity_name)
            steps.append({"skill": skill, "target": target_number})
            targets[target_number] = entity_name
        elif skill in {"place", "moveto"}:
            assert container_name is not None
            target_number = component_index(env_config, container_name)
            steps.append({"skill": skill, "target": target_number})
            targets[target_number] = container_name
        elif skill == "press":
            assert button_name is not None
            target_number = component_index(env_config, button_name)
            steps.append({"skill": skill, "target": target_number})
            targets[target_number] = button_name
        else:
            steps.append({"skill": skill})
    return steps, targets


def make_fixture(case_id: str, env_config: dict[str, Any]) -> dict[str, Any]:
    category, task_name, example_id = case_id.split("/")
    steps, targets = build_steps(task_name, env_config)
    return {
        "fixture_id": f"{category}_{task_name}_{example_id}_positive",
        "fixture_version": "canonical_procedure_control_v1",
        "fixture_mode": "positive_control",
        "family": f"{category}/{task_name}",
        "fixture_generated_from_gt": True,
        "fixture_generation_phase": "offline_gt_metadata",
        "source_fields_used": ["conditions", "components"],
        "frozen_after_generation": True,
        "dataset_snapshot": "vlabench_vlm4500_public_20260618",
        "source_case_id": case_id,
        "target_source": "frozen_numeric_fixture",
        "counts_as_model_evaluation": False,
        "expected_gate1_ok": True,
        "expected_native_ok": True,
        "expected_condition_success": True,
        "negative_type": "",
        "expected_failure_code": "",
        "expected_failure_subtype": "",
        "native_execution_should_run": True,
        "steps": steps,
        "human_readable_targets": {str(number): name for number, name in sorted(targets.items())},
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]], selected_cases: list[str], output_root: Path) -> dict[str, Any]:
    failed = [row for row in rows if str(row.get("passed", "")).lower() != "true"]
    by_family: dict[str, dict[str, int]] = {}
    by_category: dict[str, dict[str, int]] = {}
    for row in rows:
        family = row.get("family", "")
        category = row.get("category", "")
        passed = str(row.get("passed", "")).lower() == "true"
        fam = by_family.setdefault(family, {"total": 0, "passed": 0, "failed": 0})
        cat = by_category.setdefault(category, {"total": 0, "passed": 0, "failed": 0})
        fam["total"] += 1
        cat["total"] += 1
        fam["passed" if passed else "failed"] += 1
        cat["passed" if passed else "failed"] += 1
    return {
        "status": "passed" if not failed and len(rows) == len(selected_cases) else "failed",
        "case_source": "fixed execution-verified 2,133-case manifest",
        "total_selected_cases": len(selected_cases),
        "completed_cases": len(rows),
        "passed_cases": len(rows) - len(failed),
        "failed_cases": len(failed),
        "failed_case_ids": [row.get("source_case_id", "") for row in failed],
        "by_category": by_category,
        "by_family": by_family,
        "output_root": str(output_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical-procedure control over the fixed 2,133-case subset.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--case-list-json", type=Path, default=DEFAULT_CASE_LIST_JSON)
    parser.add_argument("--pair-csv", type=Path, default=None, help="Optional legacy selection ledger for compatibility.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--case-id", action="append", default=[], help="Run only the exact source case id. Can be repeated.")
    parser.add_argument("--case-list-file", type=Path, default=None, help="Read exact source case ids from a newline-delimited text file.")
    parser.add_argument("--task-name", action="append", default=[], help="Run only selected task names. Can be repeated.")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    fixture_root = args.output_root / "fixtures"
    trace_root = args.output_root / "smoke_execution_traces"
    fixture_root.mkdir(parents=True, exist_ok=True)
    trace_root.mkdir(parents=True, exist_ok=True)

    selected_cases = read_legacy_pair_csv(args.pair_csv) if args.pair_csv else read_case_list_json(args.case_list_json)
    requested_case_ids = list(args.case_id)
    if args.case_list_file:
        requested_case_ids.extend([line.strip() for line in args.case_list_file.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")])
    if requested_case_ids:
        requested_cases = set(requested_case_ids)
        selected_cases = [case_id for case_id in selected_cases if case_id in requested_cases]
    if args.task_name:
        requested_tasks = set(args.task_name)
        selected_cases = [case_id for case_id in selected_cases if case_id.split("/")[1] in requested_tasks]
    if args.max_cases and args.max_cases > 0:
        selected_cases = selected_cases[: args.max_cases]
    (args.output_root / "selected_cases.json").write_text(json.dumps(selected_cases, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    fixtures: list[dict[str, Any]] = []
    for index, case_id in enumerate(selected_cases, start=1):
        category, task_name, example_id = case_id.split("/")
        env_config_path = args.dataset_root / category / task_name / example_id / "env_config.json"
        env_config = json.loads(env_config_path.read_text(encoding="utf-8"))
        fixture = make_fixture(case_id, env_config)
        fixtures.append(fixture)
        (fixture_root / f"{fixture['fixture_id']}.json").write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.prepare_only:
            print(json.dumps({"index": index, "total": len(selected_cases), "case_id": case_id, "prepared": True}, ensure_ascii=False), flush=True)
            continue
        row = _run_one_fixture(args.dataset_root, fixture, trace_root, timeout_sec=args.timeout_sec)
        row["category"] = category
        row["task_name"] = task_name
        row["family"] = f"{category}/{task_name}"
        row["sample_index"] = index
        rows.append(row)
        write_csv(args.output_root / "canonical_procedure_control_report.csv", rows)
        print(json.dumps({"index": index, "total": len(selected_cases), "case_id": case_id, "passed": row.get("passed"), "native": row.get("actual_native_ok"), "condition": row.get("actual_condition_success"), "failure": row.get("actual_failure_code"), "subtype": row.get("actual_failure_subtype")}, ensure_ascii=False), flush=True)

    (args.output_root / "fixtures.json").write_text(json.dumps(fixtures, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize(rows, selected_cases, args.output_root)
    if args.prepare_only:
        summary["status"] = "prepared_only"
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
