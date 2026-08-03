#!/usr/bin/env python3
"""Build a source-linked structural audit of the frozen procedure registry.

This script does not modify the registry or any case-level result. It compares
the 34 retained task types with a pinned upstream VLABench source checkout and
emits a draft crosswalk for documented author-led semantic review.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import runpy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TARGET_SKILLS = {"pick", "moveto", "place", "press", "push"}


@dataclass
class ClassRecord:
    name: str
    module_path: str
    source_path: Path
    line: int
    bases: list[str]
    registered_tasks: list[str]
    expert_method: ast.FunctionDef | ast.AsyncFunctionDef | None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def registered_task_names(node: ast.ClassDef) -> list[str]:
    names: list[str] = []
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        if not dotted_name(decorator.func).endswith("add_task"):
            continue
        if decorator.args and isinstance(decorator.args[0], ast.Constant) and isinstance(decorator.args[0].value, str):
            names.append(decorator.args[0].value)
    return names


def collect_classes(source_root: Path) -> tuple[dict[str, ClassRecord], dict[str, str]]:
    classes: dict[str, ClassRecord] = {}
    task_to_class: dict[str, str] = {}
    hierarchical_root = source_root / "VLABench" / "tasks" / "hierarchical_tasks"
    for path in sorted(hierarchical_root.rglob("*.py")):
        relative = path.relative_to(hierarchical_root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            method = next(
                (
                    item
                    for item in node.body
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == "get_expert_skill_sequence"
                ),
                None,
            )
            record = ClassRecord(
                name=node.name,
                module_path=relative,
                source_path=path,
                line=node.lineno,
                bases=[dotted_name(base).split(".")[-1] for base in node.bases],
                registered_tasks=registered_task_names(node),
                expert_method=method,
            )
            if node.name in classes:
                existing = classes[node.name]
                # Config-manager helper names can recur across task modules. They
                # are outside the expert-skill inheritance graph. A duplicate is
                # only ambiguous for this audit when either definition registers
                # a task or defines the expert method being traced.
                if (
                    existing.registered_tasks
                    or record.registered_tasks
                    or existing.expert_method is not None
                    or record.expert_method is not None
                ):
                    raise RuntimeError(
                        "ambiguous duplicate task/expert class in audit scope: "
                        f"{node.name} ({existing.module_path}, {record.module_path})"
                    )
                continue
            classes[node.name] = record
            for task_name in record.registered_tasks:
                if task_name in task_to_class:
                    raise RuntimeError(f"duplicate task registration: {task_name}")
                task_to_class[task_name] = node.name
    return classes, task_to_class


def resolve_expert_method(
    class_name: str,
    classes: dict[str, ClassRecord],
    trail: list[str] | None = None,
) -> tuple[ClassRecord | None, list[str]]:
    trail = list(trail or [])
    if class_name in trail:
        raise RuntimeError(f"inheritance cycle: {' -> '.join(trail + [class_name])}")
    trail.append(class_name)
    record = classes.get(class_name)
    if record is None:
        return None, trail
    if record.expert_method is not None:
        return record, trail
    for base in record.bases:
        defining, resolved_trail = resolve_expert_method(base, classes, trail)
        if defining is not None:
            return defining, resolved_trail
    return None, trail


def skill_calls(method: ast.FunctionDef | ast.AsyncFunctionDef) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for node in ast.walk(method):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.ctx, ast.Load):
            continue
        if dotted_name(node.value) != "SkillLib":
            continue
        calls.append({"skill": node.attr, "line": node.lineno})
    calls.sort(key=lambda item: (item["line"], item["skill"]))
    return calls


def control_flow(method: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    kinds = {
        ast.If: "if",
        ast.For: "for",
        ast.While: "while",
        ast.Try: "try",
        ast.Match: "match",
    }
    found: list[str] = []
    for node in ast.walk(method):
        for cls, label in kinds.items():
            if isinstance(node, cls):
                found.append(f"{label}@L{getattr(node, 'lineno', '?')}")
                break
    return sorted(found)


def load_registry(path: Path) -> dict[str, Any]:
    namespace = runpy.run_path(str(path))
    return namespace["TASK_TEMPLATES"]


def retained_tasks(manifest_path: Path) -> tuple[dict[str, int], int]:
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, str) for row in rows):
        raise ValueError("manifest must be a JSON list of case path strings")
    counts: dict[str, int] = {}
    for row in rows:
        parts = row.replace("\\", "/").split("/")
        if len(parts) != 3:
            raise ValueError(f"unexpected manifest case id: {row}")
        counts[parts[1]] = counts.get(parts[1], 0) + 1
    return dict(sorted(counts.items())), len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--vlabench-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source_root = args.vlabench_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_path = repo_root / "procedural_interface" / "adapter" / "task_phase_templates.py"
    manifest_path = repo_root / "analysis_data" / "manifest_2133.json"
    registry = load_registry(registry_path)
    task_counts, case_count = retained_tasks(manifest_path)
    classes, task_to_class = collect_classes(source_root)

    rows: list[dict[str, Any]] = []
    critical: list[str] = []
    unresolved_branch_tasks: list[str] = []

    for task_name, applicable_count in task_counts.items():
        template = registry.get(task_name)
        registered_class = task_to_class.get(task_name)
        defining, inheritance = resolve_expert_method(registered_class, classes) if registered_class else (None, [])

        registry_skills = [phase.expected_skill for phase in template.phases] if template else []
        registry_targets = [phase.requires_target for phase in template.phases] if template else []
        contract_targets = [skill in TARGET_SKILLS for skill in registry_skills]
        source_calls = skill_calls(defining.expert_method) if defining and defining.expert_method else []
        source_skills = [item["skill"] for item in source_calls]
        branches = control_flow(defining.expert_method) if defining and defining.expert_method else []

        source_module = defining.module_path if defining else ""
        registered_module = classes[registered_class].module_path if registered_class in classes else ""
        registered_line = classes[registered_class].line if registered_class in classes else None
        registry_module = template.official_module.replace("\\", "/") if template else ""
        source_line = defining.expert_method.lineno if defining and defining.expert_method else None
        registered_permalink = (
            f"https://github.com/OpenMOSS/VLABench/blob/{args.source_commit}/"
            f"VLABench/tasks/hierarchical_tasks/{registered_module}#L{registered_line}"
            if registered_module and registered_line
            else ""
        )
        permalink = (
            f"https://github.com/OpenMOSS/VLABench/blob/{args.source_commit}/"
            f"VLABench/tasks/hierarchical_tasks/{source_module}#L{source_line}"
            if source_module and source_line
            else ""
        )

        checks = {
            "registry_present": template is not None,
            "source_registration_present": registered_class is not None,
            "official_class_matches_registration": bool(template and registered_class == template.official_class),
            "official_module_matches_registration": bool(template and registered_module == registry_module),
            "effective_method_resolved": defining is not None,
            "phase_order_matches": source_skills == registry_skills,
            "target_flag_contract_rule_matches": registry_targets == contract_targets,
        }
        failed = [name for name, passed in checks.items() if not passed]
        structural_status = "PASS" if not failed else "FAIL"
        if failed:
            critical.append(f"{task_name}: {', '.join(failed)}")
        if branches:
            unresolved_branch_tasks.append(task_name)

        rows.append(
            {
                "task_name": task_name,
                "applicable_case_count": applicable_count,
                "case_assignment_method": "UNRESOLVED" if branches else "not_applicable_single_sequence",
                "registered_class": registered_class or "",
                "registry_official_class": template.official_class if template else "",
                "registered_module": registered_module,
                "registered_class_line": registered_line or "",
                "registered_source_permalink": registered_permalink,
                "registered_source_file_sha256": sha256(classes[registered_class].source_path) if registered_class in classes else "",
                "registry_official_module": registry_module,
                "effective_defining_class": defining.name if defining else "",
                "effective_method_source": source_module,
                "effective_method_line": source_line or "",
                "source_permalink": permalink,
                "inheritance_path": " -> ".join(inheritance),
                "source_call_order": " -> ".join(source_skills),
                "registry_sequence": " -> ".join(registry_skills),
                "registry_target_taking_phases": ";".join(
                    f"{index}:{skill}" for index, (skill, takes_target) in enumerate(zip(registry_skills, registry_targets), 1) if takes_target
                ),
                "source_control_flow": ";".join(branches),
                "source_file_sha256": sha256(defining.source_path) if defining else "",
                "structural_check": structural_status,
                "failed_checks": ";".join(failed),
                "semantic_review": "PENDING_AUTHOR_LED_REVIEW",
                "mapping_rationale": "",
                "discrepancy_class": "UNRESOLVED",
                "resolution": "",
                "impact": "",
            }
        )

    fieldnames = list(rows[0].keys())
    csv_path = output_dir / "source_registry_crosswalk_draft.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "source_registry_crosswalk_draft.json"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "audit_status": "STRUCTURAL_FAIL" if critical else ("BRANCH_REVIEW_REQUIRED" if unresolved_branch_tasks else "STRUCTURAL_PASS_SEMANTIC_REVIEW_PENDING"),
        "artifact_baseline_commit": "31f4c8ce50943ac8282b4bab0d567cc5a597d730",
        "source_commit": args.source_commit,
        "manifest_case_count": case_count,
        "retained_task_count": len(task_counts),
        "registry_sha256": sha256(registry_path),
        "manifest_sha256": sha256(manifest_path),
        "source_traceable_task_count": sum(1 for row in rows if row["effective_method_source"]),
        "structural_pass_task_count": sum(1 for row in rows if row["structural_check"] == "PASS"),
        "critical_structural_findings": critical,
        "tasks_requiring_branch_review": unresolved_branch_tasks,
        "semantic_review_status": "PENDING_AUTHOR_LED_REVIEW",
        "labels_modified": 0,
    }
    (output_dir / "structural_audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if critical else 0


if __name__ == "__main__":
    raise SystemExit(main())
