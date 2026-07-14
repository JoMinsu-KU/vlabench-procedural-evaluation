"""Canonical the procedural evaluation project status labels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Status:
    """Structured failure/success code used in audit rows."""

    code: str
    label: str


TASK_SUCCESS = Status("C0", "Task Success")
EXECUTABLE_CONDITION_FAILURE = Status("C1", "Executable Condition Failure")
PARSER_SCHEMA_FAILURE = Status("C2", "Parser or Schema Conversion Failure")
TARGET_BINDING_FAILURE = Status("C3", "Target or Entity Binding Failure")
UNSUPPORTED_SKILL_FAILURE = Status("C4", "Unsupported Skill Failure")
NATIVE_EXECUTION_EXCEPTION = Status("C5", "Native Execution Exception")
INFRASTRUCTURE_FAILURE = Status("INFRA", "Infrastructure Failure")


def status_record(status: Status, subtype: str = "", detail: str = "") -> dict[str, str]:
    """Create a compact status dictionary."""

    return {"code": status.code, "label": status.label, "subtype": subtype, "detail": detail}
