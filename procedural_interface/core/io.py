"""Small filesystem helpers for structured the procedural evaluation project artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a resolved Path-like object."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_json(path: str | Path) -> Any:
    """Read UTF-8 JSON without adding fallback behavior."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> Path:
    """Write UTF-8 JSON with stable indentation."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> Path:
    """Write a deterministic CSV file from dictionaries."""

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    materialized_rows = list(rows)
    with out_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized_rows)
    return out_path
