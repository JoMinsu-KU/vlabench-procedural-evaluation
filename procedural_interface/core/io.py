"""Small filesystem helpers for structured legacy artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    """Create a directory and return it as a resolved Path-like object."""

    # Normalize caller input into a Path so all downstream writes are uniform.
    directory = Path(path)
    # Create parent directories explicitly because audit roots are often new.
    directory.mkdir(parents=True, exist_ok=True)
    # Return the directory path for fluent call sites.
    return directory


def read_json(path: str | Path) -> Any:
    """Read UTF-8 JSON without adding fallback behavior."""

    # Keep JSON loading strict so malformed artifacts fail early.
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> Path:
    """Write UTF-8 JSON with stable indentation."""

    # Normalize output path once so parent creation and writing use same target.
    out_path = Path(path)
    # Ensure the artifact directory exists before writing.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Use ensure_ascii=False so Korean task notes and audit messages stay readable.
    out_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    # Return the written path for report metadata.
    return out_path


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> Path:
    """Write a deterministic CSV file from dictionaries."""

    # Normalize output path so caller can pass strings or Path objects.
    out_path = Path(path)
    # Ensure the destination directory exists.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Materialize rows once because callers may pass generators.
    materialized_rows = list(rows)
    # Use utf-8-sig so Excel on Windows opens Korean headers cleanly.
    with out_path.open("w", newline="", encoding="utf-8-sig") as handle:
        # Create a stable writer with explicit columns.
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        # Write headers first for audit readability.
        writer.writeheader()
        # Write each row with missing keys rendered as empty strings by DictWriter.
        writer.writerows(materialized_rows)
    # Return the written path for summary JSON.
    return out_path
