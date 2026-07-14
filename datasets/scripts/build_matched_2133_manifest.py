"""Build the paired 2133-case manifest for symbolic and procedural evaluation.

The main experiment compares VLABench-style symbolic evaluation and the
Procedural Execution Interface on exactly the same examples. This script copies
the verified subset into a matched manifest and checks that every public-view
case has the required input files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_PUBLIC_FILES = ["input.png", "input_mask.png", "instruction.txt", "env_config.json"]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def sort_case_id(case_id: str) -> tuple[str, str, int]:
    category, task_name, example_name = case_id.split("/")
    return category, task_name, int(example_name.replace("example", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "manifests" / "verified_subset_2133_cases.json",
    )
    parser.add_argument(
        "--public-root",
        type=Path,
        default=Path(r"${LEGACY_ARTIFACT_ROOT}\data\public_view\vlabench_vlm4500_public"),
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "manifests" / "matched_2133_symbolic_procedural_cases.json",
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dataset_audits" / "matched_2133_manifest_audit.json",
    )
    args = parser.parse_args()

    cases = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError(f"source manifest must contain a list: {args.source_manifest}")

    sorted_cases = sorted([str(case_id) for case_id in cases], key=sort_case_id)
    missing: list[dict[str, str]] = []
    for case_id in sorted_cases:
        case_path = args.public_root / Path(*case_id.split("/"))
        for filename in REQUIRED_PUBLIC_FILES:
            candidate = case_path / filename
            if not candidate.exists():
                missing.append({"case_id": case_id, "missing_file": str(candidate)})

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(sorted_cases, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "source_manifest": str(args.source_manifest),
        "public_root": str(args.public_root),
        "output_manifest": str(args.output_manifest),
        "total_cases": len(sorted_cases),
        "required_public_files": REQUIRED_PUBLIC_FILES,
        "missing_count": len(missing),
        "missing": missing,
        "status": "passed" if len(sorted_cases) == 2133 and not missing else "failed",
    }
    args.audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

