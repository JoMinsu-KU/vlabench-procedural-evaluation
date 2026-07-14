from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


REQUIRED_FILES = ("input.png", "input_mask.png", "instruction.txt", "env_config.json")
FORBIDDEN_FILES = (
    "operation_sequence.json",
    "parsed_output.json",
    "raw_output.json",
    "validation.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the model-visible 2,133-case public view from a VLABench evaluation dataset."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    case_ids = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(case_ids, list) or len(case_ids) != 2_133:
        raise ValueError("manifest must contain exactly 2,133 case identifiers")

    copied = 0
    missing: list[dict[str, str]] = []
    for case_id in case_ids:
        relative = Path(*str(case_id).split("/"))
        source_case = args.source_root / relative
        output_case = args.output_root / relative
        output_case.mkdir(parents=True, exist_ok=True)
        for filename in REQUIRED_FILES:
            source = source_case / filename
            if not source.exists():
                missing.append({"case_id": str(case_id), "file": filename})
                continue
            shutil.copy2(source, output_case / filename)
        for forbidden in FORBIDDEN_FILES:
            if (output_case / forbidden).exists():
                raise RuntimeError(f"forbidden artifact copied: {output_case / forbidden}")
        copied += 1

    report = {
        "requested_cases": len(case_ids),
        "created_case_directories": copied,
        "required_files": list(REQUIRED_FILES),
        "missing_count": len(missing),
        "missing": missing,
        "status": "passed" if not missing else "failed",
    }
    report_path = args.output_root / "public_view_build_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
