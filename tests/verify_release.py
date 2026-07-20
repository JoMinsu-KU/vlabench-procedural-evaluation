from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


MODELS = ("qwen", "gemma", "internvl")
FORBIDDEN_PATTERNS = {
    "external_ipv4": re.compile(r"\b(?!127\.0\.0\.1\b)(?:\d{1,3}\.){3}\d{1,3}\b"),
    "local_windows_user_path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+", re.IGNORECASE),
    "local_linux_user_path": re.compile(r"/home/(?!conda/)[^/\s]+", re.IGNORECASE),
    "local_wsl_mount_path": re.compile(r"/mnt/[a-z]/", re.IGNORECASE),
    "secret_assignment": re.compile(r"(?:api[_-]?key|authorization|bearer)\s*[:=]\s*[^\s\"']+", re.IGNORECASE),
    "excluded_legacy_study_term": re.compile(
        r"\bSILVER\b|\bTrack\s+[A-D]\b|replanning|metadata[-_ ]control",
        re.IGNORECASE,
    ),
}
TEXT_EXTENSIONS = {".py", ".json", ".jsonl", ".csv", ".md", ".txt", ".yml", ".yaml", ".toml", ".cff"}
PINNED_DEPENDENCY = re.compile(
    r"^[A-Za-z0-9_.-]+(?:\[[^\]]+\])?==[0-9]+(?:\.[0-9]+){1,4}(?:[+.-][A-Za-z0-9.]+)?$"
)


def count_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def count_jsonl(path: Path) -> tuple[int, int]:
    rows = 0
    case_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            case_ids.add(str(json.loads(line)["case_id"]))
    return rows, len(case_ids)


def is_pinned_dependency_version(text: str, match: re.Match[str]) -> bool:
    line_start = text.rfind("\n", 0, match.start()) + 1
    line_end = text.find("\n", match.end())
    if line_end == -1:
        line_end = len(text)
    return PINNED_DEPENDENCY.fullmatch(text[line_start:line_end].strip()) is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repository_root", type=Path)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    errors: list[str] = []

    manifest = json.loads((root / "datasets" / "manifests" / "matched_2133_symbolic_procedural_cases.json").read_text(encoding="utf-8"))
    if len(manifest) != 2_133 or len(set(manifest)) != 2_133:
        errors.append("paired manifest is not 2,133 unique cases")
    rebuilt_manifest_path = root / "datasets" / "manifests" / "rebuilt_verified_subset_2133_cases.json"
    if rebuilt_manifest_path.exists():
        rebuilt_manifest = json.loads(rebuilt_manifest_path.read_text(encoding="utf-8"))
        if rebuilt_manifest != manifest:
            errors.append("manifest rebuilt from the two-replay audit does not match the released manifest")

    oracle_expected = {
        "qwen": {"phase_oracle": 427, "target_oracle": 488},
        "gemma": {"phase_oracle": 344, "target_oracle": 585},
        "internvl": {"phase_oracle": 314, "target_oracle": 191},
    }
    for model in MODELS:
        a_csv = root / "results" / "model_outputs" / model / "experiment_a" / "official_vlm_symbolic_scores.csv"
        b_csv = root / "results" / "model_outputs" / model / "experiment_b" / "procedural_vlm_execution_results.csv"
        if count_csv(a_csv) != 2_133:
            errors.append(f"{model}: Experiment A does not contain 2,133 rows")
        if count_csv(b_csv) != 2_133:
            errors.append(f"{model}: Experiment B does not contain 2,133 rows")
        for condition in ("phase_oracle", "target_oracle"):
            condition_root = root / "results" / "model_outputs" / model / "oracle_ablations" / condition
            result_csv = condition_root / "procedural_vlm_execution_results.csv"
            rows = read_csv(result_csv)
            if len(rows) != 2_133:
                errors.append(f"{model}/{condition}: result CSV does not contain 2,133 rows")
            elif [row["case_id"] for row in rows] != manifest:
                errors.append(f"{model}/{condition}: result CSV does not match manifest order")
            success_count = sum(
                row["procedural_success"].lower() == "true" and row["native_selected_attempt"] == "1"
                for row in rows
            )
            if success_count != oracle_expected[model][condition]:
                errors.append(
                    f"{model}/{condition}: expected {oracle_expected[model][condition]} successes, found {success_count}"
                )
            summary = json.loads((condition_root / "summary.json").read_text(encoding="utf-8"))
            if int(summary["procedural_success"]) != success_count:
                errors.append(f"{model}/{condition}: summary success count does not match CSV")
            jsonl_rows, jsonl_case_ids = count_jsonl(condition_root / "procedural_vlm_execution_rows.jsonl")
            if jsonl_rows != 2_133 or jsonl_case_ids != 2_133:
                errors.append(
                    f"{model}/{condition}: canonical JSONL is not 2,133 rows and case IDs"
                )

    oracle_main = read_csv(root / "results" / "reported_statistics" / "oracle_main_results_clustered.csv")
    if len(oracle_main) != 3:
        errors.append("oracle main table does not contain three model rows")
    else:
        for row in oracle_main:
            if float(row["phase_minus_full_ci_low"]) <= 0 or float(row["target_minus_full_ci_low"]) <= 0:
                errors.append(f"{row['model']}: an oracle-vs-full clustered interval includes zero")
            if not (
                float(row["target_minus_phase_ci_low"]) <= 0
                <= float(row["target_minus_phase_ci_high"])
            ):
                errors.append(f"{row['model']}: target-vs-phase clustered interval excludes zero")

    adjudication_rows = count_csv(
        root / "results" / "outcome_audits" / "oracle_attribute_error_adjudication.csv"
    )
    if adjudication_rows != 472:
        errors.append(f"oracle AttributeError adjudication contains {adjudication_rows}, not 472 rows")

    schema = json.loads(
        (root / "shared" / "schemas" / "procedural_plan_schema.json").read_text(encoding="utf-8")
    )
    alternatives = schema["properties"]["steps"]["items"].get("oneOf", [])
    if len(alternatives) != 2:
        errors.append("procedural schema does not separate target-taking and no-target skills")

    findings: list[dict[str, object]] = []
    for path in root.rglob("*"):
        if ".git" in path.parts or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if path.relative_to(root).as_posix() in {
            "tests/verify_release.py",
            "PACKAGE_VERIFICATION.json",
            "SHA256SUMS.txt",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            continue
        for name, pattern in FORBIDDEN_PATTERNS.items():
            for match in pattern.finditer(text):
                if name == "external_ipv4" and is_pinned_dependency_version(text, match):
                    continue
                findings.append({"type": name, "path": path.relative_to(root).as_posix(), "match": match.group(0)[:120]})
    if findings:
        errors.append(f"anonymity scan found {len(findings)} forbidden strings")

    checksum_rows = []
    for path in sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
        and p.name not in {"SHA256SUMS.txt", "PACKAGE_VERIFICATION.json"}
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_rows.append(f"{digest}  {path.relative_to(root).as_posix()}")
    (root / "SHA256SUMS.txt").write_text("\n".join(checksum_rows) + "\n", encoding="utf-8")

    report = {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "anonymity_findings": findings,
        "manifest_cases": len(manifest),
        "model_result_pairs": 3,
        "oracle_model_condition_runs": 6,
        "oracle_adjudication_rows": adjudication_rows,
        "checksummed_files": len(checksum_rows),
    }
    (root / "PACKAGE_VERIFICATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
