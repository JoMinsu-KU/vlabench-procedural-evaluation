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
}
TEXT_EXTENSIONS = {".py", ".json", ".jsonl", ".csv", ".md", ".txt", ".yml", ".yaml", ".toml", ".cff"}
PINNED_DEPENDENCY = re.compile(
    r"^[A-Za-z0-9_.-]+(?:\[[^\]]+\])?==[0-9]+(?:\.[0-9]+){1,4}(?:[+.-][A-Za-z0-9.]+)?$"
)


def count_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


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

    for model in MODELS:
        a_csv = root / "results" / "model_outputs" / model / "experiment_a" / "official_vlm_symbolic_scores.csv"
        b_csv = root / "results" / "model_outputs" / model / "experiment_b" / "procedural_vlm_execution_results.csv"
        if count_csv(a_csv) != 2_133:
            errors.append(f"{model}: Experiment A does not contain 2,133 rows")
        if count_csv(b_csv) != 2_133:
            errors.append(f"{model}: Experiment B does not contain 2,133 rows")

    findings: list[dict[str, object]] = []
    for path in root.rglob("*"):
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
        if p.is_file() and p.name not in {"SHA256SUMS.txt", "PACKAGE_VERIFICATION.json"}
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
        "checksummed_files": len(checksum_rows),
    }
    (root / "PACKAGE_VERIFICATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
