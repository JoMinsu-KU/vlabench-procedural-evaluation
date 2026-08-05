from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MACRO_TASKS = {
    "insert_bloom_flower",
    "insert_flower",
    "insert_flower_common_sense",
    "insert_flower_semantic",
    "add_condiment",
    "add_condiment_common_sense",
    "add_condiment_semantic",
    "add_condiment_spatial",
}


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def pct(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.2f}" if denominator else "NA"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: list[dict[str, str]], analysis_set: str, model: str) -> dict[str, object]:
    projectable = [row for row in rows if as_bool(row["projection_supported"])]
    original_success = [row for row in projectable if as_bool(row["original_symbolic_success"])]
    projected_success = [
        row for row in projectable if as_bool(row["projected_full_symbolic_success"])
    ]
    both = [
        row
        for row in projectable
        if as_bool(row["original_symbolic_success"])
        and as_bool(row["projected_full_symbolic_success"])
    ]
    original_only_n = len(original_success) - len(both)
    projected_only_n = len(projected_success) - len(both)
    union_n = len(both) + original_only_n + projected_only_n
    unprojectable_original_success_n = sum(
        not as_bool(row["projection_supported"])
        and as_bool(row["original_symbolic_success"])
        for row in rows
    )
    return {
        "analysis_set": analysis_set,
        "model": model,
        "all_rows_n": len(rows),
        "projectable_rows_n": len(projectable),
        "original_success_projectable_n": len(original_success),
        "projected_success_n": len(projected_success),
        "both_success_n": len(both),
        "original_only_n": original_only_n,
        "projected_only_n": projected_only_n,
        "success_union_n": union_n,
        "projectable_subset_jaccard_pct": pct(len(both), union_n),
        "original_success_unprojectable_n": unprojectable_original_success_n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute original/projected success-set Jaccard within projectable frozen outputs."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to reproducibility_core/cross_scoring under --repo-root.",
    )
    args = parser.parse_args()

    root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else root / "reproducibility_core" / "cross_scoring"
    )
    input_path = output_dir / "full_to_symbolic_projection_rows.csv"
    rows = read_rows(input_path)
    if len(rows) != 6_399:
        raise ValueError(f"Expected 6,399 frozen rows, found {len(rows)}")

    macro_excluded = [row for row in rows if row["task_name"] not in MACRO_TASKS]
    analysis_sets = (
        ("full_retained_manifest", rows),
        ("macro_family_excluded", macro_excluded),
    )
    models = sorted({row["model"] for row in rows})
    output_rows: list[dict[str, object]] = []
    for analysis_set, selected in analysis_sets:
        for model in models:
            output_rows.append(
                summarize(
                    [row for row in selected if row["model"] == model],
                    analysis_set,
                    model,
                )
            )
        output_rows.append(summarize(selected, analysis_set, "Pooled"))

    fieldnames = list(output_rows[0].keys())
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "projectable_subset_jaccard.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    full_pooled = next(
        row
        for row in output_rows
        if row["analysis_set"] == "full_retained_manifest" and row["model"] == "Pooled"
    )
    excluded_pooled = next(
        row
        for row in output_rows
        if row["analysis_set"] == "macro_family_excluded" and row["model"] == "Pooled"
    )
    summary = {
        "status": "passed",
        "analysis_version": "projectable_subset_jaccard_v1",
        "input": "reproducibility_core/cross_scoring/full_to_symbolic_projection_rows.csv",
        "conditioning_rule": "projection_supported == True",
        "full_manifest_pooled": full_pooled,
        "macro_family_excluded_pooled": excluded_pooled,
        "qualitative_findings_preserved": (
            int(full_pooled["original_only_n"]) > 0
            and int(full_pooled["projected_only_n"]) > 0
            and float(full_pooled["projectable_subset_jaccard_pct"]) < 50.0
        ),
        "model_inference_calls": 0,
        "native_rollouts": 0,
        "success_labels_modified": 0,
        "interpretation": (
            "This is a projectability-conditioned sensitivity analysis, not a replacement "
            "for the full-pipeline Jaccard estimand."
        ),
    }
    (output_dir / "projectable_subset_jaccard_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
