#!/usr/bin/env python3
"""Macro-family exclusion and projectable-only sensitivities for every symbolic endpoint tier.

Reads reproducibility_core/cross_scoring/full_to_symbolic_projection_rows.csv (artifact
v1.7 columns) and repeats the v1.4 whole-family exclusion and the v1.5 projectable-only
conditioning under the released PM=100 rule, strict sequence equality, and strict
equality after executor-phase filtering.  No model inference, native rollout, label
modification, or projection-rule change is involved.

Outputs:
  reproducibility_core/cross_scoring/endpoint_tier_sensitivities.csv
  reproducibility_core/cross_scoring/endpoint_tier_sensitivities_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

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
ENDPOINTS = {
    "pm100": ("original_pm100_success", "projected_pm100_success"),
    "strict": ("original_strict_sequence_success", "projected_strict_sequence_success"),
    "strict_executor_filtered": (
        "original_strict_sequence_success",
        "projected_strict_executor_filtered_success",
    ),
}


def as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def pct(numerator: int, denominator: int) -> str:
    return f"{100.0 * numerator / denominator:.2f}" if denominator else "NA"


def summarize(rows: list[dict[str, str]], analysis_set: str, model: str, endpoint: str) -> dict[str, Any]:
    original_key, projected_key = ENDPOINTS[endpoint]
    n = len(rows)
    projectable = [row for row in rows if as_bool(row["projection_supported"])]
    original = sum(as_bool(row[original_key]) for row in rows)
    projected = sum(as_bool(row[projected_key]) for row in rows)
    full = sum(as_bool(row["full_procedure_success"]) for row in rows)
    both = sum(as_bool(row[original_key]) and as_bool(row[projected_key]) for row in rows)
    downstream = sum(as_bool(row[projected_key]) and not as_bool(row["full_procedure_success"]) for row in rows)
    original_projectable = sum(as_bool(row[original_key]) for row in projectable)
    both_projectable = sum(as_bool(row[original_key]) and as_bool(row[projected_key]) for row in projectable)
    union = original + projected - both
    union_projectable = original_projectable + projected - both_projectable
    return {
        "analysis_set": analysis_set,
        "symbolic_endpoint": endpoint,
        "model": model,
        "n": n,
        "projectable_n": len(projectable),
        "original_symbolic_success_n": original,
        "projected_symbolic_success_n": projected,
        "projected_success_full_failure_n": downstream,
        "full_procedure_success_n": full,
        "original_projected_both_n": both,
        "original_only_n": original - both,
        "projected_only_n": projected - both,
        "original_projected_jaccard_pct": pct(both, union),
        "projectable_subset_original_success_n": original_projectable,
        "projectable_subset_both_n": both_projectable,
        "projectable_subset_jaccard_pct": pct(both_projectable, union_projectable),
        "original_success_unprojectable_n": original - original_projectable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    output_dir = root / "reproducibility_core" / "cross_scoring"
    with (output_dir / "full_to_symbolic_projection_rows.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 6_399:
        raise ValueError(f"Expected 6,399 frozen rows, found {len(rows)}")
    missing = [key for keys in ENDPOINTS.values() for key in keys if key not in rows[0]]
    if missing:
        raise ValueError(f"row table lacks v1.7 endpoint columns: {sorted(set(missing))}")

    models = sorted({row["model"] for row in rows})
    output: list[dict[str, Any]] = []
    for analysis_set, selected in (
        ("full_retained_manifest", rows),
        ("macro_family_excluded", [row for row in rows if row["task_name"] not in MACRO_TASKS]),
    ):
        for endpoint in ENDPOINTS:
            for model in models:
                output.append(summarize([row for row in selected if row["model"] == model], analysis_set, model, endpoint))
            output.append(summarize(selected, analysis_set, "Pooled", endpoint))

    with (output_dir / "endpoint_tier_sensitivities.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)

    pooled = {
        (row["analysis_set"], row["symbolic_endpoint"]): row
        for row in output
        if row["model"] == "Pooled"
    }
    summary = {
        "status": "passed",
        "analysis_version": "endpoint_tier_sensitivities_v1",
        "input": "reproducibility_core/cross_scoring/full_to_symbolic_projection_rows.csv",
        "pooled": {
            f"{analysis_set}/{endpoint}": {
                key: row[key]
                for key in (
                    "n",
                    "projectable_n",
                    "original_symbolic_success_n",
                    "projected_symbolic_success_n",
                    "projected_success_full_failure_n",
                    "original_projected_jaccard_pct",
                    "projectable_subset_jaccard_pct",
                )
            }
            for (analysis_set, endpoint), row in sorted(pooled.items())
        },
        "qualitative_findings_preserved": all(
            int(row["original_only_n"]) > 0
            and int(row["projected_only_n"]) > 0
            and int(row["projected_success_full_failure_n"]) > 0
            for row in pooled.values()
        ),
        "model_inference_calls": 0,
        "native_rollouts": 0,
        "success_labels_modified": 0,
        "projection_rules_modified": False,
    }
    (output_dir / "endpoint_tier_sensitivities_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
