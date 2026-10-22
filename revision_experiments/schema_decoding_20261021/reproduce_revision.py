"""Reproduce paired decoding statistics from immutable, linked analysis rows."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_schema_results import MODELS, METRICS, REPETITIONS, SEED, cluster_intervals, holm, mcnemar_exact

HERE = Path(__file__).resolve().parent


def reproduce(root):
    frozen = (root / "frozen_analysis_rows.jsonl").read_bytes()
    provenance = json.loads((root / "EXPORT_PROVENANCE.json").read_text())
    if hashlib.sha256(frozen).hexdigest() != provenance["derived_file_sha256"]:
        raise ValueError("Frozen analysis-row checksum mismatch")
    rows = [json.loads(line) for line in frozen.decode("utf-8").splitlines()]
    keyed = {(r["model"], r["condition"], r["case_id"]): r for r in rows}
    if len(rows) != 1932 or len(keyed) != 1932:
        raise ValueError("Expected 1,932 unique frozen outcomes")
    ids = [entry["case_id"] for entry in json.loads((root / "cloud_bundle/bundle_manifest.json").read_text())["cases"]]
    tasks = sorted({case.rsplit("/", 1)[0] for case in ids})
    if len(ids) != 322 or len(set(ids)) != 322 or len(tasks) != 34:
        raise ValueError("Unexpected analysis population")
    expected = {(model, condition, case) for model in MODELS for condition in ("json_object", "json_schema") for case in ids}
    if set(keyed) != expected or any(r["native_attempt_count"] > 1 or r["failure_code"] == "INFRA" for r in rows):
        raise ValueError("Population, attempt policy, or infrastructure outcome mismatch")
    for row in rows:
        key = row["case_id"].replace("/", "__")
        evaluation = root / "evaluation_outputs" / row["model"] / row["condition"] / "case_outputs" / (key + ".json")
        generation = root / "gcp_outputs" / row["model"] / row["condition"] / "case_outputs" / key / "row.json"
        ev_bytes, gen_bytes = evaluation.read_bytes(), generation.read_bytes()
        if hashlib.sha256(ev_bytes).hexdigest() != row["evaluation_sha256"] or hashlib.sha256(gen_bytes).hexdigest() != row["generation_sha256"]:
            raise ValueError("Frozen source checksum mismatch")
        ev = json.loads(ev_bytes)
        if ev["source_sha256"] != row["generation_sha256"] or ev.get("api_called") is not False:
            raise ValueError("Offline evaluation linkage mismatch")
        if row["metrics"] != {metric: ev.get(metric) is True for metric in METRICS}:
            raise ValueError("Compact analysis row differs from its original evaluation")
    grouped = defaultdict(list)
    for case in ids:
        grouped[case.rsplit("/", 1)[0]].append(case)
    sizes = [len(grouped[task]) for task in tasks]
    samples = np.random.default_rng(SEED).integers(0, len(tasks), size=(REPETITIONS, len(tasks)))
    result, primary = [], []
    for model in MODELS:
        for metric in METRICS:
            a = {case: keyed[model, "json_object", case]["metrics"][metric] for case in ids}
            b = {case: keyed[model, "json_schema", case]["metrics"][metric] for case in ids}
            counts = [[sum(a[c] for c in grouped[t]), sum(b[c] for c in grouped[t])] for t in tasks]
            ci = cluster_intervals(counts, sizes, samples)
            only_a = sum(a[c] and not b[c] for c in ids)
            only_b = sum(b[c] and not a[c] for c in ids)
            row = {"model": model, "metric": metric, "n": 322,
                   "json_object_n": sum(a.values()), "json_schema_n": sum(b.values()),
                   "both": sum(a[c] and b[c] for c in ids), "object_only": only_a, "schema_only": only_b,
                   "neither": sum(not a[c] and not b[c] for c in ids),
                   "difference_pp_schema_minus_object": 100 * (only_b - only_a) / 322,
                   "difference_ci_low_pp": 100 * ci["difference_ci"][0],
                   "difference_ci_high_pp": 100 * ci["difference_ci"][1],
                   "object_cluster_ci_low_pct": 100 * ci["object_ci"][0],
                   "object_cluster_ci_high_pct": 100 * ci["object_ci"][1],
                   "schema_cluster_ci_low_pct": 100 * ci["schema_ci"][0],
                   "schema_cluster_ci_high_pct": 100 * ci["schema_ci"][1]}
            if metric == "procedural_success":
                row["mcnemar_exact_auxiliary_p"] = mcnemar_exact(only_a, only_b)
                primary.append(row)
            result.append(row)
    for row, value in zip(primary, holm([r["mcnemar_exact_auxiliary_p"] for r in primary])):
        row["mcnemar_holm_3_models_p"] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=HERE / "reproduced/paired_metrics.csv")
    args = parser.parse_args()
    results = reproduce(args.root)
    expected = list(csv.DictReader((args.root / "analysis/paired_metrics.csv").open(encoding="utf-8-sig")))
    if len(results) != len(expected):
        raise ValueError("Metric count mismatch")
    for row, old in zip(results, expected):
        for field, value in row.items():
            if isinstance(value, str):
                assert value == old[field], (field, value, old[field])
            else:
                assert abs(float(value) - float(old[field])) < 1e-10, (field, value, old[field])
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite an existing reproduction; choose a new --output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in results for k in row))
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print("Verified all 18 paired metric rows, intervals, and three Holm-adjusted success tests.")


if __name__ == "__main__":
    main()
