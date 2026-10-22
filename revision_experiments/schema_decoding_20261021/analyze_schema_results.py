"""Paired decoding-format sensitivity analysis of complete, frozen native results."""

from collections import Counter, defaultdict
import csv
import math
from pathlib import Path

import numpy as np
import schema_decoding as sd

HERE = Path(__file__).resolve().parent
MODELS = ("qwen", "gemma", "internvl")
METRICS = ("raw_json_schema_valid", "strict_parse_ok", "model_parse_ok", "actual_gate1_ok",
           "actual_native_ok", "procedural_success")
REPETITIONS = 20000
SEED = 20261021


def mcnemar_exact(object_only, schema_only):
    discordant = object_only + schema_only
    if not discordant:
        return 1.0
    return min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(object_only, schema_only) + 1)) / 2**discordant)


def holm(values):
    result = [None] * len(values)
    previous = 0.0
    for index, original in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        previous = max(previous, min(1.0, values[original] * (len(values) - index)))
        result[original] = previous
    return result


def cluster_intervals(task_counts, task_sizes, samples):
    # All conditions/models use the same sampled task indices, preserving pairing.
    numerator = np.asarray(task_counts, dtype=float)[samples].sum(axis=1)
    denominator = np.asarray(task_sizes, dtype=float)[samples].sum(axis=1)
    rates = numerator / denominator[:, None]
    differences = rates[:, 1] - rates[:, 0]
    return {"object_ci": np.percentile(rates[:, 0], [2.5, 97.5]).tolist(),
            "schema_ci": np.percentile(rates[:, 1], [2.5, 97.5]).tolist(),
            "difference_ci": np.percentile(differences, [2.5, 97.5]).tolist()}


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_model(model, expected_ids):
    output = HERE / "evaluation_outputs" / model
    complete = sd.read_json(output / "evaluation_complete.json")
    if complete != {"rows": 644, "api_calls": 0, "native_max_attempts": 1}:
        raise RuntimeError(f"Incomplete or non-first-attempt evaluation: {model}")
    config, generations = sd.load_frozen_generation(HERE / "gcp_outputs" / model, HERE / "cloud_bundle")
    if set(config["case_ids"]) != set(expected_ids) or config["conditions"] != list(sd.CONDITIONS):
        raise RuntimeError("Conditions or selected population differ")
    rows = {}
    for source_path, source in generations:
        key = source["case_id"], source["condition"]
        path = output / source["condition"] / "case_outputs" / (sd.safe_case_id(source["case_id"]) + ".json")
        row = sd.read_json(path)
        if row["source_sha256"] != sd.digest(source_path) or row.get("api_called") is not False:
            raise RuntimeError("Evaluation source or offline-only guarantee mismatch")
        if row.get("native_attempt_count", 0) > 1 or row.get("actual_failure_code") == "INFRA":
            raise RuntimeError("Infrastructure failures/retries require adjudication, not model scoring")
        if key in rows:
            raise RuntimeError("Duplicate evaluation case")
        rows[key] = (row, source)
    if len(rows) != 644:
        raise RuntimeError("A model must have all 644 outcomes")
    return rows


def main():
    bundle = sd.load_bundle(HERE / "cloud_bundle")
    ids = [entry["case_id"] for entry in bundle["cases"]]
    tasks = sorted({case.rsplit("/", 1)[0] for case in ids})
    if len(ids) != 322 or len(tasks) != 34:
        raise RuntimeError("Unexpected fixed population")
    grouped = {task: [case for case in ids if case.rsplit("/", 1)[0] == task] for task in tasks}
    sizes = [len(grouped[task]) for task in tasks]
    samples = np.random.default_rng(SEED).integers(0, len(tasks), size=(REPETITIONS, len(tasks)))
    paired, taxonomy, tokens, by_task, primary = [], [], [], [], []
    input_hashes = {}
    for model in MODELS:
        rows = load_model(model, ids)
        input_hashes[model] = {"evaluation_complete_sha256": sd.digest(HERE / "evaluation_outputs" / model / "evaluation_complete.json"),
                               "generation_complete_sha256": sd.digest(HERE / "gcp_outputs" / model / "generation_complete.json")}
        for metric in METRICS:
            object_values = {case: rows[case, "json_object"][0].get(metric) is True for case in ids}
            schema_values = {case: rows[case, "json_schema"][0].get(metric) is True for case in ids}
            object_only = sum(object_values[case] and not schema_values[case] for case in ids)
            schema_only = sum(schema_values[case] and not object_values[case] for case in ids)
            task_counts = [[sum(object_values[case] for case in grouped[task]), sum(schema_values[case] for case in grouped[task])] for task in tasks]
            intervals = cluster_intervals(task_counts, sizes, samples)
            record = {"model": model, "metric": metric, "n": len(ids),
                      "json_object_n": sum(object_values.values()), "json_schema_n": sum(schema_values.values()),
                      "both": sum(object_values[case] and schema_values[case] for case in ids),
                      "object_only": object_only, "schema_only": schema_only,
                      "neither": sum(not object_values[case] and not schema_values[case] for case in ids),
                      "difference_pp_schema_minus_object": 100 * (schema_only - object_only) / len(ids),
                      "difference_ci_low_pp": 100 * intervals["difference_ci"][0], "difference_ci_high_pp": 100 * intervals["difference_ci"][1],
                      "object_cluster_ci_low_pct": 100 * intervals["object_ci"][0], "object_cluster_ci_high_pct": 100 * intervals["object_ci"][1],
                      "schema_cluster_ci_low_pct": 100 * intervals["schema_ci"][0], "schema_cluster_ci_high_pct": 100 * intervals["schema_ci"][1]}
            paired.append(record)
            if metric == "procedural_success":
                record["mcnemar_exact_auxiliary_p"] = mcnemar_exact(object_only, schema_only)
                primary.append(record)
        for condition in sd.CONDITIONS:
            selected = [rows[case, condition] for case in ids]
            failures = Counter(row.get("actual_failure_subtype") or row.get("actual_failure_code") or "success" for row, _ in selected)
            taxonomy.extend({"model": model, "condition": condition, "failure_subtype": key, "n": value, "denominator": 322}
                            for key, value in sorted(failures.items()))
            for case in ids:
                row, source = rows[case, condition]
                usage = source.get("usage") or {}
                tokens.append({"model": model, "condition": condition, "case_id": case,
                               "finish_reason": source.get("finish_reason"), "truncated": source.get("finish_reason") == "length",
                               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                               "failure_subtype": row.get("actual_failure_subtype"), "success": row.get("procedural_success")})
            for task in tasks:
                selected_task = [rows[case, condition][0] for case in grouped[task]]
                by_task.append({"model": model, "condition": condition, "task": task, "n": len(selected_task),
                                **{metric: sum(row.get(metric) is True for row in selected_task) for metric in METRICS}})
    adjusted = holm([row["mcnemar_exact_auxiliary_p"] for row in primary])
    for row, value in zip(primary, adjusted):
        row["mcnemar_holm_3_models_p"] = value
    destination = HERE / "analysis"
    destination.mkdir(exist_ok=True)
    write_csv(destination / "paired_metrics.csv", paired)
    write_csv(destination / "failure_taxonomy.csv", taxonomy)
    write_csv(destination / "token_and_truncation_audit.csv", tokens)
    write_csv(destination / "task_metrics.csv", by_task)
    sd.write_json(destination / "analysis_manifest.json", {"cases": 322, "tasks": 34, "models": list(MODELS), "rows": 1932,
                  "conditions": list(sd.CONDITIONS), "native_attempts": 1, "bootstrap_repetitions": REPETITIONS,
                  "bootstrap_seed": SEED, "primary_uncertainty": "paired task-cluster percentile bootstrap",
                  "auxiliary_test": "case-level exact McNemar; Holm family = three models for final success",
                  "input_hashes": input_hashes, "analysis_source_sha256": sd.digest(__file__)})
    lines = ["# JSON-format sensitivity experiment", "", "Fixed 322 cases, 34 task clusters, three BF16 backbones. Both arms are fresh paired outputs.",
             "Native success is first-attempt condition success; rejected plans remain in the denominator.", "",
             "| Model | JSON object success | JSON Schema success | Difference (pp) | Task-clustered 95% CI (pp) |",
             "|---|---:|---:|---:|---:|"]
    for row in primary:
        lines.append(f"| {row['model']} | {row['json_object_n']}/322 | {row['json_schema_n']}/322 | {row['difference_pp_schema_minus_object']:.2f} | [{row['difference_ci_low_pp']:.2f}, {row['difference_ci_high_pp']:.2f}] |")
    lines += ["", "## Interpretation Boundaries", "",
              "This experiment isolates response-format constraints within one fixed new serving environment. It does not replace the original 2,133-case A/B study.",
              "Schema validity, procedural gate validity, and native goal success are separate endpoints. A schema-valid plan need not select the correct phases or targets.",
              "Task-clustered confidence intervals are primary. Case-level McNemar results are auxiliary and do not remove task dependence.",
              "No conclusion of improved planning competence follows from formatting improvements alone. Inspect phase/target failures and condition success before revising claims.",
              "", "The category/task table and full token audit are descriptive diagnostics. Follow-up analyses must retain these fixed results and their provenance.", ""]
    (destination / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Analysis complete: 1932 verified outcomes", flush=True)


if __name__ == "__main__":
    main()
