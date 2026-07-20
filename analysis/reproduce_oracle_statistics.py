from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis_data"
TABLE_DIR = ROOT / "analysis" / "tables"
RESULT_TABLE_DIR = ROOT / "results" / "reported_statistics"
AUDIT_DIR = ROOT / "results" / "outcome_audits"

MODELS = {
    "qwen": "Qwen3-VL-8B",
    "gemma": "Gemma-3-12B-it",
    "internvl": "InternVL3-8B",
}
STRUCTURAL_FAILURES = {"extra_phase", "missing_phase", "phase_skill_mismatch"}
FAILURE_CLASSES = (
    "output_schema_failure",
    "procedural_structure_failure",
    "target_context_failure",
    "native_execution_failure",
    "condition_failure",
    "success",
)
ATTRIBUTE_ERROR_DETAILS = {
    "'BilliardBall' object has no attribute 'get_place_point'",
    "'Fruit' object has no attribute 'get_place_point'",
    "'Toy' object has no attribute 'get_place_point'",
    "'BilliardTable' object has no attribute 'get_grasped_keypoints'",
}
BOOTSTRAP_SEED = 20260711
BOOTSTRAP_REPLICATES = 20_000


def first_attempt_success(frame: pd.DataFrame) -> pd.Series:
    return frame["procedural_success"].astype(bool) & frame["native_selected_attempt"].eq(1)


def result_source(model_key: str, condition: str) -> str:
    if condition == "full":
        return f"results/model_outputs/{model_key}/experiment_b/procedural_vlm_execution_results.csv"
    return (
        f"results/model_outputs/{model_key}/oracle_ablations/{condition}/"
        "procedural_vlm_execution_results.csv"
    )


def load_condition(model_key: str, condition: str) -> pd.DataFrame:
    if condition == "full":
        return pd.read_csv(DATA_DIR / f"{model_key}_B.csv")
    return pd.read_csv(DATA_DIR / f"{model_key}_{condition}.csv")


def paired_frame(model_key: str) -> pd.DataFrame:
    full = load_condition(model_key, "full")
    phase = load_condition(model_key, "phase_oracle")
    target = load_condition(model_key, "target_oracle")
    frame = full[["case_id", "category", "task_name"]].copy()
    frame["full"] = first_attempt_success(full).to_numpy()
    frame = frame.merge(
        phase[["case_id"]].assign(phase=first_attempt_success(phase).to_numpy()),
        on="case_id",
        validate="one_to_one",
    )
    frame = frame.merge(
        target[["case_id"]].assign(target=first_attempt_success(target).to_numpy()),
        on="case_id",
        validate="one_to_one",
    )
    if len(frame) != 2_133:
        raise ValueError(f"{model_key}: expected 2,133 paired rows, found {len(frame)}")
    frame["task_cluster"] = frame["category"].astype(str) + "/" + frame["task_name"].astype(str)
    return frame


def clustered_bootstrap(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    clusters = frame["task_cluster"].drop_duplicates().tolist()
    if len(clusters) != 34:
        raise ValueError(f"expected 34 task clusters, found {len(clusters)}")
    cluster_rows = {
        cluster: frame.loc[
            frame["task_cluster"] == cluster, ["full", "phase", "target"]
        ].to_numpy(float)
        for cluster in clusters
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty((BOOTSTRAP_REPLICATES, 6), dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        rows = np.concatenate([cluster_rows[cluster] for cluster in sampled], axis=0)
        rates = rows.mean(axis=0)
        estimates[index] = (
            rates[0],
            rates[1],
            rates[2],
            rates[1] - rates[0],
            rates[2] - rates[0],
            rates[2] - rates[1],
        )
    bounds = np.quantile(estimates, [0.025, 0.975], axis=0)
    names = (
        "full",
        "phase",
        "target",
        "phase_minus_full",
        "target_minus_full",
        "target_minus_phase",
    )
    return {name: (float(bounds[0, i]), float(bounds[1, i])) for i, name in enumerate(names)}


def observed_failure_class(frame: pd.DataFrame) -> pd.Series:
    classes = np.full(len(frame), "condition_failure", dtype=object)
    success = first_attempt_success(frame).to_numpy()
    parse_ok = frame["model_parse_ok"].astype(bool).to_numpy()
    subtype = frame["actual_failure_subtype"].fillna("").astype(str)
    structural = subtype.isin(STRUCTURAL_FAILURES).to_numpy()
    gate = frame["actual_gate1_ok"].astype(bool).to_numpy()
    native = frame["actual_native_ok"].astype(bool).to_numpy()
    classes[success] = "success"
    classes[~parse_ok] = "output_schema_failure"
    classes[parse_ok & structural] = "procedural_structure_failure"
    classes[parse_ok & ~structural & ~gate] = "target_context_failure"
    classes[gate & ~native] = "native_execution_failure"
    classes[native & ~success] = "condition_failure"
    return pd.Series(classes, index=frame.index)


def build_result_tables() -> dict[str, pd.DataFrame]:
    main_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    task_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    contrasts = (
        ("full", "phase", "phase_minus_full"),
        ("full", "target", "target_minus_full"),
        ("phase", "target", "target_minus_phase"),
    )
    for model_key, model_name in MODELS.items():
        frame = paired_frame(model_key)
        intervals = clustered_bootstrap(frame)
        task_rates = frame.groupby(
            ["category", "task_name", "task_cluster"], sort=False
        )[["full", "phase", "target"]].agg(["size", "sum", "mean"])
        task_rates.columns = ["_".join(column) for column in task_rates.columns]
        task_rates = task_rates.reset_index()
        main_rows.append(
            {
                "model": model_name,
                "n": len(frame),
                "task_clusters": frame["task_cluster"].nunique(),
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "full_success_count": int(frame["full"].sum()),
                "full_success_rate": float(frame["full"].mean()),
                "full_cluster_ci_low": intervals["full"][0],
                "full_cluster_ci_high": intervals["full"][1],
                "phase_success_count": int(frame["phase"].sum()),
                "phase_success_rate": float(frame["phase"].mean()),
                "phase_cluster_ci_low": intervals["phase"][0],
                "phase_cluster_ci_high": intervals["phase"][1],
                "target_success_count": int(frame["target"].sum()),
                "target_success_rate": float(frame["target"].mean()),
                "target_cluster_ci_low": intervals["target"][0],
                "target_cluster_ci_high": intervals["target"][1],
                "phase_minus_full": float(frame["phase"].mean() - frame["full"].mean()),
                "phase_minus_full_ci_low": intervals["phase_minus_full"][0],
                "phase_minus_full_ci_high": intervals["phase_minus_full"][1],
                "target_minus_full": float(frame["target"].mean() - frame["full"].mean()),
                "target_minus_full_ci_low": intervals["target_minus_full"][0],
                "target_minus_full_ci_high": intervals["target_minus_full"][1],
                "target_minus_phase": float(frame["target"].mean() - frame["phase"].mean()),
                "target_minus_phase_ci_low": intervals["target_minus_phase"][0],
                "target_minus_phase_ci_high": intervals["target_minus_phase"][1],
                "full_task_macro_rate": float(task_rates["full_mean"].mean()),
                "phase_task_macro_rate": float(task_rates["phase_mean"].mean()),
                "target_task_macro_rate": float(task_rates["target_mean"].mean()),
            }
        )
        for _, task in task_rates.iterrows():
            task_rows.append(
                {
                    "model": model_name,
                    "category": task["category"],
                    "task_name": task["task_name"],
                    "task_cluster": task["task_cluster"],
                    "cases": int(task["full_size"]),
                    "full_success_count": int(task["full_sum"]),
                    "full_success_rate": float(task["full_mean"]),
                    "phase_success_count": int(task["phase_sum"]),
                    "phase_success_rate": float(task["phase_mean"]),
                    "target_success_count": int(task["target_sum"]),
                    "target_success_rate": float(task["target_mean"]),
                    "phase_minus_full": float(task["phase_mean"] - task["full_mean"]),
                    "target_minus_full": float(task["target_mean"] - task["full_mean"]),
                    "target_minus_phase": float(task["target_mean"] - task["phase_mean"]),
                }
            )
        for first, second, difference_name in contrasts:
            first_values = frame[first].astype(bool)
            second_values = frame[second].astype(bool)
            paired_rows.append(
                {
                    "model": model_name,
                    "first_condition": first,
                    "second_condition": second,
                    "both_success": int((first_values & second_values).sum()),
                    "first_only": int((first_values & ~second_values).sum()),
                    "second_only": int((~first_values & second_values).sum()),
                    "neither": int((~first_values & ~second_values).sum()),
                    "second_minus_first": float(second_values.mean() - first_values.mean()),
                    "cluster_ci_low": intervals[difference_name][0],
                    "cluster_ci_high": intervals[difference_name][1],
                }
            )
            differences = task_rates[f"{second}_mean"] - task_rates[f"{first}_mean"]
            distribution_rows.append(
                {
                    "model": model_name,
                    "first_condition": first,
                    "second_condition": second,
                    "positive_task_clusters": int((differences > 0).sum()),
                    "zero_task_clusters": int((differences == 0).sum()),
                    "negative_task_clusters": int((differences < 0).sum()),
                    "median_task_difference": float(differences.median()),
                    "minimum_task_difference": float(differences.min()),
                    "maximum_task_difference": float(differences.max()),
                }
            )
    return {
        "oracle_main_results_clustered.csv": pd.DataFrame(main_rows),
        "oracle_paired_recovery.csv": pd.DataFrame(paired_rows),
        "oracle_task_level_results.csv": pd.DataFrame(task_rows),
        "oracle_task_recovery_distribution.csv": pd.DataFrame(distribution_rows),
    }


def build_failure_tables() -> dict[str, pd.DataFrame]:
    observed_rows: list[dict[str, object]] = []
    adjudicated_rows: list[dict[str, object]] = []
    adjudication_cases: list[dict[str, object]] = []
    for model_key, model_name in MODELS.items():
        for condition in ("full", "phase_oracle", "target_oracle"):
            frame = load_condition(model_key, condition)
            observed = observed_failure_class(frame)
            adjudicated = observed.copy()
            details = frame["actual_failure_detail"].fillna("").astype(str)
            mask = (
                frame["actual_failure_code"].eq("C5")
                & frame["actual_failure_subtype"].eq("AttributeError")
                & details.isin(ATTRIBUTE_ERROR_DETAILS)
            )
            adjudicated.loc[mask] = "target_context_failure"
            observed_counts = observed.value_counts()
            adjudicated_counts = adjudicated.value_counts()
            source = result_source(model_key, condition)
            for failure_class in FAILURE_CLASSES:
                observed_count = int(observed_counts.get(failure_class, 0))
                adjudicated_count = int(adjudicated_counts.get(failure_class, 0))
                observed_rows.append(
                    {
                        "model": model_name,
                        "condition": condition,
                        "failure_class": failure_class,
                        "count": observed_count,
                        "rate": observed_count / len(frame),
                        "classification_layer": "observed_first_failure_stage",
                        "source": source,
                    }
                )
                adjudicated_rows.append(
                    {
                        "model": model_name,
                        "condition": condition,
                        "failure_class": failure_class,
                        "count": adjudicated_count,
                        "rate": adjudicated_count / len(frame),
                        "classification_layer": "responsibility_adjudicated",
                        "source": source,
                    }
                )
            for index in frame.index[mask]:
                detail = str(frame.at[index, "actual_failure_detail"])
                adjudication_cases.append(
                    {
                        "model": model_name,
                        "condition": condition,
                        "case_id": frame.at[index, "case_id"],
                        "observed_failure_code": frame.at[index, "actual_failure_code"],
                        "observed_failure_subtype": frame.at[index, "actual_failure_subtype"],
                        "observed_failure_detail": detail,
                        "missing_native_method": (
                            "get_grasped_keypoints"
                            if "get_grasped_keypoints" in detail
                            else "get_place_point"
                        ),
                        "observed_failure_class": "native_execution_failure",
                        "adjudicated_failure_class": "target_context_failure",
                        "adjudication_reason": "affordance_gate_false_negative",
                        "primary_success_changed": False,
                        "source": source,
                    }
                )
    if len(adjudication_cases) != 472:
        raise ValueError(f"expected 472 adjudication cases, found {len(adjudication_cases)}")
    adjudication = pd.DataFrame(adjudication_cases)
    adjudication.to_csv(AUDIT_DIR / "oracle_attribute_error_adjudication.csv", index=False)
    return {
        "oracle_failure_taxonomy_observed.csv": pd.DataFrame(observed_rows),
        "oracle_failure_taxonomy_adjudicated.csv": pd.DataFrame(adjudicated_rows),
    }


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    tables = {**build_result_tables(), **build_failure_tables()}
    for filename, frame in tables.items():
        analysis_path = TABLE_DIR / filename
        frame.to_csv(analysis_path, index=False, quoting=csv.QUOTE_MINIMAL)
        shutil.copy2(analysis_path, RESULT_TABLE_DIR / filename)

    main_table = tables["oracle_main_results_clustered.csv"]
    expected = [[80, 427, 488], [57, 344, 585], [40, 314, 191]]
    actual = main_table[
        ["full_success_count", "phase_success_count", "target_success_count"]
    ].astype(int).values.tolist()
    if actual != expected:
        raise ValueError(f"unexpected success counts: {actual}")
    if not main_table[["phase_minus_full_ci_low", "target_minus_full_ci_low"]].gt(0).all().all():
        raise ValueError("an oracle-vs-full confidence interval includes zero")
    if not (
        main_table["target_minus_phase_ci_low"].le(0)
        & main_table["target_minus_phase_ci_high"].ge(0)
    ).all():
        raise ValueError("a target-vs-phase confidence interval excludes zero")
    print(
        json.dumps(
            {
                "status": "passed",
                "models": len(MODELS),
                "cases_per_model": 2_133,
                "task_clusters": 34,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "tables": sorted(tables),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
