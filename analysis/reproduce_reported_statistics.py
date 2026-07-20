from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent
DATA_DIR = REPOSITORY_ROOT / "analysis_data"
FIGURE_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"
REMOTE_AUDIT_DIR = DATA_DIR / "remote_audits"
GT_SELECTION_DIR = REMOTE_AUDIT_DIR / "gt_two_replay_selection"

MODELS = {
    "qwen": "Qwen3-VL-8B",
    "gemma": "Gemma-3-12B-it",
    "internvl": "InternVL3-8B",
}

COLORS = {
    "symbolic": "#4C78A8",
    "procedural": "#E45756",
    "both": "#54A24B",
    "a_only": "#4C78A8",
    "b_only": "#F2CF5B",
    "neither": "#BAB0AC",
}

BOOTSTRAP_SEED = 20260711
BOOTSTRAP_REPLICATES = 20_000


def load_paired_rows(model_key: str) -> pd.DataFrame:
    a = pd.read_csv(DATA_DIR / f"{model_key}_A.csv")
    b = pd.read_csv(DATA_DIR / f"{model_key}_B.csv")
    paired = a.merge(
        b[
            [
                "case_id",
                "model_parse_ok",
                "actual_gate1_ok",
                "actual_native_ok",
                "procedural_success",
                "native_selected_attempt",
                "native_attempt_count",
                "actual_failure_code",
                "actual_failure_subtype",
            ]
        ],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != 2_133:
        raise ValueError(f"{model_key}: expected 2,133 paired rows, found {len(paired)}")
    paired["task_cluster"] = paired["category"].astype(str) + "/" + paired["task_name"].astype(str)
    paired["a_success"] = paired["type_normalized_symbolic_success"].astype(bool)
    paired["b_success_any_attempt"] = paired["procedural_success"].astype(bool)
    paired["b_success"] = paired["b_success_any_attempt"] & paired["native_selected_attempt"].eq(1)
    return paired


def clustered_bootstrap(paired: pd.DataFrame) -> dict[str, tuple[float, float]]:
    clusters = paired["task_cluster"].drop_duplicates().tolist()
    cluster_rows = {
        cluster: paired.loc[paired["task_cluster"] == cluster, ["a_success", "b_success"]].to_numpy(float)
        for cluster in clusters
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty((BOOTSTRAP_REPLICATES, 3), dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        rows = np.concatenate([cluster_rows[cluster] for cluster in sampled], axis=0)
        a_rate = rows[:, 0].mean()
        b_rate = rows[:, 1].mean()
        estimates[index] = (a_rate, b_rate, a_rate - b_rate)
    intervals = np.quantile(estimates, [0.025, 0.975], axis=0)
    return {
        "a": (float(intervals[0, 0]), float(intervals[1, 0])),
        "b": (float(intervals[0, 1]), float(intervals[1, 1])),
        "difference": (float(intervals[0, 2]), float(intervals[1, 2])),
    }


def exact_mcnemar_p(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    return float(binomtest(min(first_only, second_only), discordant, 0.5, alternative="two-sided").pvalue)


def holm_adjust(raw_p_values: list[float]) -> list[float]:
    order = np.argsort(raw_p_values)
    adjusted = np.empty(len(raw_p_values), dtype=float)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (len(raw_p_values) - rank) * raw_p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    paired_by_model = {key: load_paired_rows(key) for key in MODELS}

    main_rows = []
    agreement_rows = []
    funnel_rows = []
    normalization_rows = []
    failure_rows = []
    category_rows = []
    retry_rows = []

    structural_failures = {"extra_phase", "missing_phase", "phase_skill_mismatch"}

    for key, display_name in MODELS.items():
        paired = paired_by_model[key]
        intervals = clustered_bootstrap(paired)
        n = len(paired)
        a_count = int(paired["a_success"].sum())
        b_count = int(paired["b_success"].sum())
        b_any_attempt_count = int(paired["b_success_any_attempt"].sum())
        a_rate = a_count / n
        b_rate = b_count / n
        difference = a_rate - b_rate

        task_rates = paired.groupby("task_cluster")[["a_success", "b_success"]].mean()
        main_rows.append(
            {
                "model": display_name,
                "n": n,
                "task_clusters": paired["task_cluster"].nunique(),
                "a_exact_count": a_count,
                "a_exact_rate": a_rate,
                "a_cluster_ci_low": intervals["a"][0],
                "a_cluster_ci_high": intervals["a"][1],
                "b_success_count": b_count,
                "b_success_rate": b_rate,
                "b_cluster_ci_low": intervals["b"][0],
                "b_cluster_ci_high": intervals["b"][1],
                "paired_difference": difference,
                "difference_cluster_ci_low": intervals["difference"][0],
                "difference_cluster_ci_high": intervals["difference"][1],
                "a_task_macro_rate": float(task_rates["a_success"].mean()),
                "b_task_macro_rate": float(task_rates["b_success"].mean()),
            }
        )
        retry_rows.append(
            {
                "model": display_name,
                "first_attempt_success_count": b_count,
                "first_attempt_success_rate": b_rate,
                "success_within_10_attempts_count": b_any_attempt_count,
                "success_within_10_attempts_rate": b_any_attempt_count / n,
                "additional_successes_after_retry": b_any_attempt_count - b_count,
            }
        )

        both = int((paired["a_success"] & paired["b_success"]).sum())
        a_only = int((paired["a_success"] & ~paired["b_success"]).sum())
        b_only = int((~paired["a_success"] & paired["b_success"]).sum())
        neither = int((~paired["a_success"] & ~paired["b_success"]).sum())
        agreement_rows.append(
            {
                "model": display_name,
                "both": both,
                "a_only": a_only,
                "b_only": b_only,
                "neither": neither,
                "overall_agreement": (both + neither) / n,
                "success_set_jaccard": both / (both + a_only + b_only),
                "b_only_among_b_success": b_only / b_count if b_count else np.nan,
                "mcnemar_exact_p": exact_mcnemar_p(a_only, b_only),
            }
        )

        subtype = paired["actual_failure_subtype"].fillna("")
        parse_count = int(paired["model_parse_ok"].sum())
        structural_failure_count = int(subtype.isin(structural_failures).sum())
        structural_valid_count = parse_count - structural_failure_count
        gate1_count = int(paired["actual_gate1_ok"].sum())
        native_count = int(paired["actual_native_ok"].sum())
        condition_count = b_count
        stage_values = [
            ("All cases", n),
            ("Parse/required fields valid", parse_count),
            ("Procedural structure valid", structural_valid_count),
            ("Target/context compatible", gate1_count),
            ("Native execution completed", native_count),
            ("Final condition satisfied", condition_count),
        ]
        previous = None
        for stage_order, (stage, count) in enumerate(stage_values):
            funnel_rows.append(
                {
                    "model": display_name,
                    "stage_order": stage_order,
                    "stage": stage,
                    "count": count,
                    "unconditional_rate": count / n,
                    "transition_rate": 1.0 if previous is None else count / previous if previous else np.nan,
                }
            )
            previous = count

        raw_exact = int(paired["symbolic_success"].sum())
        normalized_exact = int(paired["type_normalized_symbolic_success"].sum())
        raw_sr = float(paired["skill_match_score"].mean())
        raw_pr = float(paired["entity_match_score"].mean())
        raw_spr = float(paired["skill_with_entity_match_score"].mean())
        raw_pm = float(paired["exact_match_score"].mean())
        normalized_sr = float(paired["type_normalized_skill_match_score"].mean())
        normalized_pr = float(paired["type_normalized_entity_match_score"].mean())
        normalized_spr = float(paired["type_normalized_skill_with_entity_match_score"].mean())
        normalized_pm = float(paired["type_normalized_exact_match_score"].mean())
        normalization_rows.append(
            {
                "model": display_name,
                "raw_sr": raw_sr,
                "raw_pr": raw_pr,
                "raw_spr": raw_spr,
                "raw_pm": raw_pm,
                "raw_released_code_total": float(paired["total_score"].mean()),
                "raw_recomputed_four_metric_weighted_score": (
                    0.4 * raw_sr + 0.4 * raw_pr + 0.1 * raw_spr + 0.1 * raw_pm
                ),
                "normalized_sr": normalized_sr,
                "normalized_pr": normalized_pr,
                "normalized_spr": normalized_spr,
                "normalized_pm": normalized_pm,
                "normalized_released_code_total": float(paired["type_normalized_total_score"].mean()),
                "normalized_recomputed_four_metric_weighted_score": (
                    0.4 * normalized_sr
                    + 0.4 * normalized_pr
                    + 0.1 * normalized_spr
                    + 0.1 * normalized_pm
                ),
                "raw_pm_based_full_match_count": raw_exact,
                "raw_pm_based_full_match_rate": raw_exact / n,
                "normalized_derived_full_match_count": normalized_exact,
                "normalized_derived_full_match_rate": normalized_exact / n,
                "numeric_cast_applied_count": int(paired["type_normalized_numeric_cast_applied"].sum()),
                "raw_fail_normalized_pass_count": int(
                    ((~paired["symbolic_success"].astype(bool)) & paired["type_normalized_symbolic_success"].astype(bool)).sum()
                ),
            }
        )

        output_schema = n - parse_count
        procedural_structure = structural_failure_count
        target_context = structural_valid_count - gate1_count
        native_failure = gate1_count - native_count
        condition_failure = native_count - condition_count
        for outcome, count in [
            ("Output/schema failure", output_schema),
            ("Procedural-structure failure", procedural_structure),
            ("Target/context incompatibility", target_context),
            ("Native execution failure", native_failure),
            ("Final-condition failure", condition_failure),
            ("Success", condition_count),
        ]:
            failure_rows.append(
                {"model": display_name, "outcome": outcome, "count": count, "rate": count / n}
            )

        for (category, task_name), group in paired.groupby(["category", "task_name"]):
            category_rows.append(
                {
                    "model": display_name,
                    "category": category,
                    "task_name": task_name,
                    "cases": len(group),
                    "a_exact_rate": float(group["a_success"].mean()),
                    "b_success_rate": float(group["b_success"].mean()),
                    "paired_difference": float(group["a_success"].mean() - group["b_success"].mean()),
                }
            )

    main = pd.DataFrame(main_rows)
    agreement = pd.DataFrame(agreement_rows)
    funnel = pd.DataFrame(funnel_rows)
    normalization = pd.DataFrame(normalization_rows)
    failure = pd.DataFrame(failure_rows)
    task_results = pd.DataFrame(category_rows)
    category_results = (
        task_results.groupby(["model", "category"], as_index=False)
        .agg(
            task_clusters=("task_name", "nunique"),
            cases=("cases", "sum"),
            a_task_macro_rate=("a_exact_rate", "mean"),
            b_task_macro_rate=("b_success_rate", "mean"),
            difference_task_macro=("paired_difference", "mean"),
        )
    )
    dataset_composition = (
        paired_by_model["qwen"]
        .groupby("category")
        .agg(cases=("case_id", "size"), task_clusters=("task_cluster", "nunique"))
        .reset_index()
    )
    retry_sensitivity = pd.DataFrame(retry_rows)

    pairwise_rows = []
    keys = list(MODELS)
    for metric, column in [
        ("A type-normalized PM-based full-match (derived)", "a_success"),
        ("B first-attempt procedural success", "b_success"),
    ]:
        metric_rows = []
        raw_p = []
        for first_index in range(len(keys)):
            for second_index in range(first_index + 1, len(keys)):
                first_key = keys[first_index]
                second_key = keys[second_index]
                first = paired_by_model[first_key][["case_id", column]].rename(columns={column: "first"})
                second = paired_by_model[second_key][["case_id", column]].rename(columns={column: "second"})
                compared = first.merge(second, on="case_id", validate="one_to_one")
                first_only = int((compared["first"] & ~compared["second"]).sum())
                second_only = int((~compared["first"] & compared["second"]).sum())
                both = int((compared["first"] & compared["second"]).sum())
                neither = int((~compared["first"] & ~compared["second"]).sum())
                p_value = exact_mcnemar_p(first_only, second_only)
                raw_p.append(p_value)
                metric_rows.append(
                    {
                        "metric": metric,
                        "first_model": MODELS[first_key],
                        "second_model": MODELS[second_key],
                        "first_only": first_only,
                        "second_only": second_only,
                        "both": both,
                        "neither": neither,
                        "raw_p": p_value,
                    }
                )
        adjusted = holm_adjust(raw_p)
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["holm_adjusted_p"] = adjusted_p
            row["significant_at_0_05"] = adjusted_p < 0.05
            pairwise_rows.append(row)

    main.to_csv(TABLE_DIR / "main_paired_results_clustered.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    agreement.to_csv(TABLE_DIR / "ab_agreement_disagreement.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    funnel.to_csv(TABLE_DIR / "procedural_funnel_detailed.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    normalization.to_csv(TABLE_DIR / "experiment_a_released_evaluator_metrics.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    normalization.to_csv(TABLE_DIR / "symbolic_normalization_audit.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    failure.to_csv(TABLE_DIR / "procedural_failure_outcomes.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    task_results.to_csv(TABLE_DIR / "task_level_results.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    category_results.to_csv(TABLE_DIR / "category_macro_results.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    dataset_composition.to_csv(TABLE_DIR / "dataset_composition_2133.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(pairwise_rows).to_csv(TABLE_DIR / "pairwise_mcnemar_holm.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    retry_sensitivity.to_csv(TABLE_DIR / "retry_sensitivity.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    b_only_summary_path = REMOTE_AUDIT_DIR / "b_only_first_attempt_summary.csv"
    if b_only_summary_path.exists():
        pd.read_csv(b_only_summary_path).to_csv(
            TABLE_DIR / "b_only_first_attempt_summary.csv", index=False, quoting=csv.QUOTE_MINIMAL
        )

    return main, agreement, funnel


def build_dataset_selection_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    results = pd.read_csv(GT_SELECTION_DIR / "s0_full_4000_preload_target_official_results.csv")
    if len(results) != 4_000 or results["sample_id"].nunique() != 4_000:
        raise ValueError("Expected 4,000 unique GT eligibility rows")

    replay_1 = results["repeat_1_success"].astype(str).eq("True")
    replay_2 = results["repeat_2_success"].astype(str).eq("True")
    both = replay_1 & replay_2
    task_cluster = results["category"].astype(str) + "/" + results["task"].astype(str)

    stages = pd.DataFrame(
        [
            {"stage": "Initial non-complex set", "cases": len(results), "tasks": task_cluster.nunique()},
            {"stage": "GT replay 1 success", "cases": int(replay_1.sum()), "tasks": task_cluster[replay_1].nunique()},
            {"stage": "GT replay 2 success", "cases": int(replay_2.sum()), "tasks": task_cluster[replay_2].nunique()},
            {"stage": "Success in both replays", "cases": int(both.sum()), "tasks": task_cluster[both].nunique()},
        ]
    )
    stages["case_rate"] = stages["cases"] / len(results)
    stages.to_csv(TABLE_DIR / "dataset_selection_two_replay.csv", index=False)

    category_rows = []
    for category, group in results.assign(both_success=both).groupby("category", sort=False):
        retained = int(group["both_success"].sum())
        category_rows.append(
            {
                "category": category,
                "initial_cases": len(group),
                "retained_cases": retained,
                "retention_rate": retained / len(group),
            }
        )
    categories = pd.DataFrame(category_rows)
    categories.to_csv(TABLE_DIR / "dataset_retention_by_category.csv", index=False)

    task_rows = []
    for (category, task), group in results.assign(both_success=both).groupby(["category", "task"], sort=False):
        status_counts = group["official_status"].value_counts()
        retained = int(group["both_success"].sum())
        task_rows.append(
            {
                "category": category,
                "task": task,
                "initial_cases": len(group),
                "retained_cases": retained,
                "retention_rate": retained / len(group),
                "condition_failure": int(status_counts.get("B1_condition_failure", 0)),
                "native_exception": int(status_counts.get("B3_native_exception", 0)),
                "timeout": int(status_counts.get("B4_timeout", 0)),
                "cross_replay_mismatch": int(status_counts.get("B5_nondeterministic_mismatch", 0)),
            }
        )
    pd.DataFrame(task_rows).to_csv(TABLE_DIR / "dataset_retention_by_task.csv", index=False)
    return stages, categories


def build_expansion_stratified_table() -> pd.DataFrame:
    complexity = pd.read_csv(REMOTE_AUDIT_DIR / "procedural_complexity_by_case.csv")
    complexity["task_cluster"] = complexity["category"].astype(str) + "/" + complexity["task_name"].astype(str)
    model_columns = {
        "Qwen3-VL-8B": ("qwen3_a_success", "qwen3_b_first_success"),
        "Gemma-3-12B-it": ("gemma_a_success", "gemma_b_first_success"),
        "InternVL3-8B": ("internvl3_a_success", "internvl3_b_first_success"),
    }
    subset_masks = {
        "No added canonical phase": complexity["added_procedural_steps"].eq(0),
        "At least one added canonical phase": complexity["added_procedural_steps"].ge(1),
    }
    rows = []
    for subset_name, subset_mask in subset_masks.items():
        subset = complexity.loc[subset_mask].copy()
        for model_name, (a_column, b_column) in model_columns.items():
            paired = subset[["case_id", "task_cluster", a_column, b_column]].rename(
                columns={a_column: "a_success", b_column: "b_success"}
            )
            paired["a_success"] = paired["a_success"].astype(bool)
            paired["b_success"] = paired["b_success"].astype(bool)
            intervals = clustered_bootstrap(paired)
            a_rate = float(paired["a_success"].mean())
            b_rate = float(paired["b_success"].mean())
            rows.append(
                {
                    "subset": subset_name,
                    "model": model_name,
                    "cases": len(paired),
                    "task_clusters": paired["task_cluster"].nunique(),
                    "a_success_count": int(paired["a_success"].sum()),
                    "a_success_rate": a_rate,
                    "b_success_count": int(paired["b_success"].sum()),
                    "b_success_rate": b_rate,
                    "paired_difference": a_rate - b_rate,
                    "difference_cluster_ci_low": intervals["difference"][0],
                    "difference_cluster_ci_high": intervals["difference"][1],
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "expansion_stratified_results.csv", index=False)
    return result


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_main_results(main: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    x = np.arange(len(main))
    width = 0.34
    a = main["a_exact_rate"].to_numpy() * 100
    b = main["b_success_rate"].to_numpy() * 100
    a_low = (main["a_exact_rate"] - main["a_cluster_ci_low"]).to_numpy() * 100
    a_high = (main["a_cluster_ci_high"] - main["a_exact_rate"]).to_numpy() * 100
    b_low = (main["b_success_rate"] - main["b_cluster_ci_low"]).to_numpy() * 100
    b_high = (main["b_cluster_ci_high"] - main["b_success_rate"]).to_numpy() * 100

    bars_a = ax.bar(
        x - width / 2,
        a,
        width,
        color=COLORS["symbolic"],
        label="Type-normalized PM-based full-match (derived)",
        yerr=np.vstack([a_low, a_high]),
        capsize=4,
    )
    bars_b = ax.bar(
        x + width / 2,
        b,
        width,
        color=COLORS["procedural"],
        label="First-attempt canonical procedural success",
        yerr=np.vstack([b_low, b_high]),
        capsize=4,
    )
    for bars, counts in [(bars_a, main["a_exact_count"]), (bars_b, main["b_success_count"])]:
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.65,
                f"{bar.get_height():.2f}%\n({count}/2,133)",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_ylabel("Case-level success rate (%)")
    ax.set_xticks(x, main["model"])
    ax.set_ylim(0, max(a_high + a) + 5.0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    difference = main["paired_difference"].to_numpy() * 100
    diff_low = (difference - main["difference_cluster_ci_low"].to_numpy() * 100)
    diff_high = (main["difference_cluster_ci_high"].to_numpy() * 100 - difference)
    y = np.arange(len(main))
    axes[1].errorbar(
        difference,
        y,
        xerr=np.vstack([diff_low, diff_high]),
        fmt="o",
        color="#333333",
        ecolor="#777777",
        capsize=4,
        markersize=6,
    )
    axes[1].axvline(0, color="#A0A0A0", linewidth=1, linestyle="--")
    axes[1].set_yticks(y, main["model"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Paired A-B difference (%p)\nTask-clustered 95% CI")
    axes[1].grid(axis="x", color="#E0E0E0", linewidth=0.7)
    axes[1].set_axisbelow(True)
    axes[1].spines[["top", "right"]].set_visible(False)
    for index, value in enumerate(difference):
        axes[1].text(value + 0.35, index, f"{value:.2f}", va="center", fontsize=8)
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, "fig_responsibility_exposure_results")


def _box(ax, x: float, y: float, w: float, h: float, text: str, facecolor: str, edgecolor: str) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.2, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color="#555555"))


def plot_evaluation_framework() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, 0.31, 0.87, 0.38, 0.09, "Same 2,133 VLABench cases\nSame VLM backbones", "#F2F2F2", "#777777")
    _arrow(ax, (0.5, 0.87), (0.25, 0.77))
    _arrow(ax, (0.5, 0.87), (0.75, 0.77))
    ax.text(0.25, 0.80, "Experiment A", ha="center", fontsize=11, weight="bold", color="#315A86")
    ax.text(0.75, 0.80, "Experiment B", ha="center", fontsize=11, weight="bold", color="#A33D3B")
    left = [
        (0.08, 0.66, "Official-style symbolic interface"),
        (0.08, 0.49, "High-level skill sequence\n(skill + target + order)"),
        (0.08, 0.32, "Expert-DAG sequence scoring"),
        (0.08, 0.15, "Type-normalized PM-based\nfull-match (derived)"),
    ]
    right = [
        (0.58, 0.66, "Executor-derived canonical\nprocedural interface"),
        (0.58, 0.49, "Executor-derived phases\n(skill + target + order)"),
        (0.58, 0.32, "Validation + target binding\n+ native execution"),
        (0.58, 0.15, "First-attempt native\ntask-condition success"),
    ]
    for x, y, text_value in left:
        _box(ax, x, y, 0.34, 0.10, text_value, "#EAF2FA", "#4C78A8")
    for x, y, text_value in right:
        _box(ax, x, y, 0.34, 0.10, text_value, "#FCEDEC", "#E45756")
    for x in (0.25, 0.75):
        _arrow(ax, (x, 0.66), (x, 0.59))
        _arrow(ax, (x, 0.49), (x, 0.42))
        _arrow(ax, (x, 0.32), (x, 0.25))
    ax.text(0.5, 0.06, "Matched cases and backbones; separately generated plans and non-interchangeable endpoints",
            ha="center", fontsize=8.5, color="#444444")
    fig.tight_layout()
    save_figure(fig, "fig_evaluation_framework")


def plot_procedural_expansion() -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.95, "Representative task: select_toy", ha="center", fontsize=11, weight="bold")
    ax.text(0.19, 0.86, "Official symbolic interface", ha="center", fontsize=10, weight="bold", color="#315A86")
    _box(ax, 0.05, 0.66, 0.28, 0.10, "pick(target object)", "#EAF2FA", "#4C78A8")
    _arrow(ax, (0.19, 0.66), (0.19, 0.57))
    _box(ax, 0.05, 0.42, 0.28, 0.10, "place(target container)", "#EAF2FA", "#4C78A8")

    ax.text(0.66, 0.86, "Executor-derived canonical procedure", ha="center", fontsize=10, weight="bold", color="#A33D3B")
    procedural = [
        (0.48, 0.70, "pick(target object)"),
        (0.48, 0.54, "lift()"),
        (0.48, 0.38, "place(target container)"),
    ]
    for x, y, text_value in procedural:
        _box(ax, x, y, 0.36, 0.09, text_value, "#FCEDEC", "#E45756")
    _arrow(ax, (0.66, 0.70), (0.66, 0.63))
    _arrow(ax, (0.66, 0.54), (0.66, 0.47))
    ax.text(0.66, 0.25, "VLM explicitly selects each canonical skill,\nits required target, and the complete order.",
            ha="center", va="center", fontsize=8.5)
    _box(ax, 0.15, 0.08, 0.70, 0.09,
         "Fixed native backend: pose, orientation, offset, trajectory, gripper and MuJoCo control",
         "#EEF5EE", "#4C8C5A")
    _arrow(ax, (0.66, 0.38), (0.66, 0.17))
    fig.tight_layout()
    save_figure(fig, "fig_procedural_expansion")


def plot_dataset_selection(categories: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    _box(ax, 0.28, 0.82, 0.44, 0.11, "4,000 non-complex cases\n40 task types", "#F2F2F2", "#777777")
    _arrow(ax, (0.50, 0.82), (0.27, 0.65))
    _arrow(ax, (0.50, 0.82), (0.73, 0.65))
    _box(ax, 0.05, 0.52, 0.43, 0.12, "GT native replay 1\n2,341 passed (58.53%)\n37 task types", "#EAF2FA", "#4C78A8")
    _box(ax, 0.52, 0.52, 0.43, 0.12, "GT native replay 2\n2,316 passed (57.90%)\n35 task types", "#EAF2FA", "#4C78A8")
    _arrow(ax, (0.27, 0.52), (0.43, 0.35))
    _arrow(ax, (0.73, 0.52), (0.57, 0.35))
    _box(ax, 0.25, 0.20, 0.50, 0.14, "Intersection fixed before VLM evaluation\n2,133 cases (53.33%)\n34 task types", "#EEF5EE", "#4C8C5A")
    ax.text(0.5, 0.08, "The same fixed manifest is used in Experiments A and B.", ha="center", fontsize=8.5, color="#444444")

    ax = axes[1]
    values = categories["retention_rate"].to_numpy() * 100
    bars = ax.barh(categories["category"], values, color="#4C78A8", height=0.62)
    for bar, retained, initial, value in zip(
        bars, categories["retained_cases"], categories["initial_cases"], values
    ):
        ax.text(value + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{retained}/{initial} ({value:.2f}%)", va="center", fontsize=8.5)
    ax.set_xlabel("Retention after both GT native replays (%)")
    ax.set_xlim(0, 80)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, "fig_dataset_selection")


def plot_pooled_failure_taxonomy(failure: pd.DataFrame) -> None:
    order = [
        "Output/schema failure",
        "Procedural-structure failure",
        "Target/context incompatibility",
        "Native execution failure",
        "Final-condition failure",
        "Success",
    ]
    pooled = failure.groupby("outcome", as_index=True)["count"].sum().reindex(order)
    labels = [
        "Output/schema",
        "Canonical procedure",
        "Target/context",
        "Native execution",
        "Goal condition",
        "Success",
    ]
    colors = ["#7A7A7A", "#4C78A8", "#F2CF5B", "#B279A2", "#F58518", "#54A24B"]
    total = int(pooled.sum())
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.barh(labels[::-1], pooled.to_numpy()[::-1], color=colors[::-1], height=0.68)
    for bar, count in zip(bars, pooled.to_numpy()[::-1]):
        ax.text(bar.get_width() + 35, bar.get_y() + bar.get_height() / 2,
                f"{int(count):,} ({count / total * 100:.1f}%)", va="center", fontsize=8.5)
    pre_native = int(pooled.iloc[:3].sum())
    failures = total - int(pooled.loc["Success"])
    ax.text(0.98, 0.03,
            f"Pre-native: {pre_native:,}/{total:,} outcomes ({pre_native / total * 100:.1f}%); "
            f"{pre_native:,}/{failures:,} failures ({pre_native / failures * 100:.1f}%)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, weight="bold")
    ax.set_xlabel("Model-case outcomes pooled across three backbones")
    ax.set_xlim(0, max(pooled) * 1.22)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Mutually exclusive first-attempt outcome taxonomy", loc="left", fontsize=10, weight="bold")
    fig.tight_layout()
    save_figure(fig, "fig_procedural_failure_taxonomy")


def plot_agreement(agreement: pd.DataFrame) -> None:
    outcomes = ["both", "a_only", "b_only", "neither"]
    labels = ["Both", "A only", "B only", "Neither"]
    colors = [COLORS["both"], COLORS["a_only"], COLORS["b_only"], COLORS["neither"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": [1.25, 1]})

    left = np.zeros(len(agreement))
    for outcome, label, color in zip(outcomes, labels, colors):
        values = agreement[outcome].to_numpy() / 2_133 * 100
        axes[0].barh(agreement["model"], values, left=left, label=label, color=color, height=0.62)
        for index, (start, value, count) in enumerate(zip(left, values, agreement[outcome])):
            if value >= 2.2:
                axes[0].text(start + value / 2, index, f"{int(count)}", ha="center", va="center", fontsize=8)
        left += values
    axes[0].set_xlabel("Share of all 2,133 cases (%)")
    axes[0].set_xlim(0, 100)
    axes[0].grid(axis="x", color="#E0E0E0", linewidth=0.7)
    axes[0].set_axisbelow(True)
    axes[0].spines[["top", "right", "left"]].set_visible(False)
    axes[0].tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.01))

    union = agreement[["both", "a_only", "b_only"]].sum(axis=1)
    left = np.zeros(len(agreement))
    for outcome, label, color in zip(outcomes[:3], labels[:3], colors[:3]):
        values = agreement[outcome].to_numpy() / union.to_numpy() * 100
        axes[1].barh(agreement["model"], values, left=left, label=label, color=color, height=0.62)
        for index, (start, value, count) in enumerate(zip(left, values, agreement[outcome])):
            if value >= 8:
                axes[1].text(start + value / 2, index, f"{int(count)}", ha="center", va="center", fontsize=8)
        left += values
    axes[1].set_xlabel("Composition among cases successful in A or B (%)")
    axes[1].set_xlim(0, 100)
    axes[1].grid(axis="x", color="#E0E0E0", linewidth=0.7)
    axes[1].set_axisbelow(True)
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[1].tick_params(axis="y", length=0)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    for index, row in agreement.reset_index(drop=True).iterrows():
        axes[1].text(102, index, f"J={row['success_set_jaccard'] * 100:.1f}%", va="center", fontsize=8)
    axes[1].set_xlim(0, 118)

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, "fig_ab_agreement")


def main() -> None:
    from reproduce_oracle_statistics import main as reproduce_oracle_statistics

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "svg.fonttype": "none",
        }
    )
    main_results, agreement, funnel = build_tables()
    _, selection_categories = build_dataset_selection_tables()
    expansion_stratified = build_expansion_stratified_table()
    failure = pd.read_csv(TABLE_DIR / "procedural_failure_outcomes.csv")
    plot_evaluation_framework()
    plot_procedural_expansion()
    plot_dataset_selection(selection_categories)
    plot_main_results(main_results)
    plot_agreement(agreement)
    plot_pooled_failure_taxonomy(failure)
    reproduce_oracle_statistics()
    print(main_results.to_string(index=False))
    print(agreement.to_string(index=False))
    print(expansion_stratified.to_string(index=False))


if __name__ == "__main__":
    main()
