from __future__ import annotations

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = ANALYSIS_DIR.parent
TABLE_DIR = ANALYSIS_DIR / "tables"
FIGURE_DIR = ANALYSIS_DIR / "figures"
DATA_DIR = REPOSITORY_ROOT / "analysis_data"
ORACLE_DATA_DIR = DATA_DIR
ORACLE_VALIDATION_PATH = (
    REPOSITORY_ROOT
    / "results"
    / "validation_evidence"
    / "oracle_ablation_validation_20260715.json"
)

MAIN_PATH = TABLE_DIR / "oracle_main_results_clustered.csv"
TAXONOMY_PATH = TABLE_DIR / "oracle_failure_taxonomy_adjudicated.csv"

MODEL_ORDER = ["Qwen3-VL-8B", "Gemma-3-12B-it", "InternVL3-8B"]
MODEL_LABELS = ["Qwen3-VL-8B", "Gemma-3-12B-it", "InternVL3-8B"]
MODEL_KEYS = [("qwen", "Qwen3-VL-8B"), ("gemma", "Gemma-3-12B-it"), ("internvl", "InternVL3-8B")]
BOOTSTRAP_SEED = 20_260_711
BOOTSTRAP_REPLICATES = 20_000

COLORS = {
    "full": "#6F6F6F",
    "phase": "#4C78A8",
    "target": "#F58518",
}


def _validate_sources(main: pd.DataFrame, taxonomy: pd.DataFrame) -> None:
    if main["model"].tolist() != MODEL_ORDER:
        raise ValueError("Unexpected model ordering in oracle main-results table")
    if not main["n"].eq(2_133).all():
        raise ValueError("Every oracle model condition must contain 2,133 cases")
    if not main["task_clusters"].eq(34).all():
        raise ValueError("Oracle bootstrap must use 34 task clusters")
    if not main["bootstrap_seed"].eq(20_260_711).all():
        raise ValueError("Unexpected oracle bootstrap seed")
    if not main["bootstrap_replicates"].eq(20_000).all():
        raise ValueError("Unexpected oracle bootstrap replicate count")

    for prefix in ("phase_minus_full", "target_minus_full"):
        if not ((main[f"{prefix}_ci_low"] > 0) & (main[f"{prefix}_ci_high"] > 0)).all():
            raise ValueError(f"Not all {prefix} intervals exclude zero on the positive side")
    if not (
        (main["target_minus_phase_ci_low"] <= 0)
        & (main["target_minus_phase_ci_high"] >= 0)
    ).all():
        raise ValueError("Every direct target-minus-phase interval must include zero")

    expected_conditions = {"full", "phase_oracle", "target_oracle"}
    if set(taxonomy["condition"]) != expected_conditions:
        raise ValueError("Unexpected condition set in adjudicated taxonomy")
    totals = taxonomy.groupby(["model", "condition"])["count"].sum()
    if not totals.eq(2_133).all():
        raise ValueError("Adjudicated taxonomy does not sum to 2,133 per model-condition")
    native = taxonomy.loc[taxonomy["failure_class"].eq("native_execution_failure"), "count"]
    if not native.eq(0).all():
        raise ValueError("Adjudicated intrinsic native-failure counts must be zero")


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    svg_path = FIGURE_DIR / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    svg_lines = svg_path.read_text(encoding="utf-8").splitlines()
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_lines) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def _first_attempt_success(frame: pd.DataFrame) -> pd.Series:
    return frame["procedural_success"].astype(bool) & frame["native_selected_attempt"].eq(1)


def build_simultaneous_oracle_contrasts(main: pd.DataFrame) -> pd.DataFrame:
    """Build one six-contrast family using a joint task-cluster max-|T| bootstrap."""

    frames: dict[str, pd.DataFrame] = {}
    for model_key, _ in MODEL_KEYS:
        full = pd.read_csv(DATA_DIR / f"{model_key}_B.csv")
        phase = pd.read_csv(ORACLE_DATA_DIR / f"{model_key}_phase_oracle.csv")
        target = pd.read_csv(ORACLE_DATA_DIR / f"{model_key}_target_oracle.csv")
        frame = full[["case_id", "category", "task_name"]].copy()
        frame["full"] = _first_attempt_success(full).to_numpy()
        frame = frame.merge(
            phase[["case_id"]].assign(phase=_first_attempt_success(phase).to_numpy()),
            on="case_id",
            validate="one_to_one",
        )
        frame = frame.merge(
            target[["case_id"]].assign(target=_first_attempt_success(target).to_numpy()),
            on="case_id",
            validate="one_to_one",
        )
        if len(frame) != 2_133 or frame["case_id"].nunique() != 2_133:
            raise ValueError(f"{model_key}: expected 2,133 unique paired cases")
        frame["task_cluster"] = frame["category"].astype(str) + "/" + frame["task_name"].astype(str)
        frames[model_key] = frame

    clusters = frames["qwen"]["task_cluster"].drop_duplicates().tolist()
    if len(clusters) != 34:
        raise ValueError(f"expected 34 task clusters, found {len(clusters)}")
    for model_key, _ in MODEL_KEYS:
        if frames[model_key]["task_cluster"].drop_duplicates().tolist() != clusters:
            raise ValueError(f"{model_key}: task-cluster ordering differs from the common manifest")

    cluster_rows = {
        model_key: {
            cluster: frames[model_key].loc[
                frames[model_key]["task_cluster"].eq(cluster), ["full", "phase", "target"]
            ].to_numpy(float)
            for cluster in clusters
        }
        for model_key, _ in MODEL_KEYS
    }

    estimates: list[float] = []
    labels: list[tuple[str, str, str]] = []
    for model_key, model_name in MODEL_KEYS:
        rates = frames[model_key][["full", "phase", "target"]].mean().to_numpy()
        estimates.extend([rates[1] - rates[0], rates[2] - rates[0]])
        labels.extend(
            [
                (model_name, "phase_minus_full", "Phase-oracle - Full B"),
                (model_name, "target_minus_full", "Target-oracle - Full B"),
            ]
        )
    observed = np.asarray(estimates, dtype=float)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty((BOOTSTRAP_REPLICATES, 6), dtype=float)
    for replicate in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        column = 0
        for model_key, _ in MODEL_KEYS:
            rows = np.concatenate([cluster_rows[model_key][cluster] for cluster in sampled], axis=0)
            rates = rows.mean(axis=0)
            bootstrap[replicate, column] = rates[1] - rates[0]
            bootstrap[replicate, column + 1] = rates[2] - rates[0]
            column += 2

    standard_errors = bootstrap.std(axis=0, ddof=1)
    if np.any(standard_errors <= 0):
        raise ValueError("non-positive bootstrap standard error in primary contrast family")
    standardized = (bootstrap - observed) / standard_errors
    critical_value = float(np.quantile(np.max(np.abs(standardized), axis=1), 0.95))
    simultaneous_low = observed - critical_value * standard_errors
    simultaneous_high = observed + critical_value * standard_errors

    indexed_main = main.set_index("model")
    rows: list[dict[str, object]] = []
    for index, (model_name, contrast_key, contrast_label) in enumerate(labels):
        main_row = indexed_main.loc[model_name]
        rows.append(
            {
                "model": model_name,
                "contrast": contrast_label,
                "estimate": observed[index],
                "pointwise_ci_low": float(main_row[f"{contrast_key}_ci_low"]),
                "pointwise_ci_high": float(main_row[f"{contrast_key}_ci_high"]),
                "simultaneous_ci_low": simultaneous_low[index],
                "simultaneous_ci_high": simultaneous_high[index],
                "simultaneous_excludes_zero": bool(
                    simultaneous_low[index] > 0 or simultaneous_high[index] < 0
                ),
                "method": "joint_task_cluster_bootstrap_max_abs_t",
                "family_size": 6,
                "task_clusters": 34,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "max_abs_t_critical_value": critical_value,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "oracle_primary_contrasts_simultaneous.csv", index=False)
    if int(result["simultaneous_excludes_zero"].sum()) != 5:
        raise ValueError("expected five of six simultaneous intervals to exclude zero")
    return result


def write_no_repair_mutation_table() -> pd.DataFrame:
    validation = json.loads(
        ORACLE_VALIDATION_PATH.read_text(encoding="utf-8")
    )
    cases = int(validation["case_count"])
    total = int(validation["mutation_checks"])
    swapped = total - 2 * cases
    rows = [
        {
            "mutation_control": "Phase-oracle: remove one target slot",
            "checks": cases,
            "expected_behavior": "Reject without inferring or repairing the missing target",
            "validation_errors": 0,
        },
        {
            "mutation_control": "Target-oracle: remove one skill",
            "checks": cases,
            "expected_behavior": "Reject before reference-target access",
            "validation_errors": 0,
        },
        {
            "mutation_control": "Target-oracle: swap the first two skills when possible",
            "checks": swapped,
            "expected_behavior": "Reject without reordering and before reference-target access",
            "validation_errors": 0,
        },
    ]
    result = pd.DataFrame(rows)
    if int(result["checks"].sum()) != total or validation["semantic_repair_applied"]:
        raise ValueError("no-repair mutation accounting does not match recovered validation evidence")
    result.to_csv(TABLE_DIR / "oracle_no_repair_mutation_controls.csv", index=False)
    return result


def plot_oracle_responsibility_analysis(main: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.2),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    # Panel A: absolute first-attempt task-condition success.
    ax = axes[0]
    x = np.arange(len(main))
    width = 0.25
    conditions = [
        ("full", "Full B", "full_success_rate", "full_cluster_ci_low", "full_cluster_ci_high", "full_success_count"),
        ("phase", "Phase-oracle", "phase_success_rate", "phase_cluster_ci_low", "phase_cluster_ci_high", "phase_success_count"),
        ("target", "Target-oracle", "target_success_rate", "target_cluster_ci_low", "target_cluster_ci_high", "target_success_count"),
    ]
    offsets = [-width, 0.0, width]
    for offset, (color_key, label, rate_col, low_col, high_col, count_col) in zip(offsets, conditions):
        rates = main[rate_col].to_numpy() * 100
        lower = (main[rate_col] - main[low_col]).to_numpy() * 100
        upper = (main[high_col] - main[rate_col]).to_numpy() * 100
        bars = ax.bar(
            x + offset,
            rates,
            width,
            label=label,
            color=COLORS[color_key],
            yerr=np.vstack([lower, upper]),
            capsize=3.5,
            error_kw={"elinewidth": 1.0},
        )
        for bar, rate, count in zip(bars, rates, main[count_col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.0,
                f"{rate:.2f}%\n({int(count)})",
                ha="center",
                va="bottom",
                fontsize=7.4,
            )
    ax.set_xticks(x, MODEL_LABELS)
    ax.set_ylabel("First-attempt task-condition success (%)")
    ax.set_ylim(0, 48)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncols=3, fontsize=8)
    ax.set_title("A  Condition-level success", loc="left", fontsize=10, weight="bold")

    # Panel B: oracle-minus-full task-clustered contrasts.
    ax = axes[1]
    y = np.arange(len(main))
    shift = 0.14
    contrast_specs = [
        (
            "phase",
            "Phase − Full B",
            "phase_minus_full",
            "phase_minus_full_ci_low",
            "phase_minus_full_ci_high",
            -shift,
        ),
        (
            "target",
            "Target − Full B",
            "target_minus_full",
            "target_minus_full_ci_low",
            "target_minus_full_ci_high",
            shift,
        ),
    ]
    for color_key, label, value_col, low_col, high_col, offset in contrast_specs:
        values = main[value_col].to_numpy() * 100
        lower = (main[value_col] - main[low_col]).to_numpy() * 100
        upper = (main[high_col] - main[value_col]).to_numpy() * 100
        ax.errorbar(
            values,
            y + offset,
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color=COLORS[color_key],
            ecolor=COLORS[color_key],
            capsize=3.5,
            markersize=5.5,
            label=label,
        )
        for value, y_value in zip(values, y + offset):
            ax.text(value + 0.55, y_value, f"{value:+.2f}", va="center", fontsize=7.8)
    ax.axvline(0, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_yticks(y, MODEL_LABELS)
    ax.invert_yaxis()
    ax.set_xlim(-2, 42)
    ax.set_xlabel("Oracle − Full B difference (percentage points)\n95% task-cluster bootstrap interval")
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("B  Scaffold-conditioned contrasts", loc="left", fontsize=10, weight="bold")

    fig.tight_layout(w_pad=2.4)
    _save(fig, "fig_oracle_responsibility_controls")


def write_adjudicated_full_b_table(taxonomy: pd.DataFrame) -> pd.DataFrame:
    label_map = {
        "output_schema_failure": "Output/schema failure",
        "procedural_structure_failure": "Procedural-structure failure",
        "target_context_failure": "Target/context incompatibility",
        "native_execution_failure": "Native execution failure",
        "condition_failure": "Final-condition failure",
        "success": "Success",
    }
    full = taxonomy.loc[taxonomy["condition"].eq("full"), ["model", "failure_class", "count", "rate"]].copy()
    full["outcome"] = full["failure_class"].map(label_map)
    full["classification_layer"] = "responsibility_adjudicated"
    full = full[["model", "outcome", "count", "rate", "classification_layer"]]
    full.to_csv(TABLE_DIR / "procedural_failure_outcomes.csv", index=False)
    return full


def plot_adjudicated_full_b_taxonomy(full: pd.DataFrame) -> None:
    order = [
        "Output/schema failure",
        "Procedural-structure failure",
        "Target/context incompatibility",
        "Native execution failure",
        "Final-condition failure",
        "Success",
    ]
    labels = ["Output/schema", "Reference procedure", "Target/context", "Intrinsic native", "Goal condition", "Success"]
    colors = ["#7A7A7A", "#4C78A8", "#F2CF5B", "#B279A2", "#F58518", "#54A24B"]
    pooled = full.groupby("outcome")["count"].sum().reindex(order)
    total = int(pooled.sum())
    failures = total - int(pooled.loc["Success"])
    pre_native_responsibility = int(pooled.iloc[:3].sum())

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    bars = ax.barh(labels[::-1], pooled.to_numpy()[::-1], color=colors[::-1], height=0.68)
    for bar, count in zip(bars, pooled.to_numpy()[::-1]):
        ax.text(
            bar.get_width() + 35,
            bar.get_y() + bar.get_height() / 2,
            f"{int(count):,} ({count / total * 100:.1f}%)",
            va="center",
            fontsize=8.5,
        )
    ax.text(
        0.98,
        0.03,
        f"Pre-native responsibility: {pre_native_responsibility:,}/{failures:,} failures "
        f"({pre_native_responsibility / failures * 100:.1f}%)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        weight="bold",
    )
    ax.set_xlabel("Full-B model-case outcomes pooled across three backbones")
    ax.set_xlim(0, max(pooled) * 1.22)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Responsibility-adjudicated first-attempt outcome taxonomy", loc="left", fontsize=10, weight="bold")
    fig.tight_layout()
    _save(fig, "fig_procedural_failure_taxonomy")


def main() -> None:
    main_results = pd.read_csv(MAIN_PATH)
    taxonomy = pd.read_csv(TAXONOMY_PATH)
    _validate_sources(main_results, taxonomy)
    build_simultaneous_oracle_contrasts(main_results)
    write_no_repair_mutation_table()
    full_b = write_adjudicated_full_b_table(taxonomy)
    plot_adjudicated_full_b_taxonomy(full_b)
    plot_oracle_responsibility_analysis(main_results)
    print("Oracle manuscript assets generated from validated reference-procedure tables.")


if __name__ == "__main__":
    main()
