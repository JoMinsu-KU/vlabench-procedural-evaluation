#!/usr/bin/env python3
"""Manuscript tables and figures for the symbolic endpoint tiers (artifact v1.7).

Inputs (all archived; no inference or rollout):
  analysis_data/{qwen,gemma,internvl}_{A,B}.csv
  reproducibility_core/cross_scoring/full_to_symbolic_projection_rows.csv
  reproducibility_core/cross_scoring/symbolic_endpoint_tiers.csv

Outputs:
  results/reported_statistics/symbolic_endpoint_tier_rates.csv
  figures/fig_interface_success_rates_tiers.{png,svg}
  figures/fig_ab_agreement_tiers.{png,svg}
  figures/fig_frozen_output_tiers.{png,svg}
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis_data"
CROSS_DIR = ROOT / "reproducibility_core" / "cross_scoring"
FIGURE_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "results" / "reported_statistics"

MODELS = {"qwen": "Qwen3-VL-8B", "gemma": "Gemma-3-12B-it", "internvl": "InternVL3-8B"}
SHORT = {"Qwen3-VL-8B": "Qwen", "Gemma-3-12B-it": "Gemma", "InternVL3-8B": "InternVL"}
COLORS = {
    "pm100": "#4C78A8",
    "strict": "#72B7B2",
    "strict_executor_filtered": "#9D755D",
    "procedural": "#E45756",
    "both": "#54A24B",
    "a_only": "#4C78A8",
    "b_only": "#F2CF5B",
    "neither": "#BAB0AC",
}
TIER_LABEL = {
    "pm100": "Released PM=100",
    "strict_executor_filtered": "Strict, executor phases removed",
    "strict": "Strict sequence equality",
}
BOOTSTRAP_SEED = 20260711
BOOTSTRAP_REPLICATES = 20_000


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().eq("true")


def load_paired() -> pd.DataFrame:
    cross = pd.read_csv(CROSS_DIR / "full_to_symbolic_projection_rows.csv")
    strict = {
        (row.model, row.case_id): str(row.original_strict_sequence_success).lower() == "true"
        for row in cross.itertuples()
    }
    frames = []
    for key, name in MODELS.items():
        a = pd.read_csv(DATA_DIR / f"{key}_A.csv")
        b = pd.read_csv(DATA_DIR / f"{key}_B.csv")
        merged = a[["case_id", "category", "task_name", "type_normalized_symbolic_success"]].merge(
            b[["case_id", "procedural_success", "native_selected_attempt"]], on="case_id", validate="one_to_one"
        )
        if len(merged) != 2_133:
            raise ValueError(f"{key}: expected 2,133 paired rows, found {len(merged)}")
        merged["model"] = name
        merged["task_cluster"] = merged["category"] + "/" + merged["task_name"]
        merged["pm100"] = as_bool(merged["type_normalized_symbolic_success"])
        merged["strict"] = [strict[(name, case_id)] for case_id in merged["case_id"]]
        merged["procedure"] = as_bool(merged["procedural_success"]) & merged["native_selected_attempt"].astype(int).eq(1)
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)


def cluster_bootstrap(frame: pd.DataFrame, columns: list[str]) -> dict[str, tuple[float, float]]:
    clusters = frame["task_cluster"].drop_duplicates().tolist()
    cluster_rows = {
        cluster: frame.loc[frame["task_cluster"] == cluster, columns].to_numpy(float) for cluster in clusters
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty((BOOTSTRAP_REPLICATES, len(columns)), dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        rows = np.concatenate([cluster_rows[cluster] for cluster in sampled], axis=0)
        estimates[index] = rows.mean(axis=0)
    low, high = np.quantile(estimates, [0.025, 0.975], axis=0)
    return {column: (float(low[i]), float(high[i])) for i, column in enumerate(columns)}


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_rate_table(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in MODELS.values():
        frame = paired[paired["model"] == name]
        n = len(frame)
        intervals = cluster_bootstrap(frame, ["pm100", "strict", "procedure"])
        base = {"model": name, "n": n}
        for column in ("pm100", "strict", "procedure"):
            count = int(frame[column].sum())
            base[f"{column}_count"] = count
            base[f"{column}_rate"] = count / n
            base[f"{column}_cluster_ci_low"], base[f"{column}_cluster_ci_high"] = intervals[column]
        for column in ("pm100", "strict"):
            both = int((frame[column] & frame["procedure"]).sum())
            a_only = int((frame[column] & ~frame["procedure"]).sum())
            b_only = int((~frame[column] & frame["procedure"]).sum())
            base[f"{column}_both"] = both
            base[f"{column}_symbolic_only"] = a_only
            base[f"{column}_procedure_only"] = b_only
            base[f"{column}_neither"] = n - both - a_only - b_only
            base[f"{column}_jaccard"] = both / (both + a_only + b_only)
        rows.append(base)
    table = pd.DataFrame(rows)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(TABLE_DIR / "symbolic_endpoint_tier_rates.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    return table


def plot_success_rates(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x = np.arange(len(table))
    width = 0.26
    series = [
        ("pm100", "Symbolic, released PM=100", COLORS["pm100"]),
        ("strict", "Symbolic, strict sequence equality", COLORS["strict"]),
        ("procedure", "Procedure success (first attempt)", COLORS["procedural"]),
    ]
    for offset, (column, label, color) in zip((-width, 0, width), series):
        rate = table[f"{column}_rate"].to_numpy() * 100
        low = rate - table[f"{column}_cluster_ci_low"].to_numpy() * 100
        high = table[f"{column}_cluster_ci_high"].to_numpy() * 100 - rate
        bars = ax.bar(x + offset, rate, width, color=color, label=label, yerr=np.vstack([low, high]), capsize=3)
        for bar, value in zip(bars, rate):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + high.max() * 0 + 0.35, f"{value:.2f}%",
                    ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Case-level success rate (%)")
    ax.set_xticks(x, [SHORT[m] for m in table["model"]])
    ax.set_ylim(0, (table[["pm100_cluster_ci_high", "strict_cluster_ci_high"]].to_numpy().max() * 100) + 4)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    fig.tight_layout()
    save_figure(fig, "fig_interface_success_rates_tiers")


def plot_agreement(table: pd.DataFrame) -> None:
    labels, all_share, union_share, jaccard = [], [], [], []
    for row in table.itertuples():
        for column in ("pm100", "strict"):
            labels.append(f"{SHORT[row.model]}\n{'PM=100' if column == 'pm100' else 'strict'}")
            both = getattr(row, f"{column}_both")
            a_only = getattr(row, f"{column}_symbolic_only")
            b_only = getattr(row, f"{column}_procedure_only")
            neither = getattr(row, f"{column}_neither")
            all_share.append((both, a_only, b_only, neither))
            union_share.append((both, a_only, b_only))
            jaccard.append(getattr(row, f"{column}_jaccard"))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})
    y = np.arange(len(labels))
    keys = ["both", "a_only", "b_only", "neither"]
    names = ["Both", "Symbolic only", "Procedure only", "Neither"]
    ax = axes[0]
    left = np.zeros(len(labels))
    for index, (key, name) in enumerate(zip(keys, names)):
        values = np.array([share[index] for share in all_share], dtype=float)
        pct = values / np.array([sum(s) for s in all_share]) * 100
        ax.barh(y, pct, left=left, color=COLORS[key], label=name, height=0.68)
        for j, (value, p) in enumerate(zip(values, pct)):
            if p > 4:
                ax.text(left[j] + p / 2, y[j], f"{int(value)}", ha="center", va="center", fontsize=8)
        left += pct
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of all 2,133 cases (%)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=4, fontsize=8)
    ax = axes[1]
    left = np.zeros(len(labels))
    for index, key in enumerate(keys[:3]):
        values = np.array([share[index] for share in union_share], dtype=float)
        pct = values / np.array([sum(s) for s in union_share]) * 100
        ax.barh(y, pct, left=left, color=COLORS[key], height=0.68)
        for j, (value, p) in enumerate(zip(values, pct)):
            if p > 5:
                ax.text(left[j] + p / 2, y[j], f"{int(value)}", ha="center", va="center", fontsize=8)
        left += pct
    for j, value in enumerate(jaccard):
        ax.text(101, y[j], f"J={value * 100:.1f}%", va="center", fontsize=8)
    ax.set_yticks(y, [""] * len(labels))
    ax.invert_yaxis()
    ax.set_xlim(0, 118)
    ax.set_xlabel("Composition among cases successful in either interface (%)")
    for axis in axes:
        axis.grid(axis="x", color="#E6E6E6", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    save_figure(fig, "fig_ab_agreement_tiers")


def plot_frozen_output_tiers() -> None:
    tiers = pd.read_csv(CROSS_DIR / "symbolic_endpoint_tiers.csv")
    order = ["pm100", "strict_executor_filtered", "strict"]
    models = list(MODELS.values())
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(models))
    width = 0.26
    for offset, tier in zip((-width, 0, width), order):
        sub = tiers[(tiers["symbolic_endpoint"] == tier) & tiers["model"].isin(models)].set_index("model").loc[models]
        passed = sub["projected_symbolic_success_n"].to_numpy(float)
        downstream = sub["projected_success_full_failure_n"].to_numpy(float)
        also_full = passed - downstream
        ax.bar(x + offset, also_full, width, color=COLORS[tier], edgecolor="white", linewidth=0.5, label=TIER_LABEL[tier])
        ax.bar(x + offset, downstream, width, bottom=also_full, color=COLORS[tier], alpha=0.45, edgecolor="white",
               linewidth=0.5, hatch="///")
        for xi, total, down in zip(x + offset, passed, downstream):
            ax.text(xi, total + 3, f"{int(total)}\n({int(down)} fail)", ha="center", va="bottom", fontsize=7.5)
    solid = plt.Rectangle((0, 0), 1, 1, color="#777777")
    hatched = plt.Rectangle((0, 0), 1, 1, color="#777777", alpha=0.45, hatch="///")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [solid, hatched], labels + ["also passed full procedure", "failed full procedure"],
              frameon=False, fontsize=8, loc="upper right", ncol=1)
    ax.set_xticks(x, [SHORT[m] for m in models])
    ax.set_ylabel("Projected symbolic successes (of 2,133 frozen outputs)")
    ax.set_ylim(0, tiers[tiers["model"].isin(models)]["projected_symbolic_success_n"].max() * 1.32)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, "fig_frozen_output_tiers")


def main() -> None:
    paired = load_paired()
    table = build_rate_table(paired)
    plot_success_rates(table)
    plot_agreement(table)
    plot_frozen_output_tiers()
    print(table[["model", "pm100_count", "strict_count", "procedure_count", "pm100_jaccard", "strict_jaccard"]].to_string(index=False))


if __name__ == "__main__":
    main()
