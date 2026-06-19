from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from app.runners.convergence_plots import ALGORITHM_COLORS, EVALUATOR_LABELS, EVALUATOR_ORDER

FEATURE_FREQUENCY_151_PATH = Path("app/Results/SAParameters/final_feature_frequency_151.png")
FEATURE_COUNT_PATH = Path("app/Results/SAParameters/final_feature_count.png")


def _load_final_masks(
    sources: list[dict[str, Any]],
    dataset_id: int | None = None,
) -> pd.DataFrame:
    frames = []
    for source in sources:
        path = source.get("history_path") or source.get("path")
        if not path or not Path(path).exists():
            continue

        rows = []
        with Path(path).open(encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                config = record.get("config", {})
                if str(config.get("case_id", "")) != str(source["case_id"]):
                    continue
                rows.append(
                    {
                        "algorithm": source["algorithm"],
                        "evaluation_model": config["evaluation_model"],
                        "dataset_id": int(config["dataset_id"]),
                        "best_mask": next(
                            (
                                event["best_mask"]
                                for event in reversed(record.get("history", []))
                                if "best_mask" in event and event["best_mask"] is not None
                            ),
                            None,
                        ),
                    }
                )
        frame = pd.DataFrame(rows)
        if frame.empty:
            continue
        frame = frame[
            frame["evaluation_model"].isin(EVALUATOR_ORDER)
            & frame["best_mask"].notna()
        ]
        if dataset_id is not None:
            frame = frame[frame["dataset_id"] == dataset_id]
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=["algorithm", "evaluation_model", "dataset_id", "best_mask"])
    return pd.concat(frames, ignore_index=True)


def plot_feature_frequency_dataset(sources: list[dict]) -> None:
    dataset_id = 151
    runs = _load_final_masks(sources, dataset_id)
    if runs.empty:
        return

    runs["feature_index"] = runs["best_mask"].map(lambda mask: list(range(len(mask))))
    runs = runs.explode(["feature_index", "best_mask"], ignore_index=True).astype(
        {"feature_index": int, "best_mask": int}
    )

    freq = runs.groupby(
        ["algorithm", "evaluation_model", "feature_index"],
        as_index=False,
    )["best_mask"].mean().rename(columns={"best_mask": "selected_frequency"})
    cumulative = runs.groupby(
        ["evaluation_model", "feature_index"],
        as_index=False,
    )["best_mask"].mean().rename(columns={"best_mask": "selected_frequency"})
    cumulative["algorithm"] = "Cumulative"
    freq = pd.concat([freq, cumulative], ignore_index=True)

    algorithms = [source["algorithm"] for source in sources] + ["Cumulative"]
    fig, axes = plt.subplots(
        len(algorithms),
        len(EVALUATOR_ORDER),
        figsize=(14, 2.4 * len(algorithms)),
        squeeze=False,
        sharex=False,
        sharey=True,
    )

    for row, algorithm in enumerate(algorithms):
        for col, evaluation_model in enumerate(EVALUATOR_ORDER):
            ax = axes[row][col]
            subset = freq[
                (freq["algorithm"] == algorithm)
                & (freq["evaluation_model"] == evaluation_model)
            ].sort_values("feature_index")
            baseline = freq[
                (freq["algorithm"] == "Cumulative")
                & (freq["evaluation_model"] == evaluation_model)
            ].sort_values("feature_index")

            if subset.empty:
                ax.set_visible(False)
                continue

            baseline_series = baseline.set_index("feature_index")["selected_frequency"]
            subset_series = subset.set_index("feature_index")["selected_frequency"].reindex(
                baseline_series.index,
                fill_value=0.0,
            )

            ax.bar(
                baseline_series.index,
                baseline_series.values,
                color="darkgray",
                alpha=0.8,
                width=0.85,
            )
            if algorithm != "Cumulative":
                ax.bar(
                    subset_series.index,
                    subset_series.values,
                    color=ALGORITHM_COLORS.get(algorithm),
                    alpha=0.9,
                    width=0.55,
                )

            if row == 0:
                ax.set_title(EVALUATOR_LABELS[evaluation_model])
            if col == 0:
                ax.set_ylabel(algorithm)
            if row == len(algorithms) - 1:
                ax.set_xlabel("Feature")
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    FEATURE_FREQUENCY_151_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FEATURE_FREQUENCY_151_PATH, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_count_by_dataset_eval(sources: list[dict]) -> None:
    runs = _load_final_masks(sources)
    if runs.empty:
        return

    feature_totals = runs.assign(total_features=runs["best_mask"].map(len)).groupby("dataset_id")[
        "total_features"
    ].first()
    runs["feature_count"] = runs["best_mask"].map(sum)
    runs["feature_count_pct"] = 100 * runs["feature_count"] / runs["best_mask"].map(len)
    counts = runs.groupby(
        ["dataset_id", "evaluation_model", "algorithm"],
        as_index=False,
    )["feature_count"].agg(feature_count_mean="mean", feature_count_std="std")
    counts["feature_count_std"] = counts["feature_count_std"].fillna(0.0)
    cumulative = runs.groupby(
        ["evaluation_model", "algorithm"],
        as_index=False,
    )["feature_count_pct"].agg(feature_count_mean="mean", feature_count_std="std")
    cumulative["feature_count_std"] = cumulative["feature_count_std"].fillna(0.0)
    cumulative["dataset_id"] = "Cumulative"
    counts = pd.concat([counts, cumulative], ignore_index=True)

    dataset_order = sorted(runs["dataset_id"].drop_duplicates().tolist()) + ["Cumulative"]
    algorithms = [source["algorithm"] for source in sources]
    fig, axes = plt.subplots(
        len(dataset_order),
        len(EVALUATOR_ORDER),
        figsize=(12, 3.2 * len(dataset_order)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    axis_map = {
        (dataset_id, evaluation_model): axes[row][col]
        for row, dataset_id in enumerate(dataset_order)
        for col, evaluation_model in enumerate(EVALUATOR_ORDER)
    }

    for ax in axis_map.values():
        ax.set_visible(False)

    for (dataset_id, evaluation_model), subset in counts.groupby(
        ["dataset_id", "evaluation_model"],
        sort=False,
    ):
        ax = axis_map[(dataset_id, evaluation_model)]
        ax.set_visible(True)
        subset = subset.set_index("algorithm").reindex(algorithms).dropna().reset_index()
        ax.bar(
            subset["algorithm"],
            subset["feature_count_mean"],
            yerr=subset["feature_count_std"],
            color=subset["algorithm"].map(ALGORITHM_COLORS),
            capsize=4,
        )

        row = dataset_order.index(dataset_id)
        col = EVALUATOR_ORDER.index(evaluation_model)
        if row == 0:
            ax.set_title(EVALUATOR_LABELS[evaluation_model])
        if col == 0:
            ylabel = "Feature %" if dataset_id == "Cumulative" else "Feature count"
            ax.set_ylabel(f"{dataset_id}\n{ylabel}")
        if row == len(dataset_order) - 1:
            ax.set_xlabel("Algorithm")
        if dataset_id == "Cumulative":
            ax.set_ylim(0, 105)
            ax.axhline(100, color="black", linestyle="--", linewidth=1.2, zorder=3)
            ax.text(
                len(algorithms) - 0.5,
                98,
                "Max: 100",
                ha="right",
                va="top",
            )
        else:
            total = float(feature_totals.loc[dataset_id])
            ax.set_ylim(0, total * 1.05)
            ax.axhline(total, color="black", linestyle="--", linewidth=1.2, zorder=3)
            ax.text(
                len(algorithms) - 0.5,
                total * 0.98,
                f"Max: {int(total)}",
                ha="right",
                va="top",
            )
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    FEATURE_COUNT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FEATURE_COUNT_PATH, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    from app.runners.analysis import FINAL_COMPARISON_SOURCES

    #plot_feature_frequency_dataset(FINAL_COMPARISON_SOURCES)
    plot_feature_count_by_dataset_eval(FINAL_COMPARISON_SOURCES)
