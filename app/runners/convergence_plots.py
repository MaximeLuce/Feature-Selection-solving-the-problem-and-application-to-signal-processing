from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


EVALUATOR_ORDER = ["knn", "svm"]
EVALUATOR_LABELS = {"knn": "kNN", "svm": "SVM"}
CONVERGENCE_PLOT_PATH = Path("app/Results/SAParameters/final_convergence_by_nfe.png")
ALGORITHM_COLORS = {
    "BDE": "C0",
    "AMDE": "C1",
    "SA": "C2",
    "NBPSO": "C3",
    "NBPSO_LDIW": "C4",
}


def _load_case_history(path: str | Path, case_id: str, algorithm: str) -> pd.DataFrame:
    groups: dict[tuple[int, str], list[pd.DataFrame]] = {}

    with Path(path).open(encoding="utf-8") as file:
        for run_id, line in enumerate(file):
            if not line.strip():
                continue

            record = json.loads(line)
            config = record.get("config", {})
            if str(config.get("case_id", "")) != str(case_id):
                continue

            history = pd.DataFrame(record.get("history", []))
            if history.empty:
                continue

            history = history[["evaluations_count", "best_fitness"]].dropna()
            if history.empty:
                continue

            history = history.sort_values("evaluations_count").drop_duplicates(
                subset="evaluations_count",
                keep="last",
            )
            history["run_id"] = run_id

            group_key = (int(config["dataset_id"]), str(config["evaluation_model"]))
            groups.setdefault(group_key, []).append(history)

    frames = []
    for (dataset_id, evaluation_model), runs in groups.items():
        group = pd.concat(runs, ignore_index=True)
        stats = (
            group.groupby("evaluations_count", as_index=False)["best_fitness"]
            .agg(best_fitness_mean="mean", best_fitness_std="std")
            .sort_values("evaluations_count")
        )
        stats["best_fitness_std"] = stats["best_fitness_std"].fillna(0.0)
        stats["algorithm"] = algorithm
        stats["dataset_id"] = dataset_id
        stats["evaluation_model"] = evaluation_model
        frames.append(stats)

    if not frames:
        return pd.DataFrame(
            columns=[
                "algorithm",
                "dataset_id",
                "evaluation_model",
                "evaluations_count",
                "best_fitness_mean",
                "best_fitness_std",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def plot_final_convergence_by_nfe(sources: list[dict]) -> None:
    frames = []
    for source in sources:
        path = source.get("history_path") or source.get("path")
        if not path or not Path(path).exists():
            continue
        frame = _load_case_history(path, source["case_id"], source["algorithm"])
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return

    data = pd.concat(frames, ignore_index=True)
    dataset_order = sorted(data["dataset_id"].drop_duplicates().tolist())
    fig, axes = plt.subplots(
        len(dataset_order),
        len(EVALUATOR_ORDER),
        figsize=(12, 3.6 * len(dataset_order)),
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

    for (dataset_id, evaluation_model), subset in data.groupby(
        ["dataset_id", "evaluation_model"],
        sort=False,
    ):
        ax = axis_map[(dataset_id, evaluation_model)]
        ax.set_visible(True)
        subset = subset[subset["evaluations_count"] <= 10000]

        mean_by_nfe = subset.pivot(
            index="evaluations_count",
            columns="algorithm",
            values="best_fitness_mean",
        ).sort_index()
        std_by_nfe = subset.pivot(
            index="evaluations_count",
            columns="algorithm",
            values="best_fitness_std",
        ).reindex(mean_by_nfe.index)

        for algorithm, y in mean_by_nfe.items():
            y = y.dropna()
            std = std_by_nfe[algorithm].reindex(y.index).fillna(0.0)
            color = ALGORITHM_COLORS.get(algorithm)
            ax.plot(y.index, y, linewidth=2.0, color=color, label=algorithm)
            ax.fill_between(y.index, y - std, y + std, color=color, alpha=0.15)

        row = dataset_order.index(dataset_id)
        col = EVALUATOR_ORDER.index(evaluation_model)
        if row == 0:
            ax.set_title(EVALUATOR_LABELS[evaluation_model])
        if col == 0:
            ax.set_ylabel(f"{dataset_id}\nBest fitness")
        if row == len(dataset_order) - 1:
            ax.set_xlabel("NFE")
        ax.grid(True, linestyle="--", alpha=0.4)

    handles, labels = [], []
    for ax in axes.flat:
        if not ax.get_visible():
            continue
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            break

    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    CONVERGENCE_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONVERGENCE_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    from app.runners.analysis import FINAL_COMPARISON_SOURCES

    plot_final_convergence_by_nfe(FINAL_COMPARISON_SOURCES)
