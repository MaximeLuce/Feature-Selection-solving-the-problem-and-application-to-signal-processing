from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from app.runners.convergence_plots import ALGORITHM_COLORS, EVALUATOR_LABELS, EVALUATOR_ORDER


FINAL_COMPARISON_CSV = Path("app/Results/summary/final_comparison.csv")
FINAL_COMPARISON_PLOT_PATH = Path("app/Results/SAParameters/final_comparison_bars.png")


def plot_final_comparison_bars() -> None:
    data = pd.read_csv(FINAL_COMPARISON_CSV, sep=";").rename(
        columns={"avg_features_keep": "nb_features_keep"}
    )
    dataset_order = sorted(data["dataset_id"].drop_duplicates().tolist())
    algorithms = list(ALGORITHM_COLORS)
    fig, axes = plt.subplots(
        len(dataset_order),
        len(EVALUATOR_ORDER),
        figsize=(12, 3.2 * len(dataset_order)),
        squeeze=False,
        sharex=False,
        sharey=True,
    )

    for row, dataset_id in enumerate(dataset_order):
        for col, evaluation_model in enumerate(EVALUATOR_ORDER):
            ax = axes[row][col]
            subset = data[
                (data["dataset_id"] == dataset_id)
                & (data["evaluation_model"] == evaluation_model)
            ].set_index("algorithm").reindex(algorithms).dropna().reset_index()
            if subset.empty:
                ax.set_visible(False)
                continue

            ax.bar(
                subset["algorithm"],
                subset["score_avg"],
                yerr=subset["score_std"],
                color=subset["algorithm"].map(ALGORITHM_COLORS),
                capsize=4,
            )
            ax.axhline(subset["full_score"].iloc[0], color="black", linestyle="--", linewidth=1.2)

            if row == 0:
                ax.set_title(EVALUATOR_LABELS[evaluation_model])
            if col == 0:
                ax.set_ylabel(f"{dataset_id}\nScore")
            if row == len(dataset_order) - 1:
                ax.set_xlabel("Algorithm")
            ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    FINAL_COMPARISON_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL_COMPARISON_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_final_comparison_bars()
