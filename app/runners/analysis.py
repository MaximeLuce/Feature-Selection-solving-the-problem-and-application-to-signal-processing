# app/runners/analysis.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.io import normalize_w_column, read_csv_frame, read_jsonl_case
from app.problem import Problem
from app.runners.de_parameters import DE_CSV_SCHEMA, DE_GROUP_FIELDS
from app.runners.nbpso_parameters import NBPSO_CSV_SCHEMA, NBPSO_GROUP_FIELDS
from app.runners.sa_parameters import SA_CSV_SCHEMA, SA_GROUP_FIELDS


FINAL_COMPARISON_COLUMNS = [
    "dataset_id", "evaluation_model", "fitness_name", "alpha", "cv_folds",
    "algorithm", "case_id", "full_score", "champion_score",
    "score_avg", "score_std", "avg_improvement", "avg_improvement_pct",
    "full_features", "nb_features_keep", "champion_features", "feature_reduction_pct",
    "wall_avg_s", "cpu_avg_s", "nfe_avg",
]

FINAL_COMPARISON_SOURCES = [
    {
        "algorithm": "BDE",
        "case_id": "12",
        "summary_path": "app/Results/SAParameters/BDE_Table6_F_CR_Sweep.csv",
        "history_path": "app/Results/raw/cases_BDE_Table6_F_CR_Sweep.jsonl",
        "csv_schema": DE_CSV_SCHEMA,
        "group_fields": DE_GROUP_FIELDS,
    },
    {
        "algorithm": "AMDE",
        "case_id": "42",
        "summary_path": "app/Results/SAParameters/AMDE_Table9_F_CR_Sweep.csv",
        "history_path": "app/Results/raw/cases_AMDE_Table9_F_CR_Sweep.jsonl",
        "csv_schema": DE_CSV_SCHEMA,
        "group_fields": DE_GROUP_FIELDS,
    },
    {
        "algorithm": "SA",
        "case_id": "50",
        "summary_path": "app/Results/SAParameters/SAParameters_cases_SA_Table10.csv",
        "history_path": "app/Results/raw/SA_raw_cases_SA_Table10.jsonl",
        "csv_schema": SA_CSV_SCHEMA,
        "group_fields": SA_GROUP_FIELDS,
    },
    {
        "algorithm": "NBPSO",
        "case_id": "60",
        "summary_path": "app/Results/SAParameters/NBPSO_Table12_Acceleration.csv",
        "history_path": "app/Results/raw/cases_NBPSO_Table12_Acceleration.jsonl",
        "csv_schema": NBPSO_CSV_SCHEMA,
        "group_fields": NBPSO_GROUP_FIELDS,
    },
    {
        "algorithm": "NBPSO_LDIW",
        "case_id": "64",
        "summary_path": "app/Results/SAParameters/NBPSO_Table14_LDIW.csv",
        "history_path": "app/Results/raw/cases_NBPSO_Table14_LDIW.jsonl",
        "csv_schema": NBPSO_CSV_SCHEMA,
        "group_fields": NBPSO_GROUP_FIELDS,
    },
]
def _records_frame(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = normalize_w_column(pd.json_normalize(records, sep="_"))
    keep = [column for column in df.columns if column.startswith("config_") or column.startswith("summary_") or column == "history"]
    df = df[keep]
    subset = [column for column in df.columns if column != "history"]
    if subset:
        df = df.drop_duplicates(subset=subset)
    return df


def load_history(records: list[dict]) -> pd.DataFrame:
    runs = _records_frame(records)
    if runs.empty or "history" not in runs.columns:
        raise ValueError("No history found in records.")
    runs = runs.reset_index(names="run_id").explode("history").dropna(subset=["history"])
    history = pd.json_normalize(runs.pop("history"), sep="_")
    runs = runs.rename(columns=lambda column: column.removeprefix("config_").removeprefix("summary_"))
    return normalize_w_column(pd.concat([runs.reset_index(drop=True), history.reset_index(drop=True)], axis=1))


def summarize_group(records: list[dict], group_fields: list[str]) -> pd.DataFrame:
    df = _records_frame(records).rename(columns=lambda column: column.removeprefix("config_").removeprefix("summary_"))
    for field in [field for field in group_fields if field in df.columns]:
        df[field] = df[field].map(
            lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        ).fillna("")

    summary = df.groupby(group_fields, as_index=False).agg(
        base_seed=("base_seed", "first"),
        score_best=("final_score", "max"),
        score_worst=("final_score", "min"),
        score_avg=("final_score", "mean"),
        score_std=("final_score", "std"),
        wall_avg_s=("elapsed_wall_s", "mean"),
        cpu_avg_s=("elapsed_cpu_s", "mean"),
        nfe_avg=("evaluations_count", "mean"),
        nb_features_keep=("nb_features_keep", "mean"),
    )
    champions = df.loc[df.groupby(group_fields)["best_fitness"].idxmin()]
    summary = summary.merge(
        champions[group_fields + ["final_score", "nb_features_keep"]].rename(
            columns={"final_score": "champion_score", "nb_features_keep": "champion_features"}
        ),
        on=group_fields,
    )
    summary["score_std"] = summary["score_std"].fillna(0.0)
    return summary


def best_fitness_by_generation(records: list[dict]) -> pd.DataFrame:
    return load_history(records).pivot_table(index="generation", columns="run_id", values="best_fitness")


def diversity_by_generation(records: list[dict[str, Any]]) -> pd.DataFrame:
    return load_history(records).pivot_table(index="generation", columns="run_id", values="population_diversity")


def selection_frequency(records: list[dict[str, Any]]) -> pd.DataFrame:
    runs = load_history(records)[["run_id", "best_mask"]].dropna().copy()
    if runs.empty:
        return pd.DataFrame()
    runs["feature_index"] = runs["best_mask"].map(lambda mask: list(range(len(mask))))
    runs = runs.explode(["feature_index", "best_mask"], ignore_index=True).astype({"best_mask": int})
    return runs.groupby(["feature_index", "run_id"])["best_mask"].sum().unstack("run_id", fill_value=0)


def selection_frequency_by_group(records: list[dict[str, Any]], group_fields: list[str]) -> pd.DataFrame:
    runs = (
        load_history(records)
        .sort_values(["run_id", "generation"])
        .groupby("run_id", as_index=False)
        .tail(1)
        .dropna(subset=["best_mask"])
    )
    keep = ["run_id", "best_mask"] + [field for field in group_fields if field in runs.columns]
    runs = runs[keep].copy()
    for field in [field for field in group_fields if field in runs.columns]:
        runs[field] = runs[field].map(
            lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
        ).fillna("")
    runs["feature_index"] = runs["best_mask"].map(lambda mask: list(range(len(mask))))
    runs = runs.explode(["feature_index", "best_mask"], ignore_index=True).astype({"best_mask": int})
    result = runs.groupby(group_fields + ["feature_index"], as_index=False).agg(
        selected_count=("best_mask", "sum"),
        n_runs=("run_id", "size"),
    )
    result["selected_frequency"] = result["selected_count"] / result["n_runs"]
    return result





def full_feature_baseline_from_cases(cases: pd.DataFrame) -> pd.DataFrame:

    fields = ["dataset_id", "evaluation_model", "fitness_name", "alpha", "cv_folds"]
    rows = []
    for case in cases[fields].drop_duplicates().sort_values(fields).itertuples(index=False):
        evaluation_config = {
            "evaluation_model": case.evaluation_model,
            "fitness": case.fitness_name,
            "alpha": case.alpha,
            "cv_folds": int(case.cv_folds),
        }
        try:
            problem = Problem.load_dataset(case.dataset_id, evaluation_config)
            full_score = float(problem.evaluate_final(np.ones(problem.num_features, dtype=int)) * 100)
            full_features = int(problem.num_features)
        except Exception as exc:
            print(f"Baseline skipped: dataset={case.dataset_id} evaluator={case.evaluation_model} error={exc}")
            full_score = np.nan
            full_features = np.nan
        rows.append(
            {
                "dataset_id": case.dataset_id,
                "evaluation_model": case.evaluation_model,
                "fitness_name": case.fitness_name,
                "alpha": case.alpha,
                "cv_folds": case.cv_folds,
                "full_score": full_score,
                "full_features": full_features,
            }
        )
    return pd.DataFrame(rows)


def feature_counts_from_sources(sources: list[dict[str, Any]]) -> pd.DataFrame:
    frames = [
        load_history(read_jsonl_case(path, source["case_id"]))[["dataset_id", "best_mask"]]
        for source in sources
        for path in [source.get("history_path") or source.get("path")]
        if path and Path(path).exists()
    ]
    if not frames:
        return pd.DataFrame(columns=["dataset_id", "full_features"])
    runs = pd.concat(frames, ignore_index=True).dropna(subset=["best_mask"]).drop_duplicates()
    runs["full_features"] = runs["best_mask"].map(len)
    return runs.groupby("dataset_id", as_index=False)["full_features"].max()


def summary_from_csv(
    path: str | Path,
    algorithm: str,
    case_id: str,
    csv_schema: list[dict[str, Any]],
) -> pd.DataFrame:
    df = read_csv_frame(path)
    rename_map = {item["column"]: item["key"] for item in csv_schema if item["column"] in df.columns}
    df = normalize_w_column(df.rename(columns=rename_map))
    df = df[df["case_id"].astype(str) == str(case_id)].copy()
    df["algorithm"] = algorithm
    return df


def final_comparison_table(sources: list[dict]) -> pd.DataFrame:
    summaries = []
    for source in sources:
        summary_path = source.get("summary_path")
        history_path = source.get("history_path") or source.get("path")
        if summary_path and Path(summary_path).exists():
            summaries.append(
                summary_from_csv(
                    summary_path,
                    source["algorithm"],
                    source["case_id"],
                    source["csv_schema"],
                )
            )
        elif history_path and Path(history_path).exists():
            summary = summarize_group(read_jsonl_case(history_path, source["case_id"]), source["group_fields"]).copy()
            summary["algorithm"] = source["algorithm"]
            summaries.append(summary)

    if not summaries:
        return pd.DataFrame()

    optimized = pd.concat(summaries, ignore_index=True)
    baseline = full_feature_baseline_from_cases(optimized)
    feature_counts = feature_counts_from_sources(sources)
    join_keys = ["dataset_id", "evaluation_model", "fitness_name", "alpha", "cv_folds"]
    comparison = optimized.merge(baseline, on=join_keys, how="left").merge(
        feature_counts, on="dataset_id", how="left", suffixes=("", "_from_raw")
    )
    comparison["full_features"] = comparison["full_features"].fillna(comparison["full_features_from_raw"])
    comparison["avg_improvement"] = comparison["score_avg"] - comparison["full_score"]
    comparison["avg_improvement_pct"] = 100 * comparison["avg_improvement"] / comparison["full_score"]
    comparison["feature_reduction_pct"] = 100 * (1 - comparison["nb_features_keep"] / comparison["full_features"])
    comparison = comparison.drop(columns=["full_features_from_raw"])
    return comparison.reindex(columns=FINAL_COMPARISON_COLUMNS).sort_values(
        ["dataset_id", "evaluation_model", "algorithm"]
    )








if __name__ == "__main__":
    comparison = final_comparison_table(FINAL_COMPARISON_SOURCES)
    if comparison.empty:
        print("No raw result files found for final comparison.")
    else:
        output_path = "app/Results/SAParameters/final_comparison.csv"
        comparison.to_csv(output_path, sep=";", index=False)
        print(f"Final comparison saved: {output_path}")
        print(comparison.to_string(index=False))



