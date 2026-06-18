# app/runners/analysis.py

from __future__ import annotations

import json
import os
import numpy as np
import pandas as pd


DE_GROUP_FIELDS = [
    "case_id", "dataset_id", "strategy", "popsize", "max_generations",
    "decoder_name", "evaluation_model", "fitness_name", "alpha", "cv_folds",
]
NBPSO_GROUP_FIELDS = [
    "case_id", "dataset_id", "swarm_size", "w", "w_max", "w_min", "c1", "c2", "vmax",
    "max_generations", "evaluation_model", "fitness_name", "alpha", "cv_folds",
]
SA_GROUP_FIELDS = [
    "case_id", "dataset_id", "initial_temp", "cooling_rate", "total_evals",
    "expected_final_temperature", "final_temperature",
    "evaluation_model", "fitness_name", "alpha", "cv_folds",
]

FINAL_COMPARISON_SOURCES = [
    {
        "algorithm": "BDE",
        "case_id": "12",
        "summary_path": "app/Results/SAParameters/BDE_Table6_F_CR_Sweep.csv",
        "history_path": "app/Results/raw/cases_BDE_Table6_F_CR_Sweep.jsonl",
        "group_fields": DE_GROUP_FIELDS,
    },
    {
        "algorithm": "AMDE",
        "case_id": "42",
        "summary_path": "app/Results/SAParameters/AMDE_Table9_F_CR_Sweep.csv",
        "history_path": "app/Results/raw/cases_AMDE_Table9_F_CR_Sweep.jsonl",
        "group_fields": DE_GROUP_FIELDS,
    },
    {
        "algorithm": "SA",
        "case_id": "50",
        "summary_path": "app/Results/SAParameters/SAParameters_cases_SA_Table10.csv",
        "history_path": "app/Results/raw/SA_raw_cases_SA_Table10.jsonl",
        "group_fields": SA_GROUP_FIELDS,
    },
    {
        "algorithm": "NBPSO",
        "case_id": "60",
        "summary_path": "app/Results/SAParameters/NBPSO_Table12_Acceleration.csv",
        "history_path": "app/Results/raw/cases_NBPSO_Table12_Acceleration.jsonl",
        "group_fields": NBPSO_GROUP_FIELDS,
    },
    {
        "algorithm": "NBPSO_LDIW",
        "case_id": "64",
        "summary_path": "app/Results/SAParameters/NBPSO_Table14_LDIW.csv",
        "history_path": "app/Results/raw/cases_NBPSO_Table14_LDIW.jsonl",
        "group_fields": NBPSO_GROUP_FIELDS,
    },
]

def _flatten(record: dict, run_id: int) -> dict:
    row: dict = {"run_id": run_id}
    row.update(record.get("config", {}))
    row.update(record.get("summary", {}))
    return row


def _flatten_runs(records: list[dict]) -> pd.DataFrame:
    """Build a flat DataFrame of one row per run."""
    return pd.DataFrame(_flatten(r, i) for i, r in enumerate(records))


def _serialize_dict_fields(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    df = df.copy()
    for field in fields:
        if field in df.columns:
            df[field] = df[field].apply(
                lambda v: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
            )
        else:
            df[field] = ""
        df[field] = df[field].fillna("")
    return df


def load_history(records: list[dict]) -> pd.DataFrame:
    rows = []
    for run_id, record in enumerate(records):
        flat = _flatten(record, run_id)
        for entry in record.get("history", []):
            row = dict(flat)
            row.update(entry)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_group(records: list[dict], group_fields: list[str]) -> pd.DataFrame:
    df = _serialize_dict_fields(_flatten_runs(records), group_fields)

    group = df.groupby(group_fields, as_index=False).agg(
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
    group = group.merge(
        champions[group_fields + ["final_score", "nb_features_keep"]].rename(
            columns={"final_score": "champion_score", "nb_features_keep": "champion_features"}
        ),
        on=group_fields,
    )
    group["score_std"] = group["score_std"].fillna(0.0)
    return group


def best_fitness_by_generation(records: list[dict]) -> pd.DataFrame:
    return load_history(records).pivot_table(index="generation", columns="run_id", values="best_fitness")


def diversity_by_generation(records: list[dict]) -> pd.DataFrame:
    return load_history(records).pivot_table(index="generation", columns="run_id", values="population_diversity")


def selection_frequency(records: list[dict]) -> pd.DataFrame:
    df = load_history(records)
    results: dict[int, np.ndarray] = {}
    max_len = 0
    for rid, group in df.groupby("run_id"):
        masks = group["best_mask"].dropna().tolist()
        if not masks:
            continue
        width = max(len(m) for m in masks)
        max_len = max(max_len, width)
        stacked = np.zeros((len(masks), width), dtype=int)
        for i, m in enumerate(masks):
            stacked[i, :len(m)] = m
        results[rid] = stacked.sum(axis=0)
    for rid in results:
        if len(results[rid]) < max_len:
            results[rid] = np.pad(results[rid], (0, max_len - len(results[rid])))
    return pd.DataFrame(results, index=range(max_len))


def selection_frequency_by_group(records: list[dict], group_fields: list[str]) -> pd.DataFrame:
    history = load_history(records)
    runs = history.sort_values(["run_id", "generation"]).groupby("run_id", as_index=False).tail(1)
    runs = runs[runs["best_mask"].notna()].copy()

    keep = ["run_id", "best_mask"] + [f for f in group_fields if f in runs.columns]
    runs = _serialize_dict_fields(runs[keep], group_fields)

    runs["feature_index"] = runs["best_mask"].map(lambda m: list(range(len(m))))
    runs = runs.explode(["feature_index", "best_mask"], ignore_index=True).astype({"best_mask": int})
    result = runs.groupby(group_fields + ["feature_index"], as_index=False, dropna=False).agg(
        selected_count=("best_mask", "sum"),
        n_runs=("run_id", "size"),
    )
    result["selected_frequency"] = result["selected_count"] / result["n_runs"]
    return result


def avg_best_generation_by_group(records: list[dict], group_fields: list[str]) -> pd.DataFrame:
    df = load_history(records)
    bf = df.pivot_table(index="run_id", columns="generation", values="best_fitness")
    first_best_gen = bf.idxmin(axis=1)
    best_fitness_per_run = bf.min(axis=1)
    df["best_found_gen"] = df["run_id"].map(first_best_gen)
    df["best_found_fitness"] = df["run_id"].map(best_fitness_per_run)
    df = _serialize_dict_fields(df, group_fields)
    return df.groupby(group_fields, as_index=False).agg(
        avg_best_gen=("best_found_gen", "mean"),
        std_best_gen=("best_found_gen", "std"),
        min_best_gen=("best_found_gen", "min"),
        max_best_gen=("best_found_gen", "max"),
        avg_best_fitness=("best_found_fitness", "mean"),
        n_runs=("run_id", "count"),
    ).sort_values("avg_best_gen")


def improvement_by_generation(records: list[dict], group_fields: list[str]) -> pd.DataFrame:
    runs = load_history(records).sort_values(["run_id", "generation"]).copy()
    runs["fitness_improvement"] = runs.groupby("run_id")["best_fitness"].diff().mul(-1).clip(lower=0)
    runs["improved"] = runs["fitness_improvement"].gt(0)
    return runs.groupby(group_fields + ["generation"], as_index=False).agg(
        improvement_rate=("improved", "mean"),
        avg_improvement=("fitness_improvement", "mean"),
        best_fitness=("best_fitness", "mean"),
    )



def full_feature_baseline_from_cases(cases: pd.DataFrame) -> pd.DataFrame:
    from app.Problem.Problem import Problem

    fields = ["dataset_id", "evaluation_model", "fitness_name", "alpha", "cv_folds"]
    cases = cases[fields].drop_duplicates().sort_values(fields)
    rows = []
    for case in cases.itertuples(index=False):
        evaluation_config = {
            "evaluation_model": case.evaluation_model,
            "fitness": case.fitness_name,
            "alpha": case.alpha,
            "cv_folds": int(case.cv_folds),
        }
        try:
            problem = Problem.load_dataset(case.dataset_id, evaluation_config)
            mask = np.ones(problem.num_features, dtype=int)
            full_score = float(problem.evaluate_final(mask) * 100)
            full_features = int(problem.num_features)
        except Exception as exc:
            print(f"Baseline skipped: dataset={case.dataset_id} evaluator={case.evaluation_model} error={exc}")
            full_score = np.nan
            full_features = np.nan
        rows.append({
            "dataset_id": case.dataset_id,
            "evaluation_model": case.evaluation_model,
            "fitness_name": case.fitness_name,
            "alpha": case.alpha,
            "cv_folds": case.cv_folds,
            "full_score": full_score,
            "full_features": full_features,
        })
    return pd.DataFrame(rows)




def _read_selected_raw(path: str, case_id: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("config", {}).get("case_id", "")) == str(case_id):
                records.append(record)
    return records


def _feature_counts_from_sources(sources: list[dict]) -> pd.DataFrame:
    rows = []
    for source in sources:
        path = source.get("history_path") or source.get("path")
        if not path or not os.path.exists(path):
            continue
        records = _read_selected_raw(path, source["case_id"])
        for record in records:
            history = record.get("history", [])
            first_mask = next(
                (
                    entry.get("best_mask")
                    for entry in history
                    if isinstance(entry.get("best_mask"), list)
                ),
                None,
            )
            if first_mask is not None:
                rows.append({
                    "dataset_id": record["config"]["dataset_id"],
                    "full_features": len(first_mask),
                })
    if not rows:
        return pd.DataFrame(columns=["dataset_id", "full_features"])
    return pd.DataFrame(rows).drop_duplicates("dataset_id")


def _rename_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "Case_ID": "case_id",
        "Dataset_ID": "dataset_id",
        "Evaluator": "evaluation_model",
        "Fitness": "fitness_name",
        "Alpha": "alpha",
        "CV_Folds": "cv_folds",
        "Champion_Score": "champion_score",
        "Champion_Features": "champion_features",
        "Nb_Features_Keep": "avg_features_keep",
    })
    
    for suffix, target in {
        "_Best": "score_best",
        "_Avg": "score_avg",
        "_Std": "score_std",
        "_Wall_Time_Avg(s)": "wall_avg_s",
        "_CPU_Time_Avg(s)": "cpu_avg_s",
        "_NFE_Avg": "nfe_avg",
    }.items():
        for col in df.columns:
            if col.endswith(suffix) or col == suffix.lstrip("_"):
                df[target] = df[col]
                break
    return df


def _summary_from_csv(path: str, algorithm: str, case_id: str) -> pd.DataFrame:
    df = _rename_csv_columns(pd.read_csv(path, sep=";"))
    df = df[df["case_id"].astype(str) == str(case_id)].copy()
    df["algorithm"] = algorithm
    return df


def final_comparison_table(sources: list[dict]) -> pd.DataFrame:
    summaries = []
    for source in sources:
        if source.get("summary_path") and os.path.exists(source["summary_path"]):
            summaries.append(_summary_from_csv(source["summary_path"], source["algorithm"], source["case_id"]))
        elif source.get("path") and os.path.exists(source["path"]):
            records = _read_selected_raw(source["path"], source["case_id"])
            s = summarize_group(records, source["group_fields"]).copy()
            s["algorithm"] = source["algorithm"]
            summaries.append(s)
    if not summaries:
        return pd.DataFrame()

    optimized = pd.concat(summaries, ignore_index=True)
    baseline = full_feature_baseline_from_cases(optimized)
    join_keys = ["dataset_id", "evaluation_model", "fitness_name", "alpha", "cv_folds"]
    table = optimized.merge(baseline, on=join_keys, how="left")
    feature_counts = _feature_counts_from_sources(sources)
    table = table.merge(feature_counts, on="dataset_id", how="left", suffixes=("", "_from_raw"))
    table["full_features"] = table["full_features"].fillna(table["full_features_from_raw"])
    table = table.drop(columns=["full_features_from_raw"])
    table["avg_improvement"] = table["score_avg"] - table["full_score"]
    table["avg_improvement_pct"] = 100 * table["avg_improvement"] / table["full_score"]
    table["feature_reduction_pct"] = 100 * (1 - table["avg_features_keep"] / table["full_features"])
    return table.reindex(columns=[
        "dataset_id", "evaluation_model", "fitness_name", "alpha", "cv_folds",
        "algorithm", "case_id", "full_score", "champion_score",
        "score_avg", "score_std", "avg_improvement", "avg_improvement_pct",
        "full_features", "avg_features_keep", "champion_features", "feature_reduction_pct",
        "wall_avg_s", "cpu_avg_s", "nfe_avg",
    ]).sort_values(["dataset_id", "evaluation_model", "algorithm"])


def best_algorithm_by_dataset(comparison: pd.DataFrame) -> pd.DataFrame:
    by = ["dataset_id", "evaluation_model", "fitness_name", "alpha", "cv_folds"]
    best = comparison.loc[comparison.groupby(by)["score_avg"].idxmax()].copy()
    best["rank_note"] = "best average score in group"
    return best.sort_values(["dataset_id", "evaluation_model"])


def dimensionality_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    return comparison[[
        "dataset_id", "evaluation_model", "algorithm", "full_features",
        "avg_features_keep", "feature_reduction_pct", "score_avg", "avg_improvement",
    ]].sort_values(["full_features", "dataset_id", "evaluation_model", "algorithm"])


def plot_final_comparison(comparison: pd.DataFrame):
    import matplotlib.pyplot as plt
    for group, data in comparison.groupby(["dataset_id", "evaluation_model"]):
        labels = ["full"] + data["algorithm"].astype(str).tolist()
        scores = [data["full_score"].iloc[0]] + data["score_avg"].tolist()
        errors = [0.0] + data["score_std"].fillna(0.0).tolist()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, scores, yerr=errors, capsize=4)
        ax.set_ylabel("Average final score")
        ax.set_title(f"Dataset {group[0]}  evaluator {group[1]}")
        plt.show()


def plot_feature_counts(comparison: pd.DataFrame):
    import matplotlib.pyplot as plt
    for group, data in comparison.groupby(["dataset_id", "evaluation_model"]):
        labels = ["full"] + data["algorithm"].astype(str).tolist()
        counts = [data["full_features"].iloc[0]] + data["avg_features_keep"].tolist()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(labels, counts)
        ax.set_ylabel("Feature count")
        ax.set_title(f"Dataset {group[0]}  evaluator {group[1]}")
        plt.show()


def plot_convergence_by_nfe(sources: list[dict]):
    import matplotlib.pyplot as plt
    frames = []
    for source in sources:
        path = source.get("history_path") or source.get("path")
        if not path or not os.path.exists(path):
            continue
        runs = load_history(_read_selected_raw(path, source["case_id"]))
        runs["algorithm"] = source["algorithm"]
        frames.append(runs)
    if not frames:
        return
    runs = pd.concat(frames, ignore_index=True)
    for group, data in runs.groupby(["dataset_id", "evaluation_model"]):
        stats = data.groupby(["algorithm", "evaluations_count"], as_index=False).agg(
            best_fitness=("best_fitness", "mean"),
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        for algo, algo_data in stats.groupby("algorithm"):
            ax.plot(algo_data["evaluations_count"], algo_data["best_fitness"], label=algo)
        ax.set_ylabel("Average best fitness")
        ax.set_xlabel("NFE")
        ax.set_title(f"Dataset {group[0]}  evaluator {group[1]}")
        ax.legend()
        plt.show()


if __name__ == "__main__":
    comparison = final_comparison_table(FINAL_COMPARISON_SOURCES)
    if comparison.empty:
        print("No raw result files found for final comparison.")
    else:
        output_path = "app/Results/SAParameters/final_comparison.csv"
        comparison.to_csv(output_path, sep=";", index=False)
        print(f"Final comparison saved: {output_path}")
        print(comparison.to_string(index=False))

        winners = best_algorithm_by_dataset(comparison)
        print("\nBest optimized algorithm per dataset/evaluator:")
        print(winners.to_string(index=False))

        dims = dimensionality_summary(comparison)
        print("\nDimensionality summary:")
        print(dims.to_string(index=False))
