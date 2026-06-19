# app/runners/common.py

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterator

import pandas as pd
from tqdm import tqdm

from app.core.io import read_csv_frame, write_csv_frame


def build_run_configs(
    dataset_ids: list[int],
    cases: list[dict[str, Any]],
    evaluation_cases: list[dict[str, Any]],
    runs_per_algo: int,
) -> Iterator[dict[str, Any]]:
    run_id = 0
    group_id = 0
    for dataset_id in dataset_ids:
        for case in cases:
            for evaluation_case in evaluation_cases:
                base_seed = case["seed"]
                for i in range(runs_per_algo):
                    config = dict(case)
                    config["dataset_id"] = dataset_id
                    config["seed"] = base_seed ^ (i * 10000001)
                    config["base_seed"] = base_seed
                    config["case_id"] = case["id"]
                    config["evaluation_config"] = dict(evaluation_case)
                    config["evaluation_model"] = evaluation_case["evaluation_model"]
                    config["fitness_name"] = evaluation_case.get("fitness", "")
                    config["alpha"] = evaluation_case.get("alpha", "")
                    config["cv_folds"] = evaluation_case.get("cv_folds", "")
                    decoder_conf = case.get("decoder")
                    config["decoder_name"] = decoder_conf["name"] if decoder_conf else ""
                    config["run_id"] = run_id
                    config["sample_id"] = i
                    config["group_id"] = group_id
                    run_id += 1
                    yield config
                group_id += 1


def write_csv_rows(csv_filepath: str | Path, rows_df: pd.DataFrame, append: bool = True) -> None:
    write_csv_frame(csv_filepath, rows_df, append=append)


def format_summary(summary_df: pd.DataFrame, csv_schema: list[dict[str, Any]]) -> pd.DataFrame:
    df = summary_df.copy()
    for item in csv_schema:
        key = item["key"]
        if key in df.columns and "digits" in item:
            df[key] = df[key].round(item["digits"])
    rename_map = {item["key"]: item["column"] for item in csv_schema}
    columns = [item["column"] for item in csv_schema]
    return df.rename(columns=rename_map).reindex(columns=columns)


def summarize_results(
    runs_df: pd.DataFrame,
    group_fields: list[str],
) -> pd.DataFrame:
    if runs_df.empty:
        raise ValueError("Cannot summarize without results.")

    summary = runs_df.groupby(group_fields, as_index=False).agg(
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
    champions = runs_df.loc[runs_df.groupby(group_fields)["best_fitness"].idxmin()]
    summary = summary.merge(
        champions[group_fields + ["final_score", "nb_features_keep"]].rename(
            columns={"final_score": "champion_score", "nb_features_keep": "champion_features"}
        ),
        on=group_fields,
    )
    summary["score_std"] = summary["score_std"].fillna(0.0)
    return summary


def _completed_group_keys(
    csv_filepath: str | Path,
    key_columns: list[str],
    csv_schema: list[dict[str, Any]],
) -> set[tuple[str, ...]]:
    path = Path(csv_filepath)
    if not path.is_file() or path.stat().st_size == 0:
        return set()

    key_to_column = {item["key"]: item["column"] for item in csv_schema}
    columns = [key_to_column[column] for column in key_columns]
    df = read_csv_frame(path, dtype=str, usecols=columns)
    if df.empty:
        return set()
    return set(df[columns].fillna("").drop_duplicates().itertuples(index=False, name=None))


def _pending_groups(
    configurations: list[dict[str, Any]],
    group_fields: list[str],
    csv_schema: list[dict[str, Any]],
    csv_filepath: str | Path,
) -> list[tuple[int, list[dict[str, Any]]]]:
    completed = _completed_group_keys(csv_filepath, group_fields, csv_schema)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for config in configurations:
        grouped.setdefault(config["group_id"], []).append(config)
    return [
        (group_id, configs)
        for group_id, configs in grouped.items()
        if tuple(str(configs[0].get(field, "")) for field in group_fields) not in completed
    ]


def _collect_run_results(
    run_row: dict[str, Any],
    config: dict[str, Any],
    group_runs: dict[int, list[dict[str, Any]]],
    expected_counts: dict[int, int],
    pbar: tqdm,
    group_fields: list[str],
    csv_schema: list[dict[str, Any]],
    csv_filepath: str | Path,
) -> None:
    group_id = config["group_id"]
    group_runs[group_id].append(run_row)
    finished = len(group_runs[group_id])
    expected = expected_counts[group_id]
    pbar.update(1)
    tqdm.write(
        "Run finished: "
        f"case={config['case_id']} "
        f"dataset={config['dataset_id']} "
        f"evaluator={config['evaluation_model']} "
        f"group_progress={finished}/{expected}"
    )

    if finished == expected:
        summary_df = summarize_results(pd.DataFrame(group_runs.pop(group_id)), group_fields)
        write_csv_rows(csv_filepath, format_summary(summary_df, csv_schema), append=True)


def run_configs(
    configurations: list[dict[str, Any]],
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    csv_filepath: str | Path,
    csv_schema: list[dict[str, Any]],
    group_fields: list[str],
    parallel: bool = True,
    max_workers: int | None = None,
) -> None:
    pending_groups = _pending_groups(configurations, group_fields, csv_schema, csv_filepath)
    pending_configs = [config for _, group in pending_groups for config in group]
    group_runs = {group_id: [] for group_id, _ in pending_groups}
    expected_counts = {group_id: len(group) for group_id, group in pending_groups}

    worker_count = max_workers or (os.cpu_count() or 4)
    if pending_configs:
        if parallel:
            tqdm.write(f"Submitting {len(pending_configs)} runs across {worker_count} workers.")
        else:
            tqdm.write(f"Running {len(pending_configs)} runs sequentially.")
    else:
        tqdm.write("No pending runs to execute.")

    pbar = tqdm(total=len(pending_configs), desc="completed runs", unit="run")

    if not parallel:
        for config in pending_configs:
            try:
                run_row = worker(config)
            except Exception as exc:
                pbar.update(1)
                tqdm.write(
                    f"ERROR: Worker failed for case={config['case_id']} "
                    f"dataset={config['dataset_id']} "
                    f"evaluator={config['evaluation_model']}: {exc}"
                )
                tqdm.write("  Raw data for this run was NOT saved. Re-run to retry.")
                continue
            _collect_run_results(run_row, config, group_runs, expected_counts, pbar, group_fields, csv_schema, csv_filepath)
        pbar.close()
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(worker, config): config for config in pending_configs}
        for future in as_completed(futures):
            config = futures[future]
            try:
                run_row = future.result()
            except Exception as exc:
                pbar.update(1)
                tqdm.write(
                    f"ERROR: Worker failed for case={config['case_id']} "
                    f"dataset={config['dataset_id']} "
                    f"evaluator={config['evaluation_model']}: {exc}"
                )
                tqdm.write("  Raw data for this run was NOT saved. Re-run to retry.")
                continue
            _collect_run_results(run_row, config, group_runs, expected_counts, pbar, group_fields, csv_schema, csv_filepath)
    pbar.close()
