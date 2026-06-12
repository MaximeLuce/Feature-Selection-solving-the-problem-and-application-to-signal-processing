import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm


def build_row_from_schema(values, schema):
    row = {}
    for item in schema:
        value = values[item["key"]]
        if "digits" in item:
            value = round(value, item["digits"])
        row[item["column"]] = value
    return row


def validate_and_order_row(row, columns):
    missing = [column for column in columns if column not in row]
    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")
    return {column: row[column] for column in columns}


def append_csv_rows(csv_filepath, columns, rows):
    file_exists = os.path.isfile(csv_filepath)
    with open(csv_filepath, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, delimiter=";")

        if not file_exists:
            writer.writeheader()

        for row in rows:
            writer.writerow(validate_and_order_row(row, columns))
            file.flush()
            print(f"Case {row['Case_ID']} saved!")


def build_evaluation_cases(config):
    cases = config.get("evaluation_cases")
    if cases:
        return cases
    return [
        {
            "evaluation_model": config.get("evaluation_model", "svm"),
            "fitness": config.get("fitness", "weighted_error"),
            "alpha": config.get("alpha", 0.5),
            "cv_folds": config.get("cv_folds", 5),
        }
    ]


def build_run_configs(dataset_ids, cases, evaluation_cases, runs_per_algo, build_case_config):
    run_id = 0
    order = 0
    for dataset_id in dataset_ids:
        for case in cases:
            for evaluation_case in evaluation_cases:
                base_seed = case["seed"]
                for i in range(runs_per_algo):
                    config = build_case_config(case)
                    config["seed"] = base_seed ^ (i * 10000001)
                    config["base_seed"] = base_seed
                    config["dataset_id"] = dataset_id
                    config["case_id"] = case["id"]
                    config["evaluation_config"] = dict(evaluation_case)
                    config["order"] = order
                    config["run_id"] = run_id
                    run_id += 1
                    yield config
                order += 1


def build_groups(configurations, group_keys_fn):
    """Group configs by group_keys_fn, return list of (group_key, [configs])."""
    groups = defaultdict(list)
    for config in configurations:
        groups[group_keys_fn(config)].append(config)
    return list(groups.items())


def aggregate_grouped_results(results, group_keys_fn, aggregate_fn):
    return [
        aggregate_fn(group_key, group)
        for group_key, group in build_groups(results, group_keys_fn)
    ]


def completed_group_keys(csv_filepath, key_columns):
    """Read existing CSV and return set of completed groups."""
    completed = set()
    if not os.path.isfile(csv_filepath):
        return completed

    with open(csv_filepath, mode="r", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            key = tuple(row.get(col, "") for col in key_columns)
            completed.add(key)
    return completed


def run_parallel_configs(configurations, worker, max_workers=None):
    """Run all configurations in parallel."""
    worker_count = max_workers or (os.cpu_count() or 4)
    results = []

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(worker, config): config["run_id"]
            for config in configurations
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="runs"):
            results.append(future.result())

    return results


def run_groups(grouped_configs, worker, aggregate_fn, write_fn, print_fn,
               resume_columns=None, csv_filepath=None,
               max_workers=None, description="runs"):
    """Execute individual runs and write a row when a whole group completes."""
    worker_count = max_workers or (os.cpu_count() or 4)
    grouped_configs = sorted(
        grouped_configs,
        key=lambda item: min(config["order"] for config in item[1]),
    )

    # Resume: skip already-completed groups
    completed = set()
    if csv_filepath and resume_columns:
        completed = completed_group_keys(csv_filepath, resume_columns)
        skipped = sum(1 for gk, _ in grouped_configs if gk in completed)
        if skipped:
            print(f"Resuming: skipping {skipped} already-completed groups.")

    # Filter out completed groups
    pending = [(gk, configs) for gk, configs in grouped_configs if gk not in completed]
    if not pending:
        print("All groups already completed.")
        return

    expected_counts = {
        group_key: len(configs)
        for group_key, configs in pending
    }
    group_results = defaultdict(list)
    all_configs = []
    for group_key, configs in pending:
        for config in sorted(configs, key=lambda config: config["run_id"]):
            all_configs.append((group_key, config))

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(worker, config): group_key
            for group_key, config in all_configs
        }

        pbar = tqdm(total=len(all_configs), desc=description)
        for future in as_completed(futures):
            group_key = futures[future]
            group_results[group_key].append(future.result())
            pbar.update(1)
            if len(group_results[group_key]) == expected_counts[group_key]:
                row = aggregate_fn(group_key, group_results[group_key])
                write_fn([row])
                print_fn(row)

        pbar.close()


def run_grouped_configs(configurations, group_keys_fn, worker, aggregate_fn,
                        write_fn, print_fn, resume_columns=None,
                        csv_filepath=None, max_workers=None,
                        description="runs"):
    grouped_configs = build_groups(configurations, group_keys_fn)
    if not grouped_configs:
        return

    run_groups(
        grouped_configs=grouped_configs,
        worker=worker,
        aggregate_fn=aggregate_fn,
        write_fn=write_fn,
        print_fn=print_fn,
        resume_columns=resume_columns,
        csv_filepath=csv_filepath,
        max_workers=max_workers,
        description=description,
    )


def aggregate_feature_selection_group(group):
    first = group[0]
    dataset_id = int(first["dataset_id"])
    group_df = pd.DataFrame(group)
    score_df = group_df["final_score"].astype(float)
    champion = group_df.loc[group_df["best_fitness"].astype(float).idxmin()]

    score_std = float(score_df.std())
    if pd.isna(score_std):
        score_std = 0.0

    return {
        "first": first,
        "dataset_id": dataset_id,
        "score_best": float(score_df.max()),
        "score_worst": float(score_df.min()),
        "score_avg": float(score_df.mean()),
        "score_std": score_std,
        "wall_avg_s": float(group_df["elapsed_wall_s"].mean()),
        "cpu_avg_s": float(group_df["elapsed_cpu_s"].mean()),
        "nfe_avg": float(group_df["evaluations_count"].mean()),
        "champion_score": float(champion["final_score"]),
        "nb_features_keep": int(champion["nb_features_keep"]),
    }


def print_feature_selection_summary(row, score_columns, header_columns):
    print("-" * 135)
    summary = " | ".join(
        f"{label} {row[column]}"
        for label, column in header_columns
    )
    print(summary)
    print(
        f"Champion score: {row['Champion_Score']:.2f}% "
        f"({row['Nb_Features_Keep']} features)"
    )
    print("-" * 135)

    best_col, worst_col, avg_col, std_col = score_columns
    score_str = (
        f"{row[best_col]:>5.0f} {row[worst_col]:>6.0f} "
        f"{row[avg_col]:>6.1f} {row[std_col]:>5.1f}"
    )
    print(f"{'Score':<15} | {score_str:<26}")

    wall_col = next(column for column in row if column.endswith("Wall_Time_Avg(s)"))
    cpu_col = next(column for column in row if column.endswith("CPU_Time_Avg(s)"))
    nfe_col = next(column for column in row if column.endswith("NFE_Avg"))

    print(
        f"{'Time Avg':<15} | wall {row[wall_col]:<7.3f}s | "
        f"cpu {row[cpu_col]:<7.3f}s"
    )
    print(f"{'NFE Avg':<15} | {row[nfe_col]:<26.1f} |")
    print("-" * 135)
