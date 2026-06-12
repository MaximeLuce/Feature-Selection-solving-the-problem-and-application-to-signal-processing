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
            tqdm.write(f"Row saved: case={row['Case_ID']} dataset={row['Dataset_ID']}")


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
                    config["sample_index"] = i + 1
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


def format_run_label(data):
    evaluation_model = data.get("evaluation_model")
    if evaluation_model is None and "evaluation_config" in data:
        evaluation_model = data["evaluation_config"]["evaluation_model"]

    fitness_name = data.get("fitness_name")
    if fitness_name is None and "evaluation_config" in data:
        fitness_name = data["evaluation_config"]["fitness"]

    decoder_name = data.get("decoder_name")
    if decoder_name is None and "decoder" in data:
        decoder_name = data["decoder"]["name"]

    label = (
        f"case={data['case_id']} dataset={data['dataset_id']} "
        f"evaluator={evaluation_model}"
    )
    if decoder_name is not None:
        label += f" decoder={decoder_name}"
    if fitness_name is not None:
        label += f" fitness={fitness_name}"
    if "sample_index" in data:
        label += f" sample={data['sample_index']}"
    return label


def format_group_key(group_key):
    if isinstance(group_key, tuple):
        if len(group_key) == 7:
            case_id, dataset_id, decoder, evaluator, fitness, alpha, cv_folds = group_key
            return (
                f"case={case_id} dataset={dataset_id} decoder={decoder} "
                f"evaluator={evaluator} fitness={fitness} alpha={alpha} cv={cv_folds}"
            )
        if len(group_key) == 6:
            case_id, dataset_id, evaluator, fitness, alpha, cv_folds = group_key
            return (
                f"case={case_id} dataset={dataset_id} "
                f"evaluator={evaluator} fitness={fitness} alpha={alpha} cv={cv_folds}"
            )
    return str(group_key)


def format_run_completed(result):
    return (
        f"Run completed: {format_run_label(result)} "
        f"best={result['best_fitness']:.4f} "
        f"score={result['final_score']:.2f} "
        f"nfe={result['evaluations_count']}"
    )


def run_parallel_configs(configurations, worker, max_workers=None, description="runs"):
    """Run all configurations in parallel."""
    worker_count = max_workers or (os.cpu_count() or 4)
    results = []

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for config in configurations:
            tqdm.write(f"Run started: {format_run_label(config)}")
        futures = {
            executor.submit(worker, config): config["run_id"]
            for config in configurations
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=description,
            unit="run",
        ):
            result = future.result()
            results.append(result)
            tqdm.write(format_run_completed(result))

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

    completed = set()
    if csv_filepath and resume_columns:
        completed = completed_group_keys(csv_filepath, resume_columns)
        skipped = sum(1 for group_key, _ in grouped_configs if group_key in completed)
        if skipped:
            tqdm.write(f"Resuming: skipping {skipped} already-completed groups.")

    pending = [
        (group_key, configs)
        for group_key, configs in grouped_configs
        if group_key not in completed
    ]
    if not pending:
        tqdm.write("All groups already completed.")
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
        for _, config in all_configs:
            tqdm.write(f"Run started: {format_run_label(config)}")
        futures = {
            executor.submit(worker, config): group_key
            for group_key, config in all_configs
        }

        pbar = tqdm(total=len(all_configs), desc=description, unit="run")
        for future in as_completed(futures):
            group_key = futures[future]
            result = future.result()
            group_results[group_key].append(result)
            pbar.update(1)
            tqdm.write(format_run_completed(result))

            done = len(group_results[group_key])
            needed = expected_counts[group_key]
            tqdm.write(
                f"Group progress: {format_group_key(group_key)} sample {done}/{needed}"
            )

            if done == needed:
                tqdm.write(
                    f"Aggregating row: {format_group_key(group_key)} samples={needed}"
                )
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
    tqdm.write("-" * 135)
    summary = " | ".join(
        f"{label} {row[column]}"
        for label, column in header_columns
    )
    tqdm.write(summary)
    tqdm.write(
        f"Champion score: {row['Champion_Score']:.2f}% "
        f"({row['Nb_Features_Keep']} features)"
    )
    tqdm.write("-" * 135)

    best_col, worst_col, avg_col, std_col = score_columns
    score_str = (
        f"{row[best_col]:>5.0f} {row[worst_col]:>6.0f} "
        f"{row[avg_col]:>6.1f} {row[std_col]:>5.1f}"
    )
    tqdm.write(f"{'Score':<15} | {score_str:<26}")

    wall_col = next(column for column in row if column.endswith("Wall_Time_Avg(s)"))
    cpu_col = next(column for column in row if column.endswith("CPU_Time_Avg(s)"))
    nfe_col = next(column for column in row if column.endswith("NFE_Avg"))

    tqdm.write(
        f"{'Time Avg':<15} | wall {row[wall_col]:<7.3f}s | "
        f"cpu {row[cpu_col]:<7.3f}s"
    )
    tqdm.write(f"{'NFE Avg':<15} | {row[nfe_col]:<26.1f} |")
    tqdm.write("-" * 135)
