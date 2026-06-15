# app/runners/common.py

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import filelock
from tqdm import tqdm


def build_run_configs(dataset_ids, cases, evaluation_cases, runs_per_algo):
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


def ensure_csv_header(csv_filepath, columns):
    if not os.path.isfile(csv_filepath):
        return

    with open(csv_filepath, mode="r", newline="") as file:
        reader = csv.reader(file, delimiter=";")
        existing_header = next(reader, None)
    if existing_header != columns:
        raise ValueError(
            f"CSV header mismatch for {csv_filepath}. "
        )


def append_csv_rows(csv_filepath, columns, rows):
    ensure_csv_header(csv_filepath, columns)
    file_exists = os.path.isfile(csv_filepath)
    with open(csv_filepath, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, delimiter=";")

        if not file_exists:
            writer.writeheader()

        for row in rows:
            writer.writerow(validate_and_order_row(row, columns))
            file.flush()
            tqdm.write(
                f"Row saved: case={row['Case_ID']} dataset={row['Dataset_ID']} "
                f"evaluator={row['Evaluator']}"
            )


def save_raw_run_result(raw_dir, result_dict, filename):
    os.makedirs(raw_dir, exist_ok=True)
    filepath = os.path.join(raw_dir, filename)
    lock = filelock.FileLock(filepath + ".lock")
    with lock:
        with open(filepath, "a", encoding="utf-8") as file:
            file.write(json.dumps(result_dict) + "\n")


def _completed_group_keys(csv_filepath, key_columns, csv_schema):
    completed = set()
    if not os.path.isfile(csv_filepath):
        return completed

    key_to_column = {item["key"]: item["column"] for item in csv_schema}
    with open(csv_filepath, mode="r", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        for row in reader:
            key = tuple(
                str(row.get(key_to_column[column], ""))
                for column in key_columns
            )
            completed.add(key)
    return completed


def summarize_results_group(group_results, group_fields):
    if not group_results:
        raise ValueError("Cannot summarize an empty group.")

    first = group_results[0]
    champion = min(group_results, key=lambda result: float(result["best_fitness"]))
    final_scores = [float(result["final_score"]) for result in group_results]
    wall_times = [float(result["elapsed_wall_s"]) for result in group_results]
    cpu_times = [float(result["elapsed_cpu_s"]) for result in group_results]
    evaluations = [float(result["evaluations_count"]) for result in group_results]
    feature_counts = [float(result["nb_features_keep"]) for result in group_results]

    score_avg = sum(final_scores) / len(final_scores)
    score_std = 0.0
    if len(final_scores) > 1:
        variance = sum((score - score_avg) ** 2 for score in final_scores) / (len(final_scores) - 1)
        score_std = variance ** 0.5

    summary = {
        field: first[field]
        for field in group_fields
        if field in first
    }
    summary.update(
        {
            "base_seed": first["base_seed"],
            "score_best": max(final_scores),
            "score_worst": min(final_scores),
            "score_avg": score_avg,
            "score_std": score_std,
            "wall_avg_s": sum(wall_times) / len(wall_times),
            "cpu_avg_s": sum(cpu_times) / len(cpu_times),
            "nfe_avg": sum(evaluations) / len(evaluations),
            "champion_score": float(champion["final_score"]),
            "champion_features": int(champion["nb_features_keep"]),
            "nb_features_keep": sum(feature_counts) / len(feature_counts),
        }
    )
    return summary


def _pending_groups(configurations, group_fields, csv_schema, csv_filepath):
    completed = _completed_group_keys(csv_filepath, group_fields, csv_schema)
    grouped = {}
    for config in configurations:
        grouped.setdefault(config["group_id"], []).append(config)
    pending = []
    for group_id, configs in grouped.items():
        first = configs[0]
        key = tuple(
            str(first.get(field, ""))
            for field in group_fields
        )
        if key not in completed:
            pending.append((group_id, configs))
    return pending


def _collect_result(
    result, config, results, expected_counts, pbar, group_fields, csv_schema, csv_filepath
):
    
    columns = [item["column"] for item in csv_schema]
    group_id = config["group_id"]
    expected = expected_counts[group_id]
    finished = len(results[group_id])
    results[group_id].append(result)
    pbar.update(1)
    tqdm.write(
        "Run finished: "
        f"case={config['case_id']} "
        f"dataset={config['dataset_id']} "
        f"evaluator={config['evaluation_model']} "
        f"group_progress={finished}/{expected} "
    )

    if finished == expected:
        values = summarize_results_group(results.pop(group_id), group_fields)
        row = build_row_from_schema(values, csv_schema)
        append_csv_rows(csv_filepath, columns, [row])


def run_configs(
    configurations,
    worker,
    csv_filepath,
    csv_schema,
    group_fields,
    parallel=True,
    max_workers=None,
):
    columns = [item["column"] for item in csv_schema]
    ensure_csv_header(csv_filepath, columns)

    pending_groups = _pending_groups(configurations, group_fields, csv_schema, csv_filepath)
    pending_configs = [config for _, group in pending_groups for config in group]
    results = {group_id: [] for group_id, _ in pending_groups}
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
            result = worker(config)

            _collect_result(
                result, config, results, expected_counts, pbar,
                group_fields, csv_schema, csv_filepath
            )
        pbar.close()
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(worker, config): config for config in pending_configs}
        for future in as_completed(futures):
            config = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Worker failed for case={config['case_id']} "
                    f"dataset={config['dataset_id']} "
                    f"evaluator={config['evaluation_model']}"
                ) from exc

            _collect_result(
                result, config, results, expected_counts, pbar,
                group_fields, csv_schema, csv_filepath
            )
    pbar.close()
