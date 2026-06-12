# app/runners/common2.py

import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

import pandas as pd
from tqdm import tqdm

from app.Utilities.ConfigLoader import load_config

CSV_SCHEMA = [
    {"column": "Case_ID", "key": "case_id"},
    {"column": "Dataset_ID", "key": "dataset_id"},
    {"column": "Population_size", "key": "popsize"},
    {"column": "Generations_number", "key": "max_generations"},
    {"column": "Seed", "key": "base_seed"},
    {"column": "Decoder", "key": "decoder_name"},
    {"column": "Evaluator", "key": "evaluation_model"},
    {"column": "Fitness", "key": "fitness_name"},
    {"column": "Alpha", "key": "alpha"},
    {"column": "CV_Folds", "key": "cv_folds"},
    {"column": "Best", "key": "score_best", "digits": 1},
    {"column": "Worst", "key": "score_worst", "digits": 1},
    {"column": "Avg", "key": "score_avg", "digits": 2},
    {"column": "Std", "key": "score_std", "digits": 2},
    {"column": "Wall_Time_Avg(s)", "key": "wall_avg_s", "digits": 3},
    {"column": "CPU_Time_Avg(s)", "key": "cpu_avg_s", "digits": 3},
    {"column": "NFE_Avg", "key": "nfe_avg", "digits": 1},
    {"column": "Champion_Score", "key": "champion_score", "digits": 2},
    {"column": "Champion_Features", "key": "champion_features"},
    {"column": "Nb_Features_Keep", "key": "nb_features_keep", "digits": 2},
]

CSV_COLUMNS = [item["column"] for item in CSV_SCHEMA]

def build_run_configs(dataset_ids, cases, evaluation_cases, runs_per_algo):
    run_id = 0
    group_id = 0
    for dataset_id in dataset_ids:
        for case in cases:
            for evaluation_case in evaluation_cases:
                base_seed = case["seed"]
                for i in range(runs_per_algo):
                    config = dict(case)
                    config["seed"] = base_seed ^ (i * 10000001)
                    config["base_seed"] = base_seed
                    config["dataset_id"] = dataset_id
                    config["case_id"] = case["id"]
                    config["evaluation_config"] = dict(evaluation_case)
                    config["evaluation_model"] = evaluation_case["evaluation_model"]
                    config["run_id"] = run_id
                    config["sample_id"] = i
                    config["group_id"] = group_id
                    run_id += 1
                    yield config
                group_id += 1

def case_keys() -> list[str]:
    return [
        "case_id",
        "dataset_id",
        "evaluation_model",
    ]

def format_run_label(config):
    label = ""
    for key in case_keys():
        label += f"{key}={config[key]} "
        return label
        

def aggregate_group(group):
    group_df = pd.DataFrame(group).drop(columns=["run_id", "seed", "decoder_config"])
    score_df = group_df.groupby(["case_id", "dataset_id", "popsize", "max_generations", "decoder_name",
                                 "evaluation_model", "fitness_name", "alpha", "cv_folds"], as_index=False).agg(
        base_seed = ("base_seed", "first"),
        score_best = ("final_score", "max"),
        score_worst = ("final_score", "min"),
        score_avg = ("final_score", "mean"),
        score_std = ("final_score", "std"),
        wall_avg_s = ("elapsed_wall_s", "mean"),
        cpu_avg_s = ("elapsed_cpu_s", "mean"),
        nfe_avg = ("evaluations_count", "mean"),
        best_fitness = ("best_fitness", "max"),
        nb_features_keep = ("nb_features_keep", "mean")
    ).fillna(0)
        
    champion = group_df.loc[group_df["best_fitness"].astype(float).idxmin()].to_dict()
    

    return {
        **score_df.squeeze().to_dict(),
        "champion_score": float(champion["final_score"]),
        "champion_features": champion["nb_features_keep"]
    }

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
            tqdm.write(f"Row saved: case={row['Case_ID']} dataset={row['Dataset_ID']} evaluator={row['Evaluator']}")
            
def run_parallel_configs(configurations, worker, csv_filepath, csv_schema):
    worker_count = os.cpu_count() or 4
    

    
    df = pd.DataFrame(configurations).set_index(case_keys()).sort_index()
    results = { group: [] for group in df.groupby("group_id").groups }
    expected = dict(df.groupby("group_id").size())
    
    
    pbar = tqdm(total=len(configurations), desc="de_strategy", unit="run")
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = { executor.submit(worker, config): config["group_id"] for config in configurations }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="groups"):
            group_id = futures[future]
            results[group_id].append(future.result())
            tqdm.write(f"Run completed: {format_run_label(future.result())} ")
            tqdm.write(f"Group progress: {len(results[group_id])} / {expected[group_id]}")
            
            pbar.update(1)
            if len(results[group_id]) == expected[group_id]:
                group = results.pop(group_id)
                values = aggregate_group(group)
                row = build_row_from_schema(values, CSV_SCHEMA)
                append_csv_rows(csv_filepath, CSV_COLUMNS, [row])
                
    pbar.close()
                
            
def test(config):
    return {
        "case_id": config["case_id"],
        "dataset_id": config["dataset_id"],
        "popsize": config["popsize"],
        "max_generations": config["max_generations"],
        "base_seed": config["seed"],
        "decoder_name": config["decoder"]["name"],
        "evaluation_model": config["evaluation_config"]["evaluation_model"],
        "fitness_name": config["evaluation_config"]["fitness"],
        "alpha": config["evaluation_config"]["alpha"],
        "cv_folds": config["evaluation_config"]["cv_folds"],
        "final_score": random.uniform(0, 1),
        "best_fitness": random.uniform(0, 1),
        "nb_features_keep": random.randint(1, 100),
        "elapsed_wall_s": random.uniform(0, 100),
        "elapsed_cpu_s": random.uniform(0, 100),
        "evaluations_count": random.randint(50, 100)
    }
                    
if __name__ == "__main__":
    
    config = load_config()
    run_parallel_configs(config, test, "app/Results/SAParameters/test.csv", CSV_SCHEMA)
    
    pass
