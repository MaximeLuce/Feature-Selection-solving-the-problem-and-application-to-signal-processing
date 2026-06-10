import os
import random
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from app.decoders import build_decoder
from app.differential_evolution.algorithm import DifferentialEvolution
from app.differential_evolution.featureselection import FeatureSelectionProblem
from app.MLModels.HeavyModelSVM import HeavyModelSVM
from app.Problem.Problem import Problem
from app.Utilities.ConfigLoader import load_config
from app.runners.common import (
    append_csv_rows,
    build_row_from_schema,
    calc_stats,
    validate_and_order_row,
)

TIMING_METRICS = ["elapsed_time_ns", "cpu_time_ns"]

DE_CSV_SCHEMA = [
    {"column": "Case_ID", "key": "case_id"},
    {"column": "Dataset_ID", "key": "dataset_id"},
    {"column": "Population_size", "key": "popsize"},
    {"column": "Generations_number", "key": "max_generations"},
    {"column": "Seed", "key": "base_seed"},
    {"column": "Decoder", "key": "decoder_name"},
    {"column": "Problem_Evaluator", "key": "problem_evaluator"},
    {"column": "Fitness", "key": "fitness_name"},
    {"column": "Alpha", "key": "alpha"},
    {"column": "CV_Folds", "key": "cv_folds"},
    {"column": "Score_Evaluator", "key": "score_evaluator"},
    {"column": "DE_Best", "key": "de_b", "digits": 1},
    {"column": "DE_Worst", "key": "de_w", "digits": 1},
    {"column": "DE_Avg", "key": "de_a", "digits": 2},
    {"column": "DE_Std", "key": "de_s", "digits": 2},
    {"column": "DE_Wall_Time_Avg(s)", "key": "wall_avg_s", "digits": 3},
    {"column": "DE_CPU_Time_Avg(s)", "key": "cpu_avg_s", "digits": 3},
    {"column": "DE_NFE_Avg", "key": "de_nfe_avg", "digits": 1},
    {"column": "Champion_Score", "key": "champion_score", "digits": 2},
    {"column": "Nb_Features_Keep", "key": "nb_features_keep"},
]

DE_CSV_COLUMNS = [item["column"] for item in DE_CSV_SCHEMA]


def run_de_config(config):
    print(
        f"\n--- RUNNING CASE {config['case_id']} / "
        f"{config['evaluation_config']['evaluation_model']} ---"
    )

    dataset_id = config["dataset_id"]
    evaluation_config = config["evaluation_config"]
    problem = Problem.load_dataset(dataset_id, evaluation_config)
    print(
        f"Problem loaded: ID={dataset_id} with {problem.num_features} "
        f"features and {problem.num_instances} instances."
    )

    decoder = build_decoder(config["decoder"])
    de_problem = FeatureSelectionProblem(problem, decoder)

    de_config = dict(config)
    monitoring = dict(de_config.get("monitoring", {}))
    monitoring_metrics = list(monitoring.get("metrics", []))
    for metric in TIMING_METRICS:
        if metric not in monitoring_metrics:
            monitoring_metrics.append(metric)
    monitoring["metrics"] = monitoring_metrics
    de_config["monitoring"] = monitoring

    de = DifferentialEvolution(de_problem, de_config)
    result = de.run()

    if result.wall_ns is None:
        raise ValueError("DE result is missing wall_ns timing data.")
    if result.cpu_ns is None:
        raise ValueError("DE result is missing cpu_ns timing data.")

    return {
        "order": config["order"],
        "run_id": config["run_id"],
        "case_id": config["case_id"],
        "dataset_id": dataset_id,
        "popsize": config["popsize"],
        "max_generations": config["max_generations"],
        "base_seed": config["base_seed"],
        "seed": config["seed"],
        "decoder_name": decoder.name,
        "decoder_config": config["decoder"],
        "evaluation_model": evaluation_config["evaluation_model"],
        "fitness_name": evaluation_config["fitness"],
        "alpha": evaluation_config["alpha"],
        "cv_folds": evaluation_config["cv_folds"],
        "best_vector": np.asarray(result.best, dtype=float),
        "best_fitness": float(result.best_fitness),
        "elapsed_wall_s": float(result.wall_ns / 1_000_000_000),
        "elapsed_cpu_s": float(result.cpu_ns / 1_000_000_000),
        "evaluations_count": de_problem.evaluations_count,
    }


def run_parallel_configs(configurations, worker=run_de_config, max_workers=None):
    results = []
    worker_count = max_workers or (os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(worker, config): config["run_id"]
            for config in configurations
        }

        for future in as_completed(futures):
            run_id = futures[future]
            results.append(future.result())
            print(f"Run {run_id} done: {len(results)}/{len(configurations)}")

    return results


def score_problem_evaluator(problem, feature_mask):
    selected_indices = [i for i, bit in enumerate(feature_mask) if bit == 1]
    if not selected_indices:
        return 0.0
    x_subset = problem.X.iloc[:, selected_indices]
    return float(problem.evaluator.evaluate(x_subset, problem.y_values) * 100)


class DEParametersParallell:
    def __init__(self):
        config = load_config()
        self.dataset_ids = config.get("dataset_ids", [0])
        self.runs_per_algo = config.get("runs_per_algo", 10)
        self.cases = config.get("cases_DE_strategy", [])
        self.evaluation_cases = self._build_evaluation_cases(config)
        self.csv_filepath = config.get(
            "csv_filepath",
            "app/Results/SAParameters/DEParameters_strategy.csv",
        )

        if not self.cases:
            print("No DE cases config has been loaded.")

    def _build_evaluation_cases(self, config):
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

    def _build_de_config(self, case):
        return {
            "popsize": case["popsize"],
            "max_generations": case["max_generations"],
            "seed": case["seed"],
            "strategy": case.get("strategy", "rand/1"),
            "F1": case.get("F1", 0.5),
            "F2": case.get("F2", 0.5),
            "CR": case.get("CR", 0.7),
        }

    def _build_runs(self):
        run_id = 0
        order = 0
        for dataset_id in self.dataset_ids:
            for case in self.cases:
                for evaluation_case in self.evaluation_cases:
                    base_seed = case["seed"]
                    for i in range(self.runs_per_algo):
                        config = self._build_de_config(case)
                        config["seed"] = base_seed ^ (i * 10000001)
                        config["base_seed"] = base_seed
                        config["dataset_id"] = dataset_id
                        config["case_id"] = case["id"]
                        config["decoder"] = case["decoder"]
                        config["evaluation_config"] = dict(evaluation_case)
                        config["order"] = order
                        config["run_id"] = run_id
                        run_id += 1
                        yield config
                    order += 1

    def _build_csv_row(self, values):
        row = build_row_from_schema(values, DE_CSV_SCHEMA)
        return self._check_csv_row(row)

    def _check_csv_row(self, row):
        return validate_and_order_row(row, DE_CSV_COLUMNS)

    def _print_summary(self, row):
        print("-" * 135)
        print(
            f"Dataset {row['Dataset_ID']} | Case {row['Case_ID']} | Decoder {row['Decoder']} | "
            f"Problem evaluator {row['Problem_Evaluator']} | Score evaluator {row['Score_Evaluator']} | "
            f"Fitness {row['Fitness']} | alpha={row['Alpha']} | cv={row['CV_Folds']}"
        )
        print(
            f"Champion score: {row['Champion_Score']:.2f}% "
            f"({row['Nb_Features_Keep']} features)"
        )
        print("-" * 135)

        score_str = (
            f"{row['DE_Best']:>5.0f} {row['DE_Worst']:>6.0f} "
            f"{row['DE_Avg']:>6.1f} {row['DE_Std']:>5.1f}"
        )
        print(f"{'Score':<15} | {score_str:<26}")
        print(
            f"{'Time Avg':<15} | wall {row['DE_Wall_Time_Avg(s)']:<7.3f}s | "
            f"cpu {row['DE_CPU_Time_Avg(s)']:<7.3f}s"
        )
        print(f"{'NFE Avg':<15} | {row['DE_NFE_Avg']:<26.1f} |")
        print("-" * 135)

    def _aggregate_results(self, results):
        if not results:
            return []

        sorted_results = sorted(results, key=lambda item: (item["order"], item["run_id"]))
        grouped = {}

        for result in sorted_results:
            key = (
                result["order"],
                result["dataset_id"],
                result["case_id"],
                result["decoder_name"],
                result["evaluation_model"],
                result["fitness_name"],
                result["alpha"],
                result["cv_folds"],
            )
            grouped.setdefault(key, []).append(result)

        rows = []

        for group in grouped.values():
            first = group[0]
            dataset_id = int(first["dataset_id"])
            evaluation_config = {
                "evaluation_model": first["evaluation_model"],
                "fitness": first["fitness_name"],
                "alpha": first["alpha"],
                "cv_folds": first["cv_folds"],
            }

            problem = Problem.load_dataset(dataset_id, evaluation_config)
            heavy_model = HeavyModelSVM(dataset_id)
            decoder = build_decoder(first["decoder_config"])
            de_problem = FeatureSelectionProblem(problem, decoder)

            de_results_problem = []
            de_results_heavy = []
            de_evals = []
            wall_times = []
            cpu_times = []
            best_overall_fitness = float("inf")
            best_overall_mask = np.ones(problem.num_features, dtype=int)

            for run in group:
                best_vector = np.asarray(run["best_vector"], dtype=float)
                best_mask = de_problem.decode(best_vector)

                score_problem = score_problem_evaluator(problem, best_mask)
                score_heavy = heavy_model.evaluate(best_mask) * 100

                de_results_problem.append(score_problem)
                de_results_heavy.append(score_heavy)
                de_evals.append(float(run["evaluations_count"]))
                wall_times.append(float(run["elapsed_wall_s"]))
                if run["elapsed_cpu_s"] is not None:
                    cpu_times.append(float(run["elapsed_cpu_s"]))

                if float(run["best_fitness"]) < best_overall_fitness:
                    best_overall_fitness = float(run["best_fitness"])
                    best_overall_mask = best_mask.copy()

            de_b_problem, de_w_problem, de_a_problem, de_s_problem = calc_stats(de_results_problem)
            de_b_heavy, de_w_heavy, de_a_heavy, de_s_heavy = calc_stats(
                de_results_heavy
            )
            wall_avg_s = statistics.mean(wall_times)
            cpu_avg_s = statistics.mean(cpu_times) if cpu_times else 0.0
            de_nfe_avg = statistics.mean(de_evals)
            champion_problem_score = score_problem_evaluator(problem, best_overall_mask)
            champion_heavy_score = heavy_model.evaluate(best_overall_mask) * 100
            nb_features_keep = int(best_overall_mask.sum())

            problem_row = self._build_csv_row(
                {
                    "case_id": first["case_id"],
                    "dataset_id": dataset_id,
                    "popsize": int(first["popsize"]),
                    "max_generations": int(first["max_generations"]),
                    "base_seed": int(first["base_seed"]),
                    "decoder_name": first["decoder_name"],
                    "problem_evaluator": first["evaluation_model"],
                    "fitness_name": first["fitness_name"],
                    "alpha": first["alpha"],
                    "cv_folds": int(first["cv_folds"]),
                    "score_evaluator": first["evaluation_model"],
                    "de_b": de_b_problem,
                    "de_w": de_w_problem,
                    "de_a": de_a_problem,
                    "de_s": de_s_problem,
                    "wall_avg_s": wall_avg_s,
                    "cpu_avg_s": cpu_avg_s,
                    "de_nfe_avg": de_nfe_avg,
                    "champion_score": champion_problem_score,
                    "nb_features_keep": nb_features_keep,
                }
            )
            rows.append(problem_row)

            heavy_row = self._build_csv_row(
                {
                    "case_id": first["case_id"],
                    "dataset_id": dataset_id,
                    "popsize": int(first["popsize"]),
                    "max_generations": int(first["max_generations"]),
                    "base_seed": int(first["base_seed"]),
                    "decoder_name": first["decoder_name"],
                    "problem_evaluator": first["evaluation_model"],
                    "fitness_name": first["fitness_name"],
                    "alpha": first["alpha"],
                    "cv_folds": int(first["cv_folds"]),
                    "score_evaluator": "heavy_svm",
                    "de_b": de_b_heavy,
                    "de_w": de_w_heavy,
                    "de_a": de_a_heavy,
                    "de_s": de_s_heavy,
                    "wall_avg_s": wall_avg_s,
                    "cpu_avg_s": cpu_avg_s,
                    "de_nfe_avg": de_nfe_avg,
                    "champion_score": champion_heavy_score,
                    "nb_features_keep": nb_features_keep,
                }
            )
            rows.append(heavy_row)

        return rows

    def _write_csv(self, rows):
        append_csv_rows(self.csv_filepath, DE_CSV_COLUMNS, rows)

    def run_all(self):
        print(f"Start running... Results will be saved to {self.csv_filepath}")

        configurations = list(self._build_runs())
        if not configurations:
            return

        random.shuffle(configurations)
        results = run_parallel_configs(configurations)

        rows = self._aggregate_results(results)
        for row in rows:
            self._print_summary(row)
        self._write_csv(rows)


if __name__ == "__main__":
    DEParametersParallell().run_all()
