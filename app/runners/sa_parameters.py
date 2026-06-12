import random
import time

import numpy as np

from app.OptimizationAlgorithm.SimulatedAnnealing import SimulatedAnnealing
from app.Problem.Problem import Problem
from app.Utilities.ConfigLoader import load_config
from app.runners.common import (
    aggregate_feature_selection_group,
    aggregate_grouped_results,
    append_csv_rows,
    build_evaluation_cases,
    build_row_from_schema,
    build_run_configs,
    print_feature_selection_summary,
    run_parallel_configs,
    validate_and_order_row,
)

SA_CSV_SCHEMA = [
    {"column": "Case_ID", "key": "case_id"},
    {"column": "Dataset_ID", "key": "dataset_id"},
    {"column": "Initial_Temp", "key": "initial_temp"},
    {"column": "Cooling_Rate", "key": "cooling_rate"},
    {"column": "Total_Evals", "key": "total_evals"},
    {"column": "Evaluator", "key": "evaluator"},
    {"column": "Fitness", "key": "fitness_name"},
    {"column": "Alpha", "key": "alpha"},
    {"column": "CV_Folds", "key": "cv_folds"},
    {"column": "SA_Best", "key": "sa_b", "digits": 1},
    {"column": "SA_Worst", "key": "sa_w", "digits": 1},
    {"column": "SA_Avg", "key": "sa_a", "digits": 2},
    {"column": "SA_Std", "key": "sa_s", "digits": 2},
    {"column": "SA_Wall_Time_Avg(s)", "key": "wall_avg_s", "digits": 3},
    {"column": "SA_CPU_Time_Avg(s)", "key": "cpu_avg_s", "digits": 3},
    {"column": "SA_NFE_Avg", "key": "sa_nfe_avg", "digits": 1},
    {"column": "Champion_Score", "key": "champion_score", "digits": 2},
    {"column": "Nb_Features_Keep", "key": "nb_features_keep"},
]

SA_CSV_COLUMNS = [item["column"] for item in SA_CSV_SCHEMA]


def run_sa_config(config):
    dataset_id = config["dataset_id"]
    evaluation_config = dict(config["evaluation_config"])
    problem = Problem.load_dataset(dataset_id, evaluation_config)

    initial_temp = config["initial_temp"]
    cooling_rate = config["cooling_rate"]
    total_evals = config["total_evals"]

    sa = SimulatedAnnealing(problem, total_evals, initial_temp, cooling_rate)

    wall_start = time.perf_counter_ns()
    cpu_start = time.thread_time_ns()
    best_ind = sa.run()
    wall_ns = time.perf_counter_ns() - wall_start
    cpu_ns = time.thread_time_ns() - cpu_start

    best_mask = np.array(best_ind.features_mask, dtype=int)
    final_score = float(problem.evaluate_final(best_mask) * 100)

    return {
        "order": config["order"],
        "run_id": config["run_id"],
        "case_id": config["case_id"],
        "dataset_id": dataset_id,
        "initial_temp": initial_temp,
        "cooling_rate": cooling_rate,
        "total_evals": total_evals,
        "sample_index": config["sample_index"],
        "evaluation_model": evaluation_config["evaluation_model"],
        "fitness_name": evaluation_config["fitness"],
        "alpha": evaluation_config["alpha"],
        "cv_folds": evaluation_config["cv_folds"],
        "best_fitness": float(best_ind.fitness),
        "final_score": final_score,
        "nb_features_keep": int(best_mask.sum()),
        "elapsed_wall_s": float(wall_ns / 1_000_000_000),
        "elapsed_cpu_s": float(cpu_ns / 1_000_000_000),
        "evaluations_count": problem.evaluations_count,
    }


class SAParameters:
    def __init__(self):
        config = load_config()
        self.dataset_ids = config.get("dataset_ids", [0])
        self.runs_per_algo = config.get("runs_per_algo", 10)
        self.total_evals = config.get("total_evals", 10000)
        self.cases = config.get("cases_SA_initial_temp", [])
        self.evaluation_cases = build_evaluation_cases(config)
        self.csv_filepath = config.get(
            "csv_filepath",
            "app/Results/SAParameters/SAParameters_Best_Initial_Temperature.csv",
        )

        if not self.cases:
            print("No SA cases config has been loaded.")

    def _build_csv_row(self, values):
        row = build_row_from_schema(values, SA_CSV_SCHEMA)
        return validate_and_order_row(row, SA_CSV_COLUMNS)

    def _print_summary(self, row):
        print_feature_selection_summary(
            row,
            ("SA_Best", "SA_Worst", "SA_Avg", "SA_Std"),
            [
                ("Dataset", "Dataset_ID"),
                ("Case", "Case_ID"),
                ("Evaluator", "Evaluator"),
                ("Fitness", "Fitness"),
                ("alpha=", "Alpha"),
                ("cv=", "CV_Folds"),
            ],
        )

    def _aggregate_group(self, _group_key, group):
        stats = aggregate_feature_selection_group(group)
        first = stats["first"]

        return self._build_csv_row({
            "case_id": first["case_id"],
            "dataset_id": stats["dataset_id"],
            "initial_temp": first["initial_temp"],
            "cooling_rate": first["cooling_rate"],
            "total_evals": int(first["total_evals"]),
            "evaluator": first["evaluation_model"],
            "fitness_name": first["fitness_name"],
            "alpha": first["alpha"],
            "cv_folds": int(first["cv_folds"]),
            "sa_b": stats["score_best"],
            "sa_w": stats["score_worst"],
            "sa_a": stats["score_avg"],
            "sa_s": stats["score_std"],
            "wall_avg_s": stats["wall_avg_s"],
            "cpu_avg_s": stats["cpu_avg_s"],
            "sa_nfe_avg": stats["nfe_avg"],
            "champion_score": stats["champion_score"],
            "nb_features_keep": stats["nb_features_keep"],
        })

    def run_all(self):
        print(f"Start running... Results will be saved to {self.csv_filepath}")

        configurations = list(build_run_configs(
            self.dataset_ids,
            self.cases,
            self.evaluation_cases,
            self.runs_per_algo,
            lambda case: {
                **dict(case),
                "initial_temp": case["initial_temp"],
                "cooling_rate": case["cooling_rate"],
                "total_evals": self.total_evals,
            },
        ))
        if not configurations:
            return

        random.shuffle(configurations)
        results = run_parallel_configs(
            configurations,
            run_sa_config,
            description="SA runs",
        )

        rows = aggregate_grouped_results(
            results,
            lambda result: (
                result["case_id"],
                str(result["dataset_id"]),
                result["evaluation_model"],
                result["fitness_name"],
                result["alpha"],
                result["cv_folds"],
            ),
            self._aggregate_group,
        )
        for row in rows:
            self._print_summary(row)
        append_csv_rows(self.csv_filepath, SA_CSV_COLUMNS, rows)


if __name__ == "__main__":
    SAParameters().run_all()
