import time

import numpy as np

from app.OptimizationAlgorithm.SimulatedAnnealing import SimulatedAnnealing
from app.Problem.Problem import Problem
from app.Utilities.ConfigLoader import load_config
from app.runners.common import (
    build_run_configs,
    run_parallel_configs,
)

SA_CSV_SCHEMA = [
    {"column": "Case_ID", "key": "case_id"},
    {"column": "Dataset_ID", "key": "dataset_id"},
    {"column": "Initial_Temp", "key": "initial_temp"},
    {"column": "Cooling_Rate", "key": "cooling_rate"},
    {"column": "Total_Evals", "key": "total_evals"},
    {"column": "Evaluator", "key": "evaluation_model"},
    {"column": "Fitness", "key": "fitness_name"},
    {"column": "Alpha", "key": "alpha"},
    {"column": "CV_Folds", "key": "cv_folds"},
    {"column": "SA_Best", "key": "score_best", "digits": 1},
    {"column": "SA_Worst", "key": "score_worst", "digits": 1},
    {"column": "SA_Avg", "key": "score_avg", "digits": 2},
    {"column": "SA_Std", "key": "score_std", "digits": 2},
    {"column": "SA_Wall_Time_Avg(s)", "key": "wall_avg_s", "digits": 3},
    {"column": "SA_CPU_Time_Avg(s)", "key": "cpu_avg_s", "digits": 3},
    {"column": "SA_NFE_Avg", "key": "nfe_avg", "digits": 1},
    {"column": "Champion_Score", "key": "champion_score", "digits": 2},
    {"column": "Champion_Features", "key": "champion_features"},
    {"column": "Nb_Features_Keep", "key": "nb_features_keep", "digits": 2},
]


def run_sa_config(config):
    dataset_id = config["dataset_id"]
    evaluation_config = dict(config["evaluation_config"])
    problem = Problem.load_dataset(dataset_id, evaluation_config)

    sa = SimulatedAnnealing(problem, config["total_evals"], config["initial_temp"], config["cooling_rate"])

    wall_start = time.perf_counter_ns()
    cpu_start = time.thread_time_ns()
    best_ind = sa.run()
    wall_ns = time.perf_counter_ns() - wall_start
    cpu_ns = time.thread_time_ns() - cpu_start

    best_mask = np.array(best_ind.features_mask, dtype=int)
    final_score = float(problem.evaluate_final(best_mask) * 100)

    return {
        "run_id": config["run_id"],
        "case_id": config["case_id"],
        "dataset_id": dataset_id,
        "initial_temp": config["initial_temp"],
        "cooling_rate": config["cooling_rate"],
        "total_evals": config["total_evals"],
        "base_seed": config["base_seed"],
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
        self.evaluation_cases = config.get("evaluation_cases", [{
            "evaluation_model": "svm",
            "fitness": "weighted_error",
            "alpha": 0.5,
            "cv_folds": 5,
        }])
        self.csv_filepath = config.get(
            "csv_filepath",
            "app/Results/SAParameters/SAParameters_Best_Initial_Temperature.csv",
        )

    def run_all(self):
        configurations = list(build_run_configs(
            self.dataset_ids,
            self.cases,
            self.evaluation_cases,
            self.runs_per_algo,
        ))
        for c in configurations:
            c["total_evals"] = self.total_evals

        run_parallel_configs(
            configurations, run_sa_config, self.csv_filepath, SA_CSV_SCHEMA,
            group_columns=[
                "case_id", "dataset_id", "initial_temp", "cooling_rate",
                "evaluation_model", "fitness_name", "alpha", "cv_folds",
            ],
        )


if __name__ == "__main__":
    SAParameters().run_all()
