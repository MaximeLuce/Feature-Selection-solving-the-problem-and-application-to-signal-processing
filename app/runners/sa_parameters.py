# app/runners/sa_parameters.py

import numpy as np

from app.monitoring import MonitoringMetric
from app.core.config import load_config
from app.core.io import save_raw_run_result
from app.simulated_annealing.algorithm import SimulatedAnnealing
from app.problem import Problem
from app.runners.common import (
    build_run_configs,
    run_configs,
)

SA_CSV_SCHEMA = [
    {"column": "Case_ID", "key": "case_id"},
    {"column": "Dataset_ID", "key": "dataset_id"},
    {"column": "Initial_Temp", "key": "initial_temp"},
    {"column": "Cooling_Rate", "key": "cooling_rate"},
    {"column": "Total_Evals", "key": "total_evals"},
    {"column": "Expected_Final_Temp", "key": "expected_final_temperature", "digits": 8},
    {"column": "Final_Temp", "key": "final_temperature", "digits": 8},
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

SA_GROUP_FIELDS = [
    "case_id", "dataset_id", "initial_temp", "cooling_rate", "total_evals",
    "expected_final_temperature", "final_temperature",
    "evaluation_model", "fitness_name", "alpha", "cv_folds",
]
SA_MONITORING_METRICS = [
    MonitoringMetric.ELAPSED_TIME_NS.value,
    MonitoringMetric.CPU_TIME_NS.value,
    MonitoringMetric.EVALUATION_COUNT.value,
    MonitoringMetric.BEST_FITNESS.value,
    MonitoringMetric.BEST_MASK.value,
    MonitoringMetric.CURRENT_MASK.value,
    MonitoringMetric.CURRENT_FITNESS.value,
    MonitoringMetric.TEMPERATURE.value,
    MonitoringMetric.ACCEPTANCE_RATE.value,
    MonitoringMetric.FINAL_TEMPERATURE.value,
]
RAW_DIR = "app/Results/raw"
RAW_FILENAME = "SA_raw_cases_SA_Table10.jsonl"


def run_sa_config(config):
    dataset_id = config["dataset_id"]
    evaluation_config = dict(config["evaluation_config"])
    evaluation_config["seed"] = config["seed"]
    problem = Problem.load_dataset(dataset_id, evaluation_config)

    monitoring = dict(config.get("monitoring", {}))
    monitoring_metrics = list(monitoring.get("metrics", []))
    for metric in SA_MONITORING_METRICS:
        if metric not in monitoring_metrics:
            monitoring_metrics.append(metric)
    monitoring["metrics"] = monitoring_metrics

    sa = SimulatedAnnealing(
        problem,
        max_evaluations=config["total_evals"],
        initial_temp=config["initial_temp"],
        cooling_rate=config["cooling_rate"],
        seed=config["seed"],
        monitoring=monitoring,
    )
    result = sa.run()

    best_mask = np.array(result.best_mask, dtype=int)
    final_score = float(problem.evaluate_final(best_mask) * 100)
    
    if result.wall_ns is None:
        raise ValueError("SA result is missing wall_ns timing data.")
    if result.cpu_ns is None:
        raise ValueError("SA result is missing cpu_ns timing data.")

    # Append raw result to shared JSONL file
    raw_data = {
        "config": {
            "case_id": config["case_id"],
            "dataset_id": dataset_id,
            "initial_temp": config["initial_temp"],
            "cooling_rate": config["cooling_rate"],
            "total_evals": config["total_evals"],
            "expected_final_temperature": result.expected_final_temperature,
            "final_temperature": result.final_temperature,
            "base_seed": config["base_seed"],
            "evaluation_model": evaluation_config["evaluation_model"],
            "fitness_name": evaluation_config["fitness"],
            "alpha": evaluation_config["alpha"],
            "cv_folds": evaluation_config["cv_folds"],
        },
        "summary": {
            "best_fitness": float(result.best_fitness),
            "final_score": final_score,
            "nb_features_keep": int(best_mask.sum()),
            "elapsed_wall_s": float(result.wall_ns / 1_000_000_000),
            "elapsed_cpu_s": float(result.cpu_ns / 1_000_000_000),
            "evaluations_count": result.evaluations,
        },
        "history": result.history,
    }
    save_raw_run_result(f"{RAW_DIR}/{RAW_FILENAME}", raw_data)

    return {
        "run_id": config["run_id"],
        "case_id": config["case_id"],
        "dataset_id": dataset_id,
        "initial_temp": config["initial_temp"],
        "cooling_rate": config["cooling_rate"],
        "total_evals": config["total_evals"],
        "expected_final_temperature": float(result.expected_final_temperature),
        "final_temperature": float(result.final_temperature),
        "base_seed": config["base_seed"],
        "evaluation_model": evaluation_config["evaluation_model"],
        "fitness_name": evaluation_config["fitness"],
        "alpha": evaluation_config["alpha"],
        "cv_folds": evaluation_config["cv_folds"],
        "best_fitness": float(result.best_fitness),
        "final_score": final_score,
        "nb_features_keep": int(best_mask.sum()),
        "elapsed_wall_s": float(result.wall_ns / 1_000_000_000),
        "elapsed_cpu_s": float(result.cpu_ns / 1_000_000_000),
        "evaluations_count": result.evaluations,
    }


class SAParameters:
    def __init__(self):
        config = load_config()
        self.dataset_ids = config.get("dataset_ids", [0])
        self.runs_per_algo = config.get("runs_per_algo", 10)
        self.total_evals = config.get("total_evals", 10000)
        self.cases = config.get("cases_SA_Table10", [])
        self.evaluation_cases = config.get("evaluation_cases", [{
            "evaluation_model": "svm",
            "fitness": "weighted_error",
            "alpha": 0.5,
            "cv_folds": 5,
        }])
        self.csv_filepath = config.get(
            "csv_filepath",
            "app/Results/SAParameters/SAParameters_cases_SA_Table10.csv",
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

        run_configs(
            configurations, run_sa_config, self.csv_filepath, SA_CSV_SCHEMA,
            group_fields=SA_GROUP_FIELDS,
            max_workers=14
        )


if __name__ == "__main__":
    SAParameters().run_all()
