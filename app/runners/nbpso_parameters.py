import numpy as np

from app.monitoring import MonitoringMetric
from app.Problem.Problem import Problem
from app.Utilities.ConfigLoader import load_config
from app.particle_swarm.algorithm import NewBinaryParticleSwarmOptimization
from app.runners.common import (
    build_run_configs,
    run_parallel_configs,
    save_raw_run_result,
)

NBPSO_CSV_SCHEMA = [
    {"column": "Case_ID", "key": "case_id"},
    {"column": "Dataset_ID", "key": "dataset_id"},
    {"column": "Swarm_Size", "key": "swarm_size"},
    {"column": "W", "key": "w", "digits": 3},
    {"column": "C1", "key": "c1", "digits": 3},
    {"column": "C2", "key": "c2", "digits": 3},
    {"column": "Vmax", "key": "vmax", "digits": 3},
    {"column": "Generations_number", "key": "max_generations"},
    {"column": "Seed", "key": "base_seed"},
    {"column": "Evaluator", "key": "evaluation_model"},
    {"column": "Fitness", "key": "fitness_name"},
    {"column": "Alpha", "key": "alpha"},
    {"column": "CV_Folds", "key": "cv_folds"},
    {"column": "NBPSO_Best", "key": "score_best", "digits": 1},
    {"column": "NBPSO_Worst", "key": "score_worst", "digits": 1},
    {"column": "NBPSO_Avg", "key": "score_avg", "digits": 2},
    {"column": "NBPSO_Std", "key": "score_std", "digits": 2},
    {"column": "NBPSO_Wall_Time_Avg(s)", "key": "wall_avg_s", "digits": 3},
    {"column": "NBPSO_CPU_Time_Avg(s)", "key": "cpu_avg_s", "digits": 3},
    {"column": "NBPSO_NFE_Avg", "key": "nfe_avg", "digits": 1},
    {"column": "Champion_Score", "key": "champion_score", "digits": 2},
    {"column": "Champion_Features", "key": "champion_features"},
    {"column": "Nb_Features_Keep", "key": "nb_features_keep", "digits": 2},
]

NBPSO_GROUP_FIELDS = [
    "case_id", "dataset_id", "swarm_size", "w", "c1", "c2", "vmax",
    "max_generations", "evaluation_model", "fitness_name", "alpha", "cv_folds",
]
NBPSO_RESUME_FIELDS = list(NBPSO_GROUP_FIELDS)
NBPSO_MONITORING_METRICS = [
    MonitoringMetric.ELAPSED_TIME_NS.value,
    MonitoringMetric.CPU_TIME_NS.value,
    MonitoringMetric.EVALUATION_COUNT.value,
    MonitoringMetric.BEST_FITNESS.value,
    MonitoringMetric.BEST_MASK.value,
    MonitoringMetric.WORST_MASK.value,
    MonitoringMetric.POPULATION_DIVERSITY.value,
]


def run_nbpso_config(config):
    dataset_id = config["dataset_id"]
    evaluation_config = dict(config["evaluation_config"])
    problem = Problem.load_dataset(dataset_id, evaluation_config)

    nbpso_config = dict(config)
    monitoring = dict(nbpso_config.get("monitoring", {}))
    monitoring_metrics = list(monitoring.get("metrics", []))
    for metric in NBPSO_MONITORING_METRICS:
        if metric not in monitoring_metrics:
            monitoring_metrics.append(metric)
    monitoring["metrics"] = monitoring_metrics
    nbpso_config["monitoring"] = monitoring

    nbpso = NewBinaryParticleSwarmOptimization(problem, nbpso_config)
    result = nbpso.run()

    if result.wall_ns is None:
        raise ValueError("NBPSO result is missing wall_ns timing data.")
    if result.cpu_ns is None:
        raise ValueError("NBPSO result is missing cpu_ns timing data.")

    best_mask = np.asarray(result.best_mask, dtype=int)
    final_score = float(problem.evaluate_final(best_mask) * 100)

    raw_dir = "app/Results/raw"
    raw_filename = "NBPSO_raw.jsonl"
    raw_data = {
        "config": {
            "case_id": config["case_id"],
            "dataset_id": dataset_id,
            "swarm_size": config["swarm_size"],
            "w": config["w"],
            "c1": config["c1"],
            "c2": config["c2"],
            "vmax": config.get("vmax", 4.0),
            "max_generations": config["max_generations"],
            "base_seed": config["base_seed"],
            "seed": config["seed"],
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
            "evaluations_count": problem.evaluations_count,
        },
        "history": result.history,
    }
    save_raw_run_result(raw_dir, raw_data, raw_filename)

    return {
        "run_id": config["run_id"],
        "case_id": config["case_id"],
        "dataset_id": dataset_id,
        "swarm_size": config["swarm_size"],
        "w": config["w"],
        "c1": config["c1"],
        "c2": config["c2"],
        "vmax": config.get("vmax", 4.0),
        "max_generations": config["max_generations"],
        "base_seed": config["base_seed"],
        "seed": config["seed"],
        "evaluation_model": evaluation_config["evaluation_model"],
        "fitness_name": evaluation_config["fitness"],
        "alpha": evaluation_config["alpha"],
        "cv_folds": evaluation_config["cv_folds"],
        "best_fitness": float(result.best_fitness),
        "final_score": final_score,
        "nb_features_keep": int(best_mask.sum()),
        "elapsed_wall_s": float(result.wall_ns / 1_000_000_000),
        "elapsed_cpu_s": float(result.cpu_ns / 1_000_000_000),
        "evaluations_count": problem.evaluations_count,
    }


class NBPSOParameters:
    def __init__(self):
        config = load_config()
        self.dataset_ids = config.get("dataset_ids", [0])
        self.runs_per_algo = config.get("runs_per_algo", 10)
        self.cases = config.get("cases_NBPSO", [])
        self.evaluation_cases = config.get("evaluation_cases", [{
            "evaluation_model": "svm",
            "fitness": "weighted_error",
            "alpha": 0.5,
            "cv_folds": 5,
        }])
        self.csv_filepath = config.get(
            "csv_filepath_nbpso",
            "app/Results/SAParameters/NBPSOParameters_v2.csv",
        )

    def run_all(self):
        configurations = list(build_run_configs(
            self.dataset_ids,
            self.cases,
            self.evaluation_cases,
            self.runs_per_algo,
        ))

        run_parallel_configs(
            configurations, run_nbpso_config, self.csv_filepath, NBPSO_CSV_SCHEMA,
            group_fields=NBPSO_GROUP_FIELDS,
            resume_fields=NBPSO_RESUME_FIELDS,
        )


if __name__ == "__main__":
    NBPSOParameters().run_all()
