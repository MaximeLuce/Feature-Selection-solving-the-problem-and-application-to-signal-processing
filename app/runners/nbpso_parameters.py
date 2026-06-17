# app/runners/nbpso_parameters.py

import os
import numpy as np

from app.Archive.monitoring import MonitoringMetric
from app.Problem.Problem import Problem
from app.utilities.config_loader import load_config
from app.particle_swarm.algorithm import NovelBinaryParticleSwarmOptimization
from app.runners.common import (
    build_rebuilt_csv_path,
    build_run_configs,
    rebuild_grouped_csv_from_raw,
    run_configs,
    save_raw_run_result,
)

NBPSO_CSV_SCHEMA = [
    {"column": "Case_ID", "key": "case_id"},
    {"column": "Dataset_ID", "key": "dataset_id"},
    {"column": "Swarm_Size", "key": "swarm_size"},
    {"column": "W", "key": "w", "digits": 3},
    {"column": "W_Max", "key": "w_max", "digits": 3},
    {"column": "W_Min", "key": "w_min", "digits": 3},
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
    "case_id", "dataset_id", "swarm_size", "w", "w_max", "w_min", "c1", "c2", "vmax",
    "max_generations", "evaluation_model", "fitness_name", "alpha", "cv_folds",
]

TIMING_METRICS = [
    MonitoringMetric.ELAPSED_TIME_NS.value,
    MonitoringMetric.CPU_TIME_NS.value,
]
RAW_MONITORING_METRICS = [
    MonitoringMetric.EVALUATION_COUNT.value,
    MonitoringMetric.BEST_FITNESS.value,
    MonitoringMetric.BEST_MASK.value,
    MonitoringMetric.WORST_MASK.value,
    MonitoringMetric.POPULATION_DIVERSITY.value,
]
RAW_DIR = "app/Results/raw"
RAW_FILENAME = "cases_NBPSO_Table14_LDIW.jsonl"

def run_nbpso_config(config):
    print(
        "Run starting: "
        f"pid={os.getpid()} "
        f"case={config['case_id']} "
        f"dataset={config['dataset_id']} "
        f"evaluator={config['evaluation_model']} "
        f"sample={config.get('sample_id', '?')} "
        f"swarm_size={config['swarm_size']} "
        f"generations={config['max_generations']}",
        flush=True,
    )
    dataset_id = config["dataset_id"]
    evaluation_config = dict(config["evaluation_config"])
    problem = Problem.load_dataset(dataset_id, evaluation_config)

    nbpso_config = dict(config)
    monitoring = dict(nbpso_config.get("monitoring", {}))
    monitoring_metrics = list(monitoring.get("metrics", []))
    for metric in TIMING_METRICS + RAW_MONITORING_METRICS:
        if metric not in monitoring_metrics:
            monitoring_metrics.append(metric)
    monitoring["metrics"] = monitoring_metrics
    nbpso_config["monitoring"] = monitoring

    nbpso = NovelBinaryParticleSwarmOptimization(problem, nbpso_config)
    result = nbpso.run()

    if result.wall_ns is None:
        raise ValueError("NBPSO result is missing wall_ns timing data.")
    if result.cpu_ns is None:
        raise ValueError("NBPSO result is missing cpu_ns timing data.")

    best_mask = np.asarray(result.best_mask, dtype=int)
    final_score = float(problem.evaluate_final(best_mask) * 100)
    w = config["w"]
    w_max = 0.0
    w_min = 0.0
    if isinstance(w, dict):
        w_max = w.get("w_max", 0.9)
        w_min = w.get("w_min", 0.4)
        w = 0.0
        
        

    raw_data = {
        "config": {
            "case_id": config["case_id"],
            "dataset_id": dataset_id,
            "swarm_size": config["swarm_size"],
            "w": w,
            "w_max": w_max,
            "w_min": w_min,
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
            "evaluations_count": result.evaluations,
        },
        "history": result.history,
    }
    save_raw_run_result(RAW_DIR, raw_data, RAW_FILENAME)

    return {
        "run_id": config["run_id"],
        "case_id": config["case_id"],
        "dataset_id": dataset_id,
        "swarm_size": config["swarm_size"],
        "w": w,
        "w_max": w_max,
        "w_min": w_min,
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
        "evaluations_count": result.evaluations,
    }


class NBPSOParameters:
    def __init__(self):
        self.config = load_config()
        self.dataset_ids = self.config.get("dataset_ids", [0])
        self.runs_per_algo = self.config.get("runs_per_algo", 10)
        self.max_workers = self.config.get(
            "nbpso_max_workers",
            max(1, min(14, os.cpu_count() or 14)),
        )
        self.cases = self.config.get("cases_NBPSO_Table14_LDIW", [])
        self.evaluation_cases = self.config.get("evaluation_cases", [{
            "evaluation_model": "svm",
            "fitness": "weighted_error",
            "alpha": 0.5,
            "cv_folds": 5,
        }])
        self.csv_filepath = "app/Results/SAParameters/NBPSO_Table14_LDIW.csv"
        
        self.configurations = list(build_run_configs(
            self.dataset_ids,
            self.cases,
            self.evaluation_cases,
            self.runs_per_algo,
        ))

    def run_all(self, parallel=True):
        run_configs(
            self.configurations, run_nbpso_config, self.csv_filepath, NBPSO_CSV_SCHEMA,
            group_fields=NBPSO_GROUP_FIELDS,
            parallel=parallel,
            max_workers=self.max_workers,
        )

    def rebuild_summary_from_raw(self, output_csv_filepath=None):
        raw_filepath = os.path.join(RAW_DIR, RAW_FILENAME)
        rebuilt_csv_filepath = output_csv_filepath or build_rebuilt_csv_path(self.csv_filepath)
        return rebuild_grouped_csv_from_raw(
            raw_filepath=raw_filepath,
            csv_filepath=self.csv_filepath,
            csv_schema=NBPSO_CSV_SCHEMA,
            group_fields=NBPSO_GROUP_FIELDS,
            output_csv_filepath=rebuilt_csv_filepath,
        )


if __name__ == "__main__":
    runner = NBPSOParameters()
    runner.run_all(parallel=True)
