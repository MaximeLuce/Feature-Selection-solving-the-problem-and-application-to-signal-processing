import os

for env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(env_var, "1")

import numpy as np

from app.decoders import build_decoder
from app.differential_evolution.algorithm import DifferentialEvolution
from app.differential_evolution.featureselection import FeatureSelectionProblem
from app.Problem.Problem import Problem
from app.Utilities.ConfigLoader import load_config
from app.runners.common import (
    build_run_configs,
    run_parallel_configs,
    CSV_SCHEMA,
)

TIMING_METRICS = ["elapsed_time_ns", "cpu_time_ns"]

def run_de_config(config):
    print(
        "Run starting: "
        f"pid={os.getpid()} "
        f"case={config['case_id']} "
        f"dataset={config['dataset_id']} "
        f"evaluator={config['evaluation_model']} "
        f"sample={config.get('sample_id', '?')} "
        f"strategy={config['strategy']} "
        f"popsize={config['popsize']} "
        f"generations={config['max_generations']}",
        flush=True,
    )
    dataset_id = config["dataset_id"]
    evaluation_config = dict(config["evaluation_config"])
    problem = Problem.load_dataset(dataset_id, evaluation_config)

    decoder_config = dict(config["decoder"])
    decoder_config["seed"] = config["seed"]
    decoder = build_decoder(decoder_config)
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

    best_mask = de_problem.decode(np.asarray(result.best, dtype=float))
    final_score = float(problem.evaluate_final(best_mask) * 100)

    return {
        "run_id": config["run_id"],
        "case_id": config["case_id"],
        "dataset_id": dataset_id,
        "popsize": config["popsize"],
        "strategy": config["strategy"],
        "max_generations": config["max_generations"],
        "base_seed": config["base_seed"],
        "seed": config["seed"],
        "decoder_name": decoder.name,
        "decoder_config": decoder_config,
        "evaluation_model": evaluation_config["evaluation_model"],
        "fitness_name": evaluation_config["fitness"],
        "alpha": evaluation_config["alpha"],
        "cv_folds": evaluation_config["cv_folds"],
        "best_vector": np.asarray(result.best, dtype=float),
        "best_fitness": float(result.best_fitness),
        "final_score": final_score,
        "nb_features_keep": int(best_mask.sum()),
        "elapsed_wall_s": float(result.wall_ns / 1_000_000_000),
        "elapsed_cpu_s": float(result.cpu_ns / 1_000_000_000),
        "evaluations_count": de_problem.evaluations_count,
    }

class DEParameters:
    def __init__(self):
        self.config = load_config()
        self.dataset_ids = self.config.get("dataset_ids", [0])
        self.runs_per_algo = self.config.get("runs_per_algo", 10)
        self.max_workers = self.config.get(
            "de_max_workers",
            max(1, min(4, os.cpu_count() or 4)),
        )
        self.cases = self.config.get("cases_DE_strategy", [])
        self.evaluation_cases = self.config.get("evaluation_cases", [{
            "evaluation_model": "svm",
            "fitness": "weighted_error",
            "alpha": 0.5,
            "cv_folds": 5,
        }])
    
        self.configurations = list(build_run_configs(self.dataset_ids, self.cases, self.evaluation_cases, self.runs_per_algo))
        self.csv_filepath = "app/Results/SAParameters/DEParameters4_test.csv"
    
    def run_all(self):
        run_parallel_configs(
            self.configurations, run_de_config, self.csv_filepath, CSV_SCHEMA,
            group_columns=[
                "case_id", "dataset_id", "popsize", "max_generations",
                "decoder_name", "evaluation_model", "fitness_name", "alpha", "cv_folds",
            ],
            max_workers=self.max_workers,
        )

    def run_profile_config(
        self,
        config_index=0,
        max_generations=None,
        popsize=None,
    ):
        if not self.configurations:
            raise ValueError("No DE configurations are available to profile.")
        if not 0 <= config_index < len(self.configurations):
            raise IndexError(
                f"Configuration index {config_index} is out of range "
                f"(0..{len(self.configurations) - 1})."
            )

        config = dict(self.configurations[config_index])
        if max_generations is not None:
            config["max_generations"] = max_generations
        if popsize is not None:
            config["popsize"] = popsize
        print(
            "Profiling DE configuration "
            f"{config_index}: dataset={config['dataset_id']} "
            f"case={config['case_id']} model={config['evaluation_model']}"
        )
        return run_de_config(config)



if __name__ == "__main__":
    runner = DEParameters()
    runner.run_all()
