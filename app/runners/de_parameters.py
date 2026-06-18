# app/runners/de_parameters.py
import os
import numpy as np

from app.Archive.monitoring import MonitoringMetric
from app.decoders import build_decoder
from app.differential_evolution.algorithm import DifferentialEvolution
from app.differential_evolution.featureselection import DecodedFeatureSelectionProblem
from app.Problem import Problem
from app.runners.common import (
    build_rebuilt_csv_path,
    build_run_configs,
    rebuild_grouped_csv_from_raw,
    run_configs,
    save_raw_run_result,
)
from app.utilities.config_loader import load_config

DE_CSV_SCHEMA = [
    {"column": "Case_ID", "key": "case_id"},
    {"column": "Dataset_ID", "key": "dataset_id"},
    {"column": "Strategy", "key": "strategy"},
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

DE_GROUP_FIELDS = [
    "case_id",
    "dataset_id",
    "strategy",
    "popsize",
    "max_generations",
    "decoder_name",
    "evaluation_model",
    "fitness_name",
    "alpha",
    "cv_folds",
]

TIMING_METRICS = [
    MonitoringMetric.ELAPSED_TIME_NS.value,
    MonitoringMetric.CPU_TIME_NS.value,
]
RAW_MONITORING_METRICS = [
    MonitoringMetric.BEST_MASK.value,
    MonitoringMetric.WORST_MASK.value,
    MonitoringMetric.POPULATION_DIVERSITY.value,
    MonitoringMetric.BEST_FITNESS.value,
    MonitoringMetric.EVALUATION_COUNT.value,
]
RAW_DIR = "app/Results/raw"
RAW_FILENAME = "cases_BDE_AMDE_Table5_Table8_PopulationGenerationSweep.jsonl"


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
    decoder = build_decoder(decoder_config)
    de_problem = DecodedFeatureSelectionProblem(problem, decoder)

    de_config = dict(config)
    monitoring = dict(de_config.get("monitoring", {}))
    monitoring_metrics = list(monitoring.get("metrics", []))
    for metric in TIMING_METRICS + RAW_MONITORING_METRICS:
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

    best_mask = np.asarray(result.best_mask, dtype=int)
    final_score = float(problem.evaluate_final(best_mask) * 100)

    raw_data = {
        "config": {
            "case_id": config["case_id"],
            "dataset_id": dataset_id,
            "popsize": config["popsize"],
            "strategy": config["strategy"],
            "max_generations": config["max_generations"],
            "base_seed": config["base_seed"],
            "seed": config["seed"],
            "decoder_name": decoder.name,
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
        "best_vector": np.asarray(result.best_vector, dtype=float),
        "best_fitness": float(result.best_fitness),
        "final_score": final_score,
        "nb_features_keep": int(best_mask.sum()),
        "elapsed_wall_s": float(result.wall_ns / 1_000_000_000),
        "elapsed_cpu_s": float(result.cpu_ns / 1_000_000_000),
        "evaluations_count": result.evaluations,
    }


class DEParameters:
    def __init__(self):
        self.config = load_config()
        self.dataset_ids = self.config.get("dataset_ids", [0])
        self.runs_per_algo = self.config.get("runs_per_algo", 10)
        self.max_workers = self.config.get(
            "de_max_workers",
            max(1, min(14, os.cpu_count() or 14)),
        )
        self.cases = self.config.get(
            "cases_BDE_AMDE_Table5_Table8_PopulationGenerationSweep", []
        )

        self.evaluation_cases = self.config.get(
            "evaluation_cases",
            [
                {
                    "evaluation_model": "svm",
                    "fitness": "weighted_error",
                    "alpha": 0.5,
                    "cv_folds": 5,
                }
            ],
        )

        self.csv_filepath = "app/Results/SAParameters/BDE_AMDE_Table5_Table8_PopulationGenerationSweep_RE_RUN.csv"

        # .dataset_ids = [174]
        # self.cases = [self.cases[6]]
        # self.cases[0]["max_generations"] = 10
        # self.runs_per_algo = 2
        # self.csv_filepath = "app/Results/SAParameters/test.csv"

        self.configurations = list(
            build_run_configs(
                self.dataset_ids, self.cases, self.evaluation_cases, self.runs_per_algo
            )
        )

    def run_all(self, parallel=True):
        run_configs(
            self.configurations,
            run_de_config,
            self.csv_filepath,
            DE_CSV_SCHEMA,
            group_fields=DE_GROUP_FIELDS,
            parallel=parallel,
            max_workers=self.max_workers,
        )


if __name__ == "__main__":
    runner = DEParameters()
    runner.run_all(parallel=True)
