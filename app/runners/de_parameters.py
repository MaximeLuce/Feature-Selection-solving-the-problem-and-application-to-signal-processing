import random

import numpy as np

from app.decoders import build_decoder
from app.differential_evolution.algorithm import DifferentialEvolution
from app.differential_evolution.featureselection import FeatureSelectionProblem
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
    run_grouped_configs,
    run_parallel_configs,
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
    {"column": "Evaluator", "key": "evaluator"},
    {"column": "Fitness", "key": "fitness_name"},
    {"column": "Alpha", "key": "alpha"},
    {"column": "CV_Folds", "key": "cv_folds"},
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
        "order": config["order"],
        "run_id": config["run_id"],
        "case_id": config["case_id"],
        "dataset_id": dataset_id,
        "popsize": config["popsize"],
        "max_generations": config["max_generations"],
        "base_seed": config["base_seed"],
        "seed": config["seed"],
        "sample_index": config["sample_index"],
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
        config = load_config()
        self.dataset_ids = config.get("dataset_ids", [0])
        self.runs_per_algo = config.get("runs_per_algo", 10)
        self.cases = config.get("cases_DE_test", [])
        self.evaluation_cases = build_evaluation_cases(config)
        self.csv_filepath = config.get(
            "csv_filepath",
            "app/Results/SAParameters/DEParameters_test2.csv",
        )

        if not self.cases:
            print("No DE cases config has been loaded.")

    def _build_csv_row(self, values):
        row = build_row_from_schema(values, DE_CSV_SCHEMA)
        return validate_and_order_row(row, DE_CSV_COLUMNS)

    def _print_summary(self, row):
        print_feature_selection_summary(
            row,
            ("DE_Best", "DE_Worst", "DE_Avg", "DE_Std"),
            [
                ("Dataset", "Dataset_ID"),
                ("Case", "Case_ID"),
                ("Decoder", "Decoder"),
                ("Evaluator", "Evaluator"),
                ("Fitness", "Fitness"),
                ("alpha=", "Alpha"),
                ("cv=", "CV_Folds"),
            ],
        )

    def _aggregate_group(self, _group_key, group):
        first = group[0]
        stats = aggregate_feature_selection_group(group)
        first = stats["first"]

        return self._build_csv_row({
            "case_id": first["case_id"],
            "dataset_id": stats["dataset_id"],
            "popsize": int(first["popsize"]),
            "max_generations": int(first["max_generations"]),
            "base_seed": int(first["base_seed"]),
            "decoder_name": first["decoder_name"],
            "evaluator": first["evaluation_model"],
            "fitness_name": first["fitness_name"],
            "alpha": first["alpha"],
            "cv_folds": int(first["cv_folds"]),
            "de_b": stats["score_best"],
            "de_w": stats["score_worst"],
            "de_a": stats["score_avg"],
            "de_s": stats["score_std"],
            "wall_avg_s": stats["wall_avg_s"],
            "cpu_avg_s": stats["cpu_avg_s"],
            "de_nfe_avg": stats["nfe_avg"],
            "champion_score": stats["champion_score"],
            "nb_features_keep": stats["nb_features_keep"],
        })

    def run_all_grouped(self):
        print(f"Start running... Results will be saved to {self.csv_filepath}")

        run_grouped_configs(
            configurations=list(build_run_configs(
                self.dataset_ids,
                self.cases,
                self.evaluation_cases,
                self.runs_per_algo,
                lambda case: {
                    **dict(case),
                    "decoder": case["decoder"],
                },
            )),
            group_keys_fn=lambda config: (
                config["case_id"],
                str(config["dataset_id"]),
                config["decoder"]["name"],
                config["evaluation_config"]["evaluation_model"],
                config["evaluation_config"]["fitness"],
                config["evaluation_config"]["alpha"],
                config["evaluation_config"]["cv_folds"],
            ),
            worker=run_de_config,
            aggregate_fn=self._aggregate_group,
            write_fn=lambda rows: append_csv_rows(
                self.csv_filepath,
                DE_CSV_COLUMNS,
                rows,
            ),
            print_fn=self._print_summary,
            description="DE groups",
        )

    def run_all(self):
        print(f"Start running... Results will be saved to {self.csv_filepath}")

        configurations = list(build_run_configs(
            self.dataset_ids,
            self.cases,
            self.evaluation_cases,
            self.runs_per_algo,
            lambda case: {
                **dict(case),
                "decoder": case["decoder"],
            },
        ))
        if not configurations:
            return

        random.shuffle(configurations)
        results = run_parallel_configs(
            configurations,
            run_de_config,
            description="DE runs",
        )

        rows = aggregate_grouped_results(
            results,
            lambda result: (
                result["case_id"],
                str(result["dataset_id"]),
                result["decoder_name"],
                result["evaluation_model"],
                result["fitness_name"],
                result["alpha"],
                result["cv_folds"],
            ),
            self._aggregate_group,
        )
        for row in rows:
            self._print_summary(row)
        append_csv_rows(self.csv_filepath, DE_CSV_COLUMNS, rows)


if __name__ == "__main__":
    DEParameters().run_all_grouped()
