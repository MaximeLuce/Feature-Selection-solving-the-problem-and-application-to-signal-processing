import numpy as np

from app.Utilities.ConfigLoader import load_config
from app.particle_swarm.algorithm import NewBinaryParticleSwarmOptimization
from app.runners.common import (
    aggregate_feature_selection_group,
    append_csv_rows,
    build_evaluation_cases,
    build_row_from_schema,
    build_run_configs,
    print_feature_selection_summary,
    run_grouped_configs,
    validate_and_order_row,
)

TIMING_METRICS = ["elapsed_time_ns", "cpu_time_ns"]

NBPSO_CSV_SCHEMA = [
    {"column": "Case_ID", "key": "case_id"},
    {"column": "Dataset_ID", "key": "dataset_id"},
    {"column": "Swarm_Size", "key": "swarm_size"},
    {"column": "Generations_number", "key": "max_generations"},
    {"column": "Seed", "key": "base_seed"},
    {"column": "Evaluator", "key": "evaluator"},
    {"column": "Fitness", "key": "fitness_name"},
    {"column": "Alpha", "key": "alpha"},
    {"column": "CV_Folds", "key": "cv_folds"},
    {"column": "NBPSO_Best", "key": "nbpso_b", "digits": 1},
    {"column": "NBPSO_Worst", "key": "nbpso_w", "digits": 1},
    {"column": "NBPSO_Avg", "key": "nbpso_a", "digits": 2},
    {"column": "NBPSO_Std", "key": "nbpso_s", "digits": 2},
    {"column": "NBPSO_Wall_Time_Avg(s)", "key": "wall_avg_s", "digits": 3},
    {"column": "NBPSO_CPU_Time_Avg(s)", "key": "cpu_avg_s", "digits": 3},
    {"column": "NBPSO_NFE_Avg", "key": "nbpso_nfe_avg", "digits": 1},
    {"column": "Champion_Score", "key": "champion_score", "digits": 2},
    {"column": "Nb_Features_Keep", "key": "nb_features_keep"},
]

NBPSO_CSV_COLUMNS = [item["column"] for item in NBPSO_CSV_SCHEMA]

NBPSO_RESUME_COLUMNS = [
    "Case_ID",
    "Dataset_ID",
    "Evaluator",
    "Fitness",
    "Alpha",
    "CV_Folds",
]


def run_nbpso_config(config):
    print(
        f"\n--- RUNNING NBPSO CASE {config['case_id']} / "
        f"{config['evaluation_config']['evaluation_model']} ---"
    )

    dataset_id = config["dataset_id"]
    evaluation_config = dict(config["evaluation_config"])
    evaluation_config["seed"] = config["seed"]
    problem = Problem.load_dataset(dataset_id, evaluation_config)
    print(
        f"Problem loaded: ID={dataset_id} with {problem.num_features} "
        f"features and {problem.num_instances} instances."
    )

    nbpso_config = dict(config)
    monitoring = dict(nbpso_config.get("monitoring", {}))
    monitoring_metrics = list(monitoring.get("metrics", []))
    for metric in TIMING_METRICS:
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

    best_mask = np.asarray(result.best, dtype=int)
    final_score = float(problem.evaluate_final(best_mask) * 100)

    return {
        "order": config["order"],
        "run_id": config["run_id"],
        "case_id": config["case_id"],
        "dataset_id": dataset_id,
        "swarm_size": config["swarm_size"],
        "max_generations": config["max_generations"],
        "base_seed": config["base_seed"],
        "seed": config["seed"],
        "evaluation_model": evaluation_config["evaluation_model"],
        "fitness_name": evaluation_config["fitness"],
        "alpha": evaluation_config["alpha"],
        "cv_folds": evaluation_config["cv_folds"],
        "best_vector": np.asarray(result.best, dtype=int),
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
        self.evaluation_cases = build_evaluation_cases(config)
        self.csv_filepath = config.get(
            "csv_filepath_nbpso",
            "app/Results/SAParameters/NBPSOParameters.csv",
        )

        if not self.cases:
            print("No NBPSO cases config has been loaded.")

    def _build_csv_row(self, values):
        row = build_row_from_schema(values, NBPSO_CSV_SCHEMA)
        return validate_and_order_row(row, NBPSO_CSV_COLUMNS)

    def _print_summary(self, row):
        print_feature_selection_summary(
            row,
            ("NBPSO_Best", "NBPSO_Worst", "NBPSO_Avg", "NBPSO_Std"),
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
        first = group[0]
        stats = aggregate_feature_selection_group(group)
        first = stats["first"]

        return self._build_csv_row({
            "case_id": first["case_id"],
            "dataset_id": stats["dataset_id"],
            "swarm_size": int(first["swarm_size"]),
            "max_generations": int(first["max_generations"]),
            "base_seed": int(first["base_seed"]),
            "evaluator": first["evaluation_model"],
            "fitness_name": first["fitness_name"],
            "alpha": first["alpha"],
            "cv_folds": int(first["cv_folds"]),
            "nbpso_b": stats["score_best"],
            "nbpso_w": stats["score_worst"],
            "nbpso_a": stats["score_avg"],
            "nbpso_s": stats["score_std"],
            "wall_avg_s": stats["wall_avg_s"],
            "cpu_avg_s": stats["cpu_avg_s"],
            "nbpso_nfe_avg": stats["nfe_avg"],
            "champion_score": stats["champion_score"],
            "nb_features_keep": stats["nb_features_keep"],
        })

    def run_all(self):
        print(f"Start running... Results will be saved to {self.csv_filepath}")

        all_configs = list(build_run_configs(
            self.dataset_ids,
            self.cases,
            self.evaluation_cases,
            self.runs_per_algo,
            lambda case: dict(case),
        ))
        if not all_configs:
            return

        run_grouped_configs(
            configurations=all_configs,
            group_keys_fn=lambda config: (
                config["case_id"],
                str(config["dataset_id"]),
                config["evaluation_config"]["evaluation_model"],
                config["evaluation_config"]["fitness"],
                config["evaluation_config"]["alpha"],
                config["evaluation_config"]["cv_folds"],
            ),
            worker=run_nbpso_config,
            aggregate_fn=self._aggregate_group,
            write_fn=lambda rows: append_csv_rows(
                self.csv_filepath,
                NBPSO_CSV_COLUMNS,
                rows,
            ),
            print_fn=self._print_summary,
            resume_columns=NBPSO_RESUME_COLUMNS,
            csv_filepath=self.csv_filepath,
            description="NBPSO runs",
        )


if __name__ == "__main__":
    NBPSOParameters().run_all()
